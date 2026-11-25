# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from jlens.fitting import fit, jacobian_for_prompt, valid_position_mask
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


def test_valid_position_mask_basic():
    mask = valid_position_mask(32, skip_first=4)
    assert mask.dtype == torch.bool
    assert mask[:4].sum() == 0  # leading attention-sinks excluded
    assert not mask[-1]  # final position excluded
    assert mask[4:-1].all()
    assert mask.sum() == 32 - 4 - 1


def test_valid_position_mask_too_short():
    with pytest.raises(ValueError, match="too short"):
        valid_position_mask(5, skip_first=8)


def test_jacobian_for_prompt_tiny():
    """End-to-end on a 4-layer CPU model: shapes + late-layer diag ~= 1.

    Run with all parameters ``requires_grad=False``: the recorder's
    ``start_graph_at`` must root the autograd graph itself.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    prompt = "the quick brown fox " * 4  # > SKIP_FIRST_N_POSITIONS chars
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64
    )
    assert set(jacobians) == {0, 1, 2}
    for J in jacobians.values():
        assert J.shape == (8, 8) and J.dtype == torch.float32
    assert n_valid > 0 and seq_len > n_valid
    # Residual block is h + 0.1*W*h, so J_{n_layers-2} = I + 0.1*W -> diag ~= 1.
    diag_late = jacobians[2].diag()
    assert (diag_late - 1.0).abs().max() < 0.2
    # Earlier layers compound through more blocks -> further from identity.
    assert (jacobians[0] - torch.eye(8)).norm() > (jacobians[2] - torch.eye(8)).norm()
    # Block 3 is h + W_3 h, so J_2 == I + W_3 exactly — pins orientation/indexing.
    expected_J2 = torch.eye(8) + model.layers[3].linear.weight.detach()
    torch.testing.assert_close(jacobians[2], expected_J2, rtol=0, atol=1e-5)


def test_fit_and_apply_tiny(tmp_path):
    """fit() -> JacobianLens -> save/load -> apply() round-trip."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit(model, prompts, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64)
    assert lens.n_prompts == 2
    assert lens.source_layers == [0, 1, 2]
    assert lens.d_model == 8

    path = tmp_path / "lens.pt"
    lens.save(str(path))
    reloaded = JacobianLens.load(str(path))
    assert reloaded.source_layers == [0, 1, 2]
    assert reloaded.n_prompts == 2
    for layer in [0, 1, 2]:
        torch.testing.assert_close(
            reloaded.jacobians[layer], lens.jacobians[layer], rtol=0, atol=2e-3
        )  # fp16 round-trip

    lens_logits, model_logits, input_ids = reloaded.apply(
        model, "the quick brown fox jumps", layers=[0, 2]
    )
    assert set(lens_logits) == {0, 2}
    vocab_size = model.lm_head.out_features
    seq_len = input_ids.shape[1]
    # positions=None -> every position.
    assert model_logits.shape == (seq_len, vocab_size)
    for tensor in lens_logits.values():
        assert tensor.shape == (seq_len, vocab_size)
    # Exactly linear lens map: transported readout == model logits (atol: fp16 save).
    torch.testing.assert_close(lens_logits[2], model_logits, rtol=0, atol=1e-2)

    # Explicit positions (negative indices allowed) -> that many rows, in order.
    sub_logits, sub_model, _ = reloaded.apply(
        model, "the quick brown fox jumps", layers=[0, 2], positions=[0, -1]
    )
    assert sub_model.shape == (2, vocab_size)
    torch.testing.assert_close(sub_model[1], model_logits[-1])
    for layer in [0, 2]:
        assert sub_logits[layer].shape == (2, vocab_size)
        torch.testing.assert_close(sub_logits[layer][0], lens_logits[layer][0])

    # Logit-lens baseline path (use_jacobian=False) also works.
    baseline, _, _ = reloaded.apply(
        model, "hello world test", layers=[1], positions=[-1], use_jacobian=False
    )
    assert baseline[1].shape == (1, vocab_size)
    # Unfitted layer is rejected.
    with pytest.raises(ValueError, match="not in source_layers"):
        reloaded.apply(model, "x" * 30, layers=[3])
    # Out-of-range layers are rejected on the baseline path too.
    with pytest.raises(ValueError, match="out of range"):
        reloaded.apply(model, "x" * 30, layers=[99], use_jacobian=False)


def test_from_pretrained_local(tmp_path):
    """``from_pretrained`` resolves a file, a directory, and a directory with
    the lens at a subpath (Hub-repo style layout)."""
    lens = JacobianLens(
        jacobians={0: torch.randn(6, 6), 1: torch.randn(6, 6)}, n_prompts=3, d_model=6
    )
    # File path -> load() directly.
    single = tmp_path / "single.pt"
    lens.save(str(single))
    for layer in [0, 1]:
        torch.testing.assert_close(
            JacobianLens.from_pretrained(str(single)).jacobians[layer],
            lens.jacobians[layer],
            rtol=0,
            atol=2e-3,
        )  # fp16 round-trip
    # Directory containing lens.pt.
    one_dir = tmp_path / "one"
    one_dir.mkdir()
    lens.save(str(one_dir / "lens.pt"))
    assert JacobianLens.from_pretrained(str(one_dir)).n_prompts == 3
    # filename= may be a subpath inside the directory.
    deep = tmp_path / "hubrepo"
    sub = deep / "gemma-2-27b" / "jlens" / "wikitext"
    sub.mkdir(parents=True)
    lens.save(str(sub / "lens.pt"))
    reloaded = JacobianLens.from_pretrained(
        str(deep), filename="gemma-2-27b/jlens/wikitext/lens.pt"
    )
    assert reloaded.n_prompts == 3 and reloaded.d_model == 6


def test_merge_weighted_mean():
    """merge() is the n_prompts-weighted mean."""
    d_model = 4
    lens_a = JacobianLens(
        jacobians={
            0: torch.full((d_model, d_model), 1.0),
            1: torch.full((d_model, d_model), 2.0),
        },
        n_prompts=2,
        d_model=d_model,
    )
    lens_b = JacobianLens(
        jacobians={
            0: torch.full((d_model, d_model), 4.0),
            1: torch.full((d_model, d_model), 8.0),
        },
        n_prompts=6,
        d_model=d_model,
    )
    merged = JacobianLens.merge([lens_a, lens_b])
    assert merged.n_prompts == 8
    # (1*2 + 4*6) / 8 = 26/8 = 3.25
    torch.testing.assert_close(
        merged.jacobians[0], torch.full((d_model, d_model), 3.25)
    )
    torch.testing.assert_close(merged.jacobians[1], torch.full((d_model, d_model), 6.5))


def test_merge_mismatch_raises():
    a = JacobianLens(jacobians={0: torch.eye(4)}, n_prompts=1, d_model=4)
    b = JacobianLens(jacobians={1: torch.eye(4)}, n_prompts=1, d_model=4)
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.merge([a, b])
    with pytest.raises(ValueError, match="at least one"):
        JacobianLens.merge([])


def test_fit_checkpoint_resume(tmp_path):
    """fit() resumes from a checkpoint and produces the same result."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5, "uvwxyzabcd " * 5]
    checkpoint = str(tmp_path / "ckpt.pt")

    full = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    # Second call with the same checkpoint should resume past all 3 and be a no-op.
    resumed = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == full.n_prompts == 3
    for layer in [0, 2]:
        torch.testing.assert_close(resumed.jacobians[layer], full.jacobians[layer])


def test_fit_resume_after_skip(tmp_path):
    """Resume after a too-short prompt was skipped must not double-count.

    Regression test: a skipped prompt used to desync the success-count from the
    list-position, so the prompt after a skip was processed again on resume.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    long_a = "abcdefghij " * 5
    short = "x"  # tokenizes to 2 tokens -> ValueError -> skip
    long_b = "klmnopqrst " * 5
    prompts = [long_a, short, long_b]
    checkpoint = str(tmp_path / "ckpt.pt")

    reference = fit(model, prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    assert reference.n_prompts == 2  # short was skipped

    fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    resumed = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == 2
    for layer in [0, 2]:
        torch.testing.assert_close(resumed.jacobians[layer], reference.jacobians[layer])


def test_fit_resume_mismatched_source_layers(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5]
    checkpoint = str(tmp_path / "ckpt.pt")
    fit(
        model,
        prompts,
        source_layers=[0, 1],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="source_layers"):
        fit(
            model,
            prompts,
            source_layers=[0, 2],
            dim_batch=4,
            max_seq_len=64,
            checkpoint_path=checkpoint,
        )


def test_negative_layer_indices_normalized():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    jac_neg, _, _ = jacobian_for_prompt(
        model,
        prompt,
        source_layers=[-4, -3],
        target_layer=-1,
        dim_batch=4,
        max_seq_len=64,
    )
    jac_pos, _, _ = jacobian_for_prompt(
        model, prompt, source_layers=[0, 1], target_layer=3, dim_batch=4, max_seq_len=64
    )
    assert set(jac_neg) == {0, 1}
    for layer in (0, 1):
        torch.testing.assert_close(jac_neg[layer], jac_pos[layer])

    lens = fit(model, [prompt], source_layers=[-4], dim_batch=4, max_seq_len=64)
    assert lens.source_layers == [0]


def test_out_of_range_layers_rejected():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    with pytest.raises(ValueError, match="out of range"):
        fit(model, [prompt], source_layers=[0, 7], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="must all be < target_layer"):
        fit(model, [prompt], source_layers=[-1], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="target_layer"):
        jacobian_for_prompt(
            model,
            prompt,
            source_layers=[0],
            target_layer=9,
            dim_batch=4,
            max_seq_len=64,
        )


# ---------- layerwise: single-block Jacobians A_l = dh_{l+1} / dh_l


def _frozen_tiny(n_layers=4, d_model=8):
    model = TinyDecoder(n_layers=n_layers, d_model=d_model)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def test_cut_every_block_keeps_values_and_isolates_blocks():
    from jlens.hooks import ActivationRecorder

    model = _frozen_tiny()
    ids = model.encode("the quick brown fox " * 2)
    plain = model.forward(ids).last_hidden_state
    with ActivationRecorder(model.layers, at=[], cut_every_block=True) as rec, torch.enable_grad():
        cut = model.forward(ids).last_hidden_state
    torch.testing.assert_close(cut, plain, rtol=0, atol=0)
    assert set(rec.activations) == set(rec.leaves) == {0, 1, 2, 3}
    leaves = [rec.leaves[i] for i in range(4)]
    grads = torch.autograd.grad(rec.activations[2].sum(), leaves, allow_unused=True)
    assert [g is not None for g in grads] == [False, True, False, False]


def test_cut_every_block_with_start_graph_at_raises():
    from jlens.hooks import ActivationRecorder

    with pytest.raises(ValueError, match="start_graph_at"):
        ActivationRecorder(_frozen_tiny().layers, at=[], start_graph_at=0, cut_every_block=True)


def test_layerwise_is_each_blocks_own_jacobian():
    """Block l+1 is h + W h (no attention), so A_l == I + W_{l+1} exactly."""
    model = _frozen_tiny()
    A, _, _ = jacobian_for_prompt(
        model, "the quick brown fox " * 4, source_layers=[0, 1, 2], dim_batch=4,
        max_seq_len=64, layerwise=True,
    )
    assert set(A) == {0, 1, 2}
    for layer in [0, 1, 2]:
        expected = torch.eye(8) + model.layers[layer + 1].linear.weight
        torch.testing.assert_close(A[layer], expected, rtol=0, atol=1e-5)


def test_layerwise_matches_separate_single_hop_fits():
    model = _frozen_tiny()
    prompt = "the quick brown fox " * 4
    A, _, _ = jacobian_for_prompt(model, prompt, [0, 1, 2], dim_batch=4, max_seq_len=64, layerwise=True)
    for layer in [0, 1, 2]:
        ref, _, _ = jacobian_for_prompt(
            model, prompt, [layer], target_layer=layer + 1, dim_batch=4, max_seq_len=64
        )
        torch.testing.assert_close(A[layer], ref[layer], rtol=0, atol=0)


def test_layerwise_with_target_layer_raises():
    model = _frozen_tiny()
    with pytest.raises(ValueError, match="layerwise"):
        jacobian_for_prompt(model, "abcdefghij " * 5, [0], target_layer=2, layerwise=True)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, ["abcdefghij " * 5], target_layer=2, layerwise=True)


def test_fit_layerwise_is_the_mean_of_per_prompt_hops(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit(model, prompts, dim_batch=4, max_seq_len=64, layerwise=True,
               checkpoint_path=str(tmp_path / "hops.pt"))
    assert lens.n_prompts == 2 and lens.source_layers == [0, 1, 2]
    per = [jacobian_for_prompt(model, p, [0, 1, 2], dim_batch=4, max_seq_len=64, layerwise=True)[0]
           for p in prompts]
    for layer in [0, 1, 2]:
        torch.testing.assert_close(lens.jacobians[layer], (per[0][layer] + per[1][layer]) / 2)


def test_fit_resume_refuses_the_other_kind_of_checkpoint(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5]
    j_ckpt, a_ckpt = str(tmp_path / "J.pt"), str(tmp_path / "A.pt")
    fit(model, prompts, dim_batch=4, max_seq_len=64, checkpoint_path=j_ckpt)
    fit(model, prompts, dim_batch=4, max_seq_len=64, checkpoint_path=a_ckpt, layerwise=True)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, dim_batch=4, max_seq_len=64, checkpoint_path=j_ckpt, layerwise=True)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, dim_batch=4, max_seq_len=64, checkpoint_path=a_ckpt)
    # and a checkpoint without the key (older jlens) counts as an ordinary J fit
    state = torch.load(j_ckpt, weights_only=True)
    del state["layerwise"]
    torch.save(state, j_ckpt)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, dim_batch=4, max_seq_len=64, checkpoint_path=j_ckpt, layerwise=True)
