# SPDX-License-Identifier: Apache-2.0
"""Tests for the layerwise fit: each block's own Jacobian, A_l = dh_{l+1} / dh_l.

How it works: `ActivationRecorder(cut_every_block=True)` replaces every block's
output with a detached copy, so the next block starts a fresh autograd graph.
One backward pass from all block outputs then gives every A_l separately.

Two small CPU models, built here (no downloads, a few seconds in total):

  TinyDecoder  blocks are h + W h (no attention), so A_l = I + W_{l+1} exactly.
  Tiny Qwen3   a randomly initialised Qwen3 with real attention and norms.

Six checks:
  1. hops on TinyDecoder are I + W                     (exact answer)
  2. hops match jlens fitted one hop at a time         (same estimator as J)
  3. multiply-then-collapse is J; collapse-then-multiply is not
  4. fit() averages the hops over prompts
  5. fit() won't resume a J checkpoint as hops, or vice versa
  6. arguments that make no sense with layerwise raise

Run:
    pip install -e ".[dev]"
    pytest tests/test_layerwise.py -v
"""

import pytest
import torch
import transformers

from jlens import fit, from_hf
from jlens.fitting import jacobian_for_prompt, valid_position_mask
from jlens.hooks import ActivationRecorder

from .tiny import TinyDecoder, _ByteTokenizer

PROMPT = "the quick brown fox jumps over the lazy dog " * 2
SETTINGS = dict(dim_batch=4, max_seq_len=64)


# ---------- models


def tiny_decoder(n_layers=4):
    """Blocks are h + W h. Frozen, as from_hf does for real models."""
    model = TinyDecoder(n_layers=n_layers, d_model=8)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def tiny_qwen(n_layers=3):
    """A random 3-layer Qwen3 (d_model = 16), weights scaled up so blocks aren't ~identity."""
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=n_layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    hf = transformers.Qwen3ForCausalLM(config)
    with torch.no_grad():
        for p in hf.model.layers.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    return from_hf(hf, _ByteTokenizer())


def hops(model, prompt=PROMPT):
    """A_l for every l, from the layerwise fit on one prompt."""
    layers = list(range(model.n_layers - 1))
    return jacobian_for_prompt(model, prompt, layers, layerwise=True, **SETTINGS)[0]


def jacobians(model, prompt=PROMPT):
    """The regular J_l (to the last layer) for every l, on one prompt."""
    layers = list(range(model.n_layers - 1))
    return jacobian_for_prompt(model, prompt, layers, **SETTINGS)[0]


# ---------- the hops are right


def test_hops_on_tiny_decoder_are_identity_plus_weight():
    """Block l+1 computes h + W h, so its Jacobian is I + W.

    Also catches gradients leaking through the cut (you'd get products of
    several blocks) and off-by-one or transposed indexing.
    """
    model = tiny_decoder()
    A = hops(model)
    for l in A:
        expected = torch.eye(8) + model.layers[l + 1].linear.weight
        torch.testing.assert_close(A[l], expected, rtol=0, atol=1e-5)


def test_hops_match_jlens_fitted_one_hop_at_a_time():
    """One layerwise pass == separate regular fits with target_layer = l + 1, bit for bit.

    With attention, so the sum over later positions is exercised. Also shows the
    cut leaves the forward pass unchanged, or the numbers would differ.
    """
    model = tiny_qwen()
    A = hops(model)
    for l in A:
        one_hop = jacobian_for_prompt(model, PROMPT, [l], target_layer=l + 1, **SETTINGS)[0]
        assert torch.equal(A[l], one_hop[l])


# ---------- multiplying hops back together


def test_multiply_then_collapse_is_j_but_collapse_then_multiply_is_not():
    """The chain rule is exact per (position, position) entry, not for the averages.

    jlens collapses a full Jacobian F[target pos, i, source pos, j] by summing over
    valid target positions and averaging over valid source positions. Here we
    build the full Jacobians of blocks 1 and 2 by brute force and check:
      collapse(F_0)          == A_0   (the hop is the collapsed block Jacobian)
      collapse(F_1 @ F_0)    == J_0   (multiply first, then collapse: exact)
      A_1 @ A_0              != J_0   (collapse first, then multiply: not exact)
    """
    model = tiny_qwen()
    ids = model.encode(PROMPT, max_length=SETTINGS["max_seq_len"])
    seq_len, d = ids.shape[1], model.d_model
    valid = valid_position_mask(seq_len)

    def collapse(full):
        F = full.reshape(seq_len, d, seq_len, d)[valid][:, :, valid]
        return F.sum(0).mean(1)

    with ActivationRecorder(model.layers, at=[], cut_every_block=True) as rec, torch.enable_grad():
        model.forward(ids)
        full = {}
        for l in [0, 1]:
            out = rec.activations[l + 1][0].reshape(-1)
            n = out.numel()
            rows = torch.autograd.grad(out, rec.leaves[l], torch.eye(n), is_grads_batched=True,
                                       retain_graph=True)[0]
            full[l] = rows.reshape(n, n)

    A, J = hops(model), jacobians(model)
    torch.testing.assert_close(collapse(full[0]), A[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(collapse(full[1] @ full[0]), J[0], rtol=1e-4, atol=1e-5)
    gap = (A[1] @ A[0] - J[0]).norm() / J[0].norm()
    assert gap > 0.05


# ---------- fit() and its guards


def test_fit_averages_the_hops_over_prompts():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit(model, prompts, layerwise=True, **SETTINGS)
    per_prompt = [hops(model, p) for p in prompts]
    for l in [0, 1, 2]:
        mean = (per_prompt[0][l] + per_prompt[1][l]) / 2
        torch.testing.assert_close(lens.jacobians[l], mean)


def test_fit_refuses_to_resume_the_other_kind_of_checkpoint(tmp_path):
    """A J checkpoint must not be resumed as hops, or the other way round.

    Checkpoints from before this option have no "layerwise" key and count as J.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5]
    regular, layerwise = str(tmp_path / "J.pt"), str(tmp_path / "A.pt")
    fit(model, prompts, checkpoint_path=regular, **SETTINGS)
    fit(model, prompts, checkpoint_path=layerwise, layerwise=True, **SETTINGS)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, checkpoint_path=regular, layerwise=True, **SETTINGS)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, checkpoint_path=layerwise, **SETTINGS)

    old = torch.load(regular, weights_only=True)
    del old["layerwise"]
    torch.save(old, regular)
    with pytest.raises(ValueError, match="layerwise"):
        fit(model, prompts, checkpoint_path=regular, layerwise=True, **SETTINGS)


def test_arguments_that_make_no_sense_with_layerwise_raise():
    with pytest.raises(ValueError, match="layerwise"):
        fit(tiny_decoder(), [PROMPT], target_layer=2, layerwise=True)
    with pytest.raises(ValueError, match="start_graph_at"):
        ActivationRecorder(tiny_decoder().layers, at=[], start_graph_at=0, cut_every_block=True)
