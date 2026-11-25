# SPDX-License-Identifier: Apache-2.0
"""Layerwise (single-block) Jacobians on a real architecture with attention.

Uses a tiny randomly initialised Qwen3 (no download, CPU, seconds), so the
cross-position sum in the estimator is exercised, which the attention-free
TinyDecoder can't do.

The chain rule is exact per token and position pair, so multiplying the full
per-position block Jacobians and then collapsing positions gives exactly the
regular fit. Multiplying the collapsed hops A_l does not, and that gap is what
these tests pin down.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from jlens import from_hf  # noqa: E402
from jlens.fitting import jacobian_for_prompt, valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402

from .tiny import _ByteTokenizer  # noqa: E402

N_LAYERS, D, PROMPT, SKIP = 3, 16, "the quick brown fox jumps over", 4
KW = dict(dim_batch=8, max_seq_len=32, skip_first=SKIP)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=64, hidden_size=D, intermediate_size=32, num_hidden_layers=N_LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    hf = transformers.Qwen3ForCausalLM(cfg).float()
    # random init is close to identity blocks; scale up so blocks matter
    with torch.no_grad():
        for p in hf.model.layers.parameters():
            if p.dim() == 2:
                p.mul_(8.0)
    tok = _ByteTokenizer()
    return from_hf(hf, tok)


def full_block_jacobians(model, prompt):
    """Per prompt, the full [seq*d, seq*d] Jacobian of every block, by brute force."""
    ids = model.encode(prompt, max_length=KW["max_seq_len"])
    with ActivationRecorder(model.layers, at=[], cut_every_block=True) as rec, torch.enable_grad():
        model.forward(ids)
        full = {}
        for l in range(N_LAYERS - 1):
            out, leaf = rec.activations[l + 1][0], rec.leaves[l]
            n = out.numel()
            rows = torch.autograd.grad(
                out.reshape(-1), leaf, grad_outputs=torch.eye(n), is_grads_batched=True,
                retain_graph=True,
            )[0]  # [n, 1, seq, d]
            full[l] = rows.reshape(n, n)
    return full, ids.shape[1]


def collapse(full, seq_len):
    """The jlens estimator: sum over valid target positions, mean over valid source positions."""
    valid = valid_position_mask(seq_len, skip_first=SKIP)
    F = full.reshape(seq_len, D, seq_len, D)[valid][:, :, valid]  # [tgt, i, src, j]
    return F.sum(0).mean(1)


def test_layerwise_matches_separate_single_hop_fits(model):
    A, _, _ = jacobian_for_prompt(model, PROMPT, [0, 1], layerwise=True, **KW)
    for l in [0, 1]:
        ref, _, _ = jacobian_for_prompt(model, PROMPT, [l], target_layer=l + 1, **KW)
        torch.testing.assert_close(A[l], ref[l], rtol=0, atol=0)


def test_hops_are_the_collapsed_full_block_jacobians(model):
    full, seq_len = full_block_jacobians(model, PROMPT)
    A, _, _ = jacobian_for_prompt(model, PROMPT, [0, 1], layerwise=True, **KW)
    for l in [0, 1]:
        torch.testing.assert_close(A[l], collapse(full[l], seq_len), rtol=1e-4, atol=1e-5)


def test_product_before_collapsing_is_exactly_the_regular_fit(model):
    """Chain rule over positions: multiply full block Jacobians, then collapse == J."""
    full, seq_len = full_block_jacobians(model, PROMPT)
    J, _, _ = jacobian_for_prompt(model, PROMPT, [0], **KW)
    torch.testing.assert_close(collapse(full[1] @ full[0], seq_len), J[0], rtol=1e-4, atol=1e-5)


def test_product_after_collapsing_is_not_the_regular_fit(model):
    """Collapsing positions first loses the cross-position paths: A_1 @ A_0 != J_0."""
    J, _, _ = jacobian_for_prompt(model, PROMPT, [0], **KW)
    A, _, _ = jacobian_for_prompt(model, PROMPT, [0, 1], layerwise=True, **KW)
    gap = ((A[1] @ A[0] - J[0]).norm() / J[0].norm()).item()
    assert gap > 1e-3, gap
