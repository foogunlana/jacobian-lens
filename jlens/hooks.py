# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch import nn


class ActivationRecorder:
    """Captures residual-stream tensors at the given block indices.

    Registers a forward hook on each requested block on ``__enter__`` and
    removes them on ``__exit__``. On the next forward pass each block's output
    is stored in :attr:`activations`, keyed by block index. Stored tensors are
    not detached, so they can be passed straight to :func:`torch.autograd.grad`.

    Args:
        blocks: The sequence of residual blocks (e.g. ``model.layers``).
        at: Block indices to record at.
        start_graph_at: If given, the captured tensor at this index is marked
            ``requires_grad_(True)`` before downstream blocks see it. When the
            model's parameters all have ``requires_grad=False``, this makes the
            captured residual the leaf that roots the autograd graph, so the
            retained graph spans only this block onward.
        cut_every_block: If ``True``, hook every block (``at`` is ignored) and
            cut the residual stream after each one: the block's output is
            stored in :attr:`activations` (with its graph) and replaced by a
            detached copy that requires grad, stored in :attr:`leaves`, which
            the next block receives. Values are unchanged, but each block's
            graph now runs only from ``leaves[k - 1]`` to ``activations[k]``,
            so gradients give single-block Jacobians. Can't be combined with
            ``start_graph_at``.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
        cut_every_block: bool = False,
    ) -> None:
        if cut_every_block and start_graph_at is not None:
            raise ValueError("cut_every_block makes every block a graph root; don't pass start_graph_at")
        self._blocks = blocks
        self._cut_every_block = cut_every_block
        self._indices = list(range(len(blocks))) if cut_every_block else sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self.leaves: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., Any]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> Any:
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if self._cut_every_block:
                # Returning a value replaces the block's output: the next block
                # computes on the leaf, so the graph stops here.
                leaf = tensor.detach().requires_grad_(True)
                self.activations[index] = tensor
                self.leaves[index] = leaf
                return leaf if torch.is_tensor(output) else (leaf, *output[1:])
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
