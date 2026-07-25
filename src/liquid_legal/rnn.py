"""Batched-timespan variants of the ncps CfC/LTC recurrent layers.

ncps 1.0.1 slices per-step timespans with ``timespans[:, t].squeeze()``,
which collapses the trailing dimension the cells need: inside
``CfCCell.forward`` the interpolation gate computes ``t_a * ts`` with
``t_a`` of shape (B, units), so ``ts`` must broadcast as (B, 1). With
``.squeeze()`` any batch size > 1 produces shape (B,) and raises a
broadcasting error. These subclasses restore the intended behavior; the
forward loops are adapted from ncps (Apache License 2.0) with the minimal
slicing change.

See: https://github.com/mlech26l/ncps
"""

from __future__ import annotations

import torch
from ncps.torch import CfC, LTC


class _TimespanFixMixin:
    """Shared forward loop with correct (B, 1) per-step timespan slicing."""

    def forward(self, input, hx=None, timespans=None):
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if timespans is not None:
            if timespans.dim() == 3 and timespans.size(-1) == 1:
                timespans = timespans.squeeze(-1)
            timespans = timespans.to(input.dtype)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=input.device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=input.device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "mixed_memory=True requires a tuple (h0, c0), got Tensor"
                )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if not is_batched:
                h_state = h_state.unsqueeze(0)
                c_state = c_state.unsqueeze(0) if c_state is not None else None

        fc = getattr(self, "fc", None)
        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].reshape(batch_size, 1)
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].reshape(batch_size, 1)

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            if self.return_sequences:
                output_sequence.append(fc(h_out) if fc is not None else h_out)

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = fc(h_out) if fc is not None else h_out
        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]
        return readout, hx


class BatchedCfC(_TimespanFixMixin, CfC):
    """ncps CfC that accepts per-batch irregular timespans of shape (B, T)."""


class BatchedLTC(_TimespanFixMixin, LTC):
    """ncps LTC that accepts per-batch irregular timespans of shape (B, T)."""
