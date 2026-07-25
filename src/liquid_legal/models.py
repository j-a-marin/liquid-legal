"""Liquid neural network models for case trajectory prediction.

The :class:`CaseTrajectoryEngine` wraps an ncps CfC or LTC recurrent cell and
adds three per-timestep heads:

* settlement logit (will the case settle within the horizon?),
* log recovery (expected final recovery, log1p dollars),
* log remaining time (expected days to resolution, log1p days).

Irregular docket timing is handled natively by passing inter-arrival times as
``timespans`` to the CfC/LTC cell — this is the core advantage of liquid
networks for legal event streams.
"""

from __future__ import annotations

import torch
from torch import nn
from ncps.wirings import AutoNCP, FullyConnected, Wiring

from .rnn import BatchedCfC, BatchedLTC

from .events import N_EVENT_TYPES, STATIC_DIM
from .featurize import EVENT_FEATURE_DIM


def build_wiring(
    wiring: str | Wiring, units: int, ncp_output_size: int = 16
) -> tuple[Wiring, int]:
    """Build an ncps wiring and return ``(wiring, rnn_output_dim)``."""
    if isinstance(wiring, Wiring):
        out_dim = wiring.output_size if wiring.output_size is not None else wiring.units
        return wiring, out_dim
    if wiring == "ncp":
        # Sparse neural-circuit-policy wiring: fewer synapses, more auditable.
        return AutoNCP(units=units, output_size=ncp_output_size), ncp_output_size
    if wiring in ("fully_connected", "fc"):
        return FullyConnected(units), units
    raise ValueError(f"unknown wiring: {wiring!r}")


class CaseTrajectoryEngine(nn.Module):
    """Liquid-network engine that reads a docket stream and emits trajectories.

    Args:
        units: Number of liquid neurons.
        wiring: ``"ncp"`` for sparse Neural Circuit Policy wiring (auditable),
            ``"fully_connected"`` for a dense recurrent wiring, or an ncps
            ``Wiring`` instance.
        ncp_output_size: Motor-neuron count when ``wiring="ncp"``.
        cell: ``"cfc"`` (closed-form, fast; default) or ``"ltc"`` (full ODE).
        embed_dim: Event-type embedding dimension.
        mode: CfC gating mode (``"default"``, ``"pure"``, or ``"no_gate"``).
        time_scale: Days per unit of liquid time. ``timespans`` (in days) are
            divided by this before reaching the cell, so the learned time
            constants operate on a ~months scale. Default 30.0.
        time_mode: How inter-arrival times reach the model. ``"native"``
            passes them as liquid ``timespans`` *and* keeps the delta feature;
            ``"timespans_only"`` passes ``timespans`` but zeroes the delta
            feature; ``"feature"`` keeps the delta feature but never passes
            ``timespans`` (the discrete-RNN workaround); ``"none"`` drops
            both. Exists for ablations; default ``"native"``.
    """

    TIME_MODES = ("native", "timespans_only", "feature", "none")

    def __init__(
        self,
        units: int = 64,
        wiring: str | Wiring = "ncp",
        ncp_output_size: int = 16,
        cell: str = "cfc",
        embed_dim: int = 16,
        mode: str = "default",
        time_scale: float = 30.0,
        time_mode: str = "native",
        **rnn_kwargs,
    ):
        super().__init__()
        if time_mode not in self.TIME_MODES:
            raise ValueError(f"unknown time_mode: {time_mode!r}")
        self.time_mode = time_mode
        self.time_scale = float(time_scale)
        self.event_embedding = nn.Embedding(N_EVENT_TYPES, embed_dim)
        self.wiring, out_dim = build_wiring(wiring, units, ncp_output_size)
        input_size = embed_dim + EVENT_FEATURE_DIM + STATIC_DIM

        cell = cell.lower()
        if cell == "cfc":
            self.rnn = BatchedCfC(input_size, self.wiring, mode=mode, **rnn_kwargs)
        elif cell == "ltc":
            self.rnn = BatchedLTC(input_size, self.wiring, **rnn_kwargs)
        else:
            raise ValueError(f"unknown cell: {cell!r}")

        self.head_settle = nn.Linear(out_dim, 1)
        self.head_recovery = nn.Linear(out_dim, 1)
        self.head_remaining = nn.Linear(out_dim, 1)

    @property
    def state_size(self) -> int:
        return self.wiring.units

    def forward(
        self,
        event_ids: torch.Tensor,
        event_feats: torch.Tensor,
        static: torch.Tensor,
        timespans: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the engine over a batch.

        Args:
            event_ids: (B, T) long tensor of event type indices.
            event_feats: (B, T, EVENT_FEATURE_DIM) per-event numerics.
            static: (B, STATIC_DIM) static covariates, broadcast over time.
            timespans: (B, T) or (B, T, 1) inter-arrival times in days; this
                is what lets the liquid dynamics handle irregular dockets.
            hx: Optional initial hidden state (B, ``state_size``).

        Returns:
            Dict with per-timestep ``settle_logit``, ``log_recovery``,
            ``log_remaining`` (each (B, T)), plus the final hidden state
            ``hx`` and the full recurrent output ``rnn_out``.
        """
        B, T = event_ids.shape
        emb = self.event_embedding(event_ids)
        if self.time_mode in ("timespans_only", "none"):
            # Remove the delta feature (index 0) without mutating the caller's
            # tensor — batches are reused across epochs.
            event_feats = event_feats.clone()
            event_feats[..., 0] = 0.0
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        x = torch.cat([emb, event_feats, static_exp], dim=-1)

        if timespans is not None and self.time_mode in ("native", "timespans_only"):
            if timespans.dim() == 3:
                timespans = timespans.squeeze(-1)
            timespans = timespans.to(x.dtype) / self.time_scale
            out, hx = self.rnn(x, hx, timespans)
        else:
            out, hx = self.rnn(x, hx)

        return {
            "settle_logit": self.head_settle(out).squeeze(-1),
            "log_recovery": self.head_recovery(out).squeeze(-1),
            "log_remaining": self.head_remaining(out).squeeze(-1),
            "hx": hx,
            "rnn_out": out,
        }
