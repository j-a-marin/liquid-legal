"""Discrete-time and attention baselines for benchmarking against the liquid models.

All baselines expose the same forward interface as
:class:`liquid_legal.models.CaseTrajectoryEngine` so the training loop is
model-agnostic:

* :class:`LSTMTrajectoryModel` — discrete recurrent state; inter-arrival
  times concatenated as an input feature (the standard irregular-series
  workaround for RNNs).
* :class:`TemporalTransformerModel` — compact causal-masked Transformer
  encoder; attention over events with sinusoidal elapsed-time encodings,
  sized comparably to the recurrent models. Batches are right-padded, so the
  causal mask also makes padding invisible: real positions can never attend
  to padded ones.

Both accept ``time_mode`` so the ablation matrix is symmetric with the liquid
engine: ``"native"`` (time encodings + delta feature), ``"timespans_only"``
(encodings only), ``"feature"`` (delta feature only), ``"none"``. For the
LSTM, which has no time encodings, only ``"feature"`` and ``"none"`` apply.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .events import N_EVENT_TYPES, STATIC_DIM
from .featurize import EVENT_FEATURE_DIM


def _sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of a (B, T) non-negative signal -> (B, T, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half
    )
    args = torch.log1p(x.clamp(min=0)).unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def nonneg_non_crossing(raw: torch.Tensor) -> torch.Tensor:
    """3 quantiles (levels .1/.5/.9), nonnegative and non-crossing by
    construction: softplus base, cumulative softplus increments."""
    q1 = nn.functional.softplus(raw[..., :1])
    q2 = q1 + nn.functional.softplus(raw[..., 1:2])
    q3 = q2 + nn.functional.softplus(raw[..., 2:3])
    return torch.cat([q1, q2, q3], dim=-1)


class LSTMTrajectoryModel(nn.Module):
    """LSTM baseline with the same three prediction heads as the engine."""

    def __init__(self, units: int = 64, embed_dim: int = 16, time_mode: str = "feature"):
        super().__init__()
        if time_mode not in ("feature", "none"):
            raise ValueError(f"LSTM supports time_mode 'feature' or 'none', got {time_mode!r}")
        self.time_mode = time_mode
        self.event_embedding = nn.Embedding(N_EVENT_TYPES, embed_dim)
        input_size = embed_dim + EVENT_FEATURE_DIM + 1 + STATIC_DIM
        self.rnn = nn.LSTM(input_size, units, batch_first=True)
        self.head_settle = nn.Linear(units, 1)
        self.head_recovery = nn.Linear(units, 1)
        self.head_remaining = nn.Linear(units, 1)

    @property
    def state_size(self) -> int:
        return self.rnn.hidden_size

    def forward(
        self,
        event_ids: torch.Tensor,
        event_feats: torch.Tensor,
        static: torch.Tensor,
        timespans: torch.Tensor | None = None,
        hx=None,
    ) -> dict[str, torch.Tensor]:
        B, T = event_ids.shape
        emb = self.event_embedding(event_ids)
        if timespans is None or self.time_mode == "none":
            deltas = torch.zeros(B, T, 1, device=emb.device, dtype=emb.dtype)
        else:
            deltas = timespans.unsqueeze(-1) if timespans.dim() == 2 else timespans
            deltas = torch.log1p(deltas.clamp(min=0).to(emb.dtype))
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        x = torch.cat([emb, event_feats, deltas, static_exp], dim=-1)

        out, hx = self.rnn(x, hx)
        return {
            "settle_logit": self.head_settle(out).squeeze(-1),
            "log_recovery": self.head_recovery(out).squeeze(-1),
            "log_remaining": self.head_remaining(out).squeeze(-1),
            "hx": hx[0][-1] if isinstance(hx, tuple) else hx,
            "rnn_out": out,
        }


class TemporalTransformerModel(nn.Module):
    """Compact time-aware Transformer encoder baseline (causal-masked).

    Args:
        d_model: Model width. Default 48 keeps the parameter count in the
            same band as the 64-unit recurrent models (tens of thousands).
        nhead: Attention heads.
        num_layers: Encoder layers.
        dim_feedforward: FFN width.
        max_len: Maximum sequence length for the learned positional embedding.
        time_mode: see module docstring.
        auxiliary: if True, add the marked-event auxiliary heads (next-event
            type, next-gap and duration quantiles) so the model receives the
            same supervision signal as IDN — the auxiliary-matched primary
            opponent of the Stage-1 protocol.
    """

    TIME_MODES = ("native", "timespans_only", "feature", "none")

    def __init__(
        self,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 96,
        embed_dim: int = 16,
        max_len: int = 512,
        time_mode: str = "native",
        auxiliary: bool = False,
    ):
        super().__init__()
        if time_mode not in self.TIME_MODES:
            raise ValueError(f"unknown time_mode: {time_mode!r}")
        self.time_mode = time_mode
        self.auxiliary = auxiliary
        self.event_embedding = nn.Embedding(N_EVENT_TYPES, embed_dim)
        self.input_proj = nn.Linear(embed_dim + EVENT_FEATURE_DIM + STATIC_DIM, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head_settle = nn.Linear(d_model, 1)
        self.head_recovery = nn.Linear(d_model, 1)
        self.head_remaining = nn.Linear(d_model, 1)
        if auxiliary:
            self.head_next_type = nn.Linear(d_model, N_EVENT_TYPES)
            self.head_next_gap_q = nn.Linear(d_model, 3)
            self.head_duration_q = nn.Linear(d_model, 3)

    @property
    def state_size(self) -> int:
        return self.input_proj.out_features

    def forward(
        self,
        event_ids: torch.Tensor,
        event_feats: torch.Tensor,
        static: torch.Tensor,
        timespans: torch.Tensor | None = None,
        hx=None,
    ) -> dict[str, torch.Tensor]:
        B, T = event_ids.shape
        emb = self.event_embedding(event_ids)
        if self.time_mode in ("timespans_only", "none"):
            event_feats = event_feats.clone()
            event_feats[..., 0] = 0.0
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        x = self.input_proj(torch.cat([emb, event_feats, static_exp], dim=-1))
        x = x + self.pos_embedding.weight[:T].unsqueeze(0)

        if timespans is not None and self.time_mode in ("native", "timespans_only"):
            if timespans.dim() == 3:
                timespans = timespans.squeeze(-1)
            days = torch.cumsum(timespans.to(x.dtype).clamp(min=0), dim=1)
            x = x + _sinusoidal(days, x.shape[-1])

        # Causal mask: position t may attend to positions <= t only. Batches
        # are right-padded, so this also hides all padding from real positions.
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        out = self.encoder(x, mask=causal)
        result = {
            "settle_logit": self.head_settle(out).squeeze(-1),
            "log_recovery": self.head_recovery(out).squeeze(-1),
            "log_remaining": self.head_remaining(out).squeeze(-1),
            "hx": out[:, -1],
            "rnn_out": out,
        }
        if self.auxiliary:
            result["next_type_logit"] = self.head_next_type(out)
            result["next_gap_q"] = nonneg_non_crossing(self.head_next_gap_q(out))
            result["duration_q"] = nonneg_non_crossing(self.head_duration_q(out))
        return result
