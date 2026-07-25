"""F2 v1 candidate: `tf-tpp` — interval-supervised marked temporal point
process on the exact tf-native-aux Transformer trunk.

Frozen design (experiments/F2_PREREGISTRATION.md, section 2): the trunk is
identical to the Stage-1 primary opponent `tf-native-aux`
(liquid_legal.baselines.TemporalTransformerModel with d_model=32, nhead=4,
num_layers=2, dim_feedforward=64, max_len=128, time_mode="native", same
causal + padding masking). The two marked-event auxiliary heads (next-type
CE logits, next-gap quantiles) are REPLACED by per-mark conditional
intensity heads

    lambda_m(k) = softplus(w_m . h_k + b_m) + 1e-6     (events/day, 16 marks)

with h_k the causal Transformer output at position k. The hidden state is
constant between observed events (standard THP/RMTPP assumption), so the
conditional intensity is piecewise-constant over each observed interval and
the marked TPP likelihood is closed-form:

    per interval [t_k, t_{k+1}) with observed mark m_{k+1} and gap dt (days):
        l_k = log lambda_{m_{k+1}}(k) - dt * sum_m lambda_m(k)

The main heads (settle BCE, recovery, remaining-duration) and the
duration-quantile head are identical to tf-native-aux. All predictions the
F2 evaluation needs are derived in closed form from the intensities
(preregistration section 2, "Derived predictions"):

- next-event type probabilities: p(m) = lambda_m / Lambda, Lambda = sum_m lambda_m
  (full 16-mark distribution; FILED is not masked — prereg section 2);
- next-gap quantiles: the next gap is exponential with rate Lambda, so
  quantile q is -ln(1-q)/Lambda days, reported through log1p to match the
  frozen scoring scale;
- settlement within 180 d (cause-specific): (lambda_SETTLED/Lambda) *
  (1 - exp(-Lambda * 180)), reported descriptively alongside the main
  settle-BCE head.

Parameter cost: intensity head 16*32+16 = 528 replaces next-type
16*32+16 = 528 and next-gap 3*32+3 = 99 -> net -99 params vs tf-native-aux
(23,129), i.e. 23,030, within the declared +/-20% budget.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from liquid_legal.baselines import TemporalTransformerModel, nonneg_non_crossing
from liquid_legal.events import N_EVENT_TYPES, EventType

#: Quantile levels of the frozen scoring rule (f2_score_calibration.py).
QUANTILE_LEVELS = (0.1, 0.5, 0.9)

#: Intensity floor (events/day), part of the frozen design.
LAMBDA_FLOOR = 1e-6

#: Horizon of the TPP-derived settlement probability (days).
SETTLE_HORIZON_DAYS = 180.0


class TppTransformerModel(TemporalTransformerModel):
    """tf-native-aux trunk with per-mark intensity heads in place of the
    next-type / next-gap auxiliary heads (F2 v1 candidate `tf-tpp`)."""

    def __init__(
        self,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        embed_dim: int = 16,
        max_len: int = 128,
        time_mode: str = "native",
    ):
        # auxiliary=False: the base class builds the identical trunk and main
        # heads but not the aux heads that this model replaces.
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            embed_dim=embed_dim,
            max_len=max_len,
            time_mode=time_mode,
            auxiliary=False,
        )
        # Kept from tf-native-aux's auxiliary set: duration quantiles.
        self.head_duration_q = nn.Linear(d_model, 3)
        # Replacement for head_next_type / head_next_gap_q: per-mark
        # conditional intensities (16 marks).
        self.head_intensity = nn.Linear(d_model, N_EVENT_TYPES)

    def forward(
        self,
        event_ids: torch.Tensor,
        event_feats: torch.Tensor,
        static: torch.Tensor,
        timespans: torch.Tensor | None = None,
        hx=None,
    ) -> dict[str, torch.Tensor]:
        # Identical trunk computation (causal + padding masking) as
        # tf-native-aux; auxiliary=False so only the main heads fire here.
        result = super().forward(
            event_ids, event_feats, static, timespans=timespans, hx=hx
        )
        h = result["rnn_out"]  # (B, T, d_model)

        lambdas = nn.functional.softplus(self.head_intensity(h)) + LAMBDA_FLOOR
        total = lambdas.sum(dim=-1, keepdim=True)  # (B, T, 1) Lambda

        # Closed-form derived predictions (preregistration section 2).
        next_type_prob = lambdas / total  # (B, T, 16), sums to 1
        gap_q_days = torch.cat(
            [-math.log(1.0 - q) / total for q in QUANTILE_LEVELS], dim=-1
        )  # (B, T, 3), monotone in q by construction
        next_gap_q = torch.log1p(gap_q_days)  # log1p-day scoring scale
        settled = int(EventType.SETTLED)
        settle_prob_180d = (lambdas[..., settled : settled + 1] / total).squeeze(-1) * (
            1.0 - torch.exp(-total.squeeze(-1) * SETTLE_HORIZON_DAYS)
        )

        result["duration_q"] = nonneg_non_crossing(self.head_duration_q(h))
        result["lambdas"] = lambdas
        result["next_type_prob"] = next_type_prob
        result["next_gap_q"] = next_gap_q
        result["settle_prob_180d"] = settle_prob_180d
        return result
