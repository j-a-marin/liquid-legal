"""IDN Stage 1: a hybrid state-space marked-event model for litigation forecasting.

Predictive model, not proof of institutional dynamics (claim discipline:
IDN_GUIDE.md section 10). Core hypothesis: separating elapsed-time
transitions from event-conditioned updates improves calibrated forecasts for
irregular legal-event sequences.

Chronology contract, for event e_k arriving after interval Δt_k:

    c_{k-1} = HistoryEncoder(e_{≤k-1}, t_{≤k-1})   # causal-masked, computed once
    z_k^-   = Φ(z_{k-1}, c_{k-1}, Δt_k)            # elapsed-time flow (clock partition)
    z_k     = J(z_k^-, e_k, c_{k-1})               # event-conditioned jump

The pre-event flow never sees e_k. Automated leakage tests: tests/test_idn.py.

Clock flow (continuous-time semantics): with rate r = softplus(R(c)) ≥ 0 and
α = 1 − exp(−r·Δt),

    Φ(z, c, Δt) = (1 − α)·z + α·target(c)

so Δt = 0 is exactly the identity and Φ satisfies the interval-composition
property Φ(t₁+t₂) = Φ(t₂)∘Φ(t₁) for constant context. The selective gate
then chooses between the flowed and the un-flowed state per clock dimension.

State partition (guide section 3): z = [z_event, z_clock, z_context], with
static covariates as fixed inputs.

* ``event`` changes only at observed events (GRU jump).
* ``clock`` evolves only during intervals (gated flow above).
* ``context`` receives history-encoder information (post-event, legal).

Padding: forward accepts optional ``lengths`` (per-row valid step counts).
Attention uses a key-padding mask, state updates are frozen at padded steps,
and ``hx`` is the state at each row's last valid step. All quantile heads are
non-crossing by construction (cumulative softplus).
"""

from __future__ import annotations

import torch
from torch import nn

from liquid_legal.baselines import _sinusoidal, nonneg_non_crossing
from liquid_legal.events import N_EVENT_TYPES, STATIC_DIM
from liquid_legal.featurize import EVENT_FEATURE_DIM


class IDNModel(nn.Module):
    """Stage-1 IDN. Same forward interface as the rest of the model zoo,
    plus optional ``lengths`` for padding-correct state handling, auxiliary
    marked-event heads, and intermediates for leakage tests.

    Args:
        d_event: event-partition width (jump-only updates).
        d_clock: clock-partition width (interval-only flow).
        d_context: context-partition width (history-encoder writes).
        tf_d: history-encoder width.
        embed_dim: event-type embedding dimension.
        time_scale: days per unit of flow time.
    """

    def __init__(
        self,
        d_event: int = 16,
        d_clock: int = 16,
        d_context: int = 32,
        tf_d: int = 32,
        embed_dim: int = 16,
        max_len: int = 128,
        time_scale: float = 30.0,
    ):
        super().__init__()
        self.d_event, self.d_clock, self.d_context = d_event, d_clock, d_context
        self.state_size = d_event + d_clock + d_context
        self.time_scale = float(time_scale)

        self.event_embedding = nn.Embedding(N_EVENT_TYPES, embed_dim)

        # --- causally masked history encoder (over events, time-aware) --- #
        self.step_proj = nn.Linear(embed_dim + EVENT_FEATURE_DIM + STATIC_DIM, tf_d)
        self.pos_embedding = nn.Embedding(max_len, tf_d)
        layer = nn.TransformerEncoderLayer(
            d_model=tf_d, nhead=4, dim_feedforward=2 * tf_d,
            batch_first=True, norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(layer, num_layers=1,
                                                     enable_nested_tensor=False)
        self.context_proj = nn.Linear(tf_d, d_context)
        self.c_null = nn.Parameter(torch.zeros(d_context))

        # --- elapsed-time flow Φ (clock partition, pre-event inputs only) --- #
        self.flow_rate = nn.Linear(d_context, d_clock)     # learned rate R(c)
        self.flow_target = nn.Linear(d_context, d_clock)
        self.flow_gate = nn.Linear(d_context + 1, d_clock)  # selective gate
        self.softplus = nn.Softplus()

        # --- event-conditioned jump J (event + context partitions) --- #
        self.jump_event = nn.GRUCell(embed_dim + d_context, d_event)
        self.context_blend = nn.Linear(d_context, d_context)
        self.context_norm = nn.LayerNorm(d_context)

        # --- heads (main tasks + marked-event auxiliaries) --- #
        head_in = self.state_size + STATIC_DIM
        self.head_settle = nn.Linear(head_in, 1)
        self.head_recovery = nn.Linear(head_in, 1)
        self.head_remaining = nn.Linear(head_in, 1)
        self.head_next_type = nn.Linear(head_in, N_EVENT_TYPES)
        self.head_next_gap_q = nn.Linear(head_in, 3)   # quantiles .1/.5/.9
        self.head_duration_q = nn.Linear(head_in, 3)

    # ------------------------------------------------------------------ #

    def _clock_flow(self, z_clock, c, dt):
        """Φ(z, c, Δt) = (1−α)z + α·target(c), α = 1 − exp(−softplus(R(c))·Δt).

        Exact identity at Δt = 0; exact interval composition for constant c.
        """
        rate = self.softplus(self.flow_rate(c))
        alpha = 1.0 - torch.exp(-rate * dt.clamp(min=0))
        target = torch.tanh(self.flow_target(c))
        return (1.0 - alpha) * z_clock + alpha * target

    @staticmethod
    def _non_crossing(raw):
        """3 quantiles, nonnegative and non-crossing by construction."""
        return nonneg_non_crossing(raw)

    def _contexts(self, event_ids, event_feats, static, timespans, lengths):
        """Run the history encoder once; position k holds c_k = context from
        e_{<=k} only (causal mask + key-padding mask)."""
        B, T = event_ids.shape
        emb = self.event_embedding(event_ids)
        static_exp = static.unsqueeze(1).expand(B, T, static.shape[-1])
        x = self.step_proj(torch.cat([emb, event_feats, static_exp], dim=-1))
        x = x + self.pos_embedding.weight[:T].unsqueeze(0)
        days = torch.cumsum(timespans.clamp(min=0), dim=1)
        x = x + _sinusoidal(days, x.shape[-1])
        causal = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        pad = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        return self.context_proj(
            self.history_encoder(x, mask=causal, src_key_padding_mask=pad)
        )

    def forward(
        self,
        event_ids: torch.Tensor,
        event_feats: torch.Tensor,
        static: torch.Tensor,
        timespans: torch.Tensor | None = None,
        hx: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, T = event_ids.shape
        device = event_ids.device
        if timespans is None:
            timespans = torch.ones(B, T, device=device, dtype=event_feats.dtype)
        if timespans.dim() == 3:
            timespans = timespans.squeeze(-1)
        deltas = timespans.to(event_feats.dtype) / self.time_scale
        if lengths is None:
            lengths = torch.full((B,), T, device=device, dtype=torch.long)
        lengths = lengths.to(device).clamp(min=0, max=T)

        contexts = self._contexts(event_ids, event_feats, static, deltas, lengths)

        z = torch.zeros(B, self.state_size, device=device, dtype=event_feats.dtype) if hx is None else hx
        z_event, z_clock, z_context = (
            z[:, : self.d_event],
            z[:, self.d_event : self.d_event + self.d_clock],
            z[:, self.d_event + self.d_clock :],
        )

        outs, gates, z_minuses = [], [], []
        for k in range(T):
            c_prev = contexts[:, k - 1] if k > 0 else self.c_null.expand(B, -1)

            # ---- elapsed-time flow (pre-event: z_{k-1}, c_{k-1}, Δt_k only)
            flowed = self._clock_flow(z_clock, c_prev, deltas[:, k].unsqueeze(-1))
            gate_in = torch.cat([c_prev, torch.log1p(deltas[:, k].clamp(min=0)).unsqueeze(-1)], dim=-1)
            g = torch.sigmoid(self.flow_gate(gate_in))
            z_clock_minus = g * flowed + (1.0 - g) * z_clock
            z_minus = torch.cat([z_event, z_clock_minus, z_context], dim=-1)
            z_minuses.append(z_minus)

            # ---- event-conditioned jump (now e_k may be seen)
            emb_k = self.event_embedding(event_ids[:, k])
            z_event_new = self.jump_event(torch.cat([emb_k, c_prev], dim=-1), z_event)
            blend = torch.sigmoid(self.context_blend(contexts[:, k]))
            z_context_new = self.context_norm(
                (1.0 - blend) * z_context + blend * contexts[:, k]
            )

            # ---- padding freeze: no state change at padded steps
            active = (k < lengths).unsqueeze(-1)
            z_event = torch.where(active, z_event_new, z_event)
            z_clock = torch.where(active, z_clock_minus, z_clock)
            z_context = torch.where(active, z_context_new, z_context)

            z = torch.cat([z_event, z_clock, z_context], dim=-1)
            outs.append(z)
            gates.append(g)

        Z = torch.stack(outs, dim=1)                      # (B, T, state)
        head_in = torch.cat(
            [Z, static.unsqueeze(1).expand(B, T, static.shape[-1])], dim=-1
        )
        result = {
            "settle_logit": self.head_settle(head_in).squeeze(-1),
            "log_recovery": self.head_recovery(head_in).squeeze(-1),
            "log_remaining": self.head_remaining(head_in).squeeze(-1),
            "next_type_logit": self.head_next_type(head_in),       # (B, T, 16)
            "next_gap_q": self._non_crossing(self.head_next_gap_q(head_in)),
            "duration_q": self._non_crossing(self.head_duration_q(head_in)),
            "gate_clock": torch.stack(gates, dim=1),               # (B, T, d_clock)
            "hx": z,  # state at each row's last valid step (padding-frozen)
            "rnn_out": Z,
        }
        if return_intermediates:
            result["z_minus"] = torch.stack(z_minuses, dim=1)
            result["contexts"] = contexts
        return result
