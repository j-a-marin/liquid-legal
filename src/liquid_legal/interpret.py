"""Audit hooks: event saliency and wiring-sparsity reports.

Litigation finance is regulated and high-stakes, so the package ships with
first-class interpretability:

* :func:`event_saliency` attributes each settlement prediction to the docket
  events that drove it, via input gradients.
* :func:`wiring_report` quantifies how sparse an NCP wiring is relative to a
  dense recurrent wiring — the structural auditability argument for NCPs.
"""

from __future__ import annotations

import numpy as np
import torch
from ncps.wirings import Wiring

from .featurize import Batch


def event_saliency(
    model: torch.nn.Module, batch: Batch, device: str = "cpu"
) -> np.ndarray:
    """Per-timestep saliency of the settlement head w.r.t. event features.

    Returns an (B, T) array: the L2 norm of d(settle_logit)/d(event_feats)
    at each timestep, zero on padding.
    """
    model.train()
    batch = batch.to(device)
    feats = batch.event_feats.clone().detach().requires_grad_(True)
    out = model(batch.event_ids, feats, batch.static, timespans=batch.deltas)
    (out["settle_logit"] * batch.mask).sum().backward()
    sal = feats.grad.norm(dim=-1) * batch.mask
    return sal.detach().cpu().numpy()


def wiring_report(wiring: Wiring) -> dict[str, float]:
    """Synapse counts and sparsity of a wiring vs. its dense equivalent."""
    units = wiring.units
    sensory = recurrent = None
    try:
        sensory = int(np.count_nonzero(wiring.sensory_adjacency_matrix))
        recurrent = int(np.count_nonzero(wiring.adjacency_matrix))
    except Exception:  # pragma: no cover - wiring without exposed matrices
        pass
    report: dict[str, float] = {"units": float(units)}
    if sensory is not None and recurrent is not None:
        input_dim = getattr(wiring, "input_dim", None) or 0
        dense = units * units + units * input_dim
        actual = sensory + recurrent
        report.update(
            sensory_synapses=float(sensory),
            recurrent_synapses=float(recurrent),
            dense_equivalent=float(dense),
            sparsity=1.0 - actual / dense if dense else 0.0,
        )
    return report
