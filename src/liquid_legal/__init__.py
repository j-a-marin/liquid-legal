"""liquid-legal: liquid neural networks (LTC/CfC) for legal case trajectories.

Models settlement probability, expected recovery, and remaining duration
from irregular docket event streams, using the ncps implementations of
Liquid Time-Constant and Closed-form Continuous-time networks.
"""

from .agents import CaseSnapshot, snapshot
from .baselines import LSTMTrajectoryModel
from .events import (
    N_EVENT_TYPES,
    STATIC_DIM,
    STATIC_FIELDS,
    TERMINAL_EVENTS,
    CaseEvent,
    CaseTimeline,
    EventType,
)
from .featurize import Batch, collate_timelines, featurize_timeline
from .interpret import event_saliency, wiring_report
from .metrics import auc_score
from .models import CaseTrajectoryEngine, build_wiring
from .synthetic import GeneratorConfig, SyntheticLitigationGenerator
from .train import TrainConfig, evaluate, initialize_output_biases, train_model

__version__ = "0.1.0"

__all__ = [
    "Batch",
    "CaseEvent",
    "CaseSnapshot",
    "CaseTimeline",
    "CaseTrajectoryEngine",
    "EventType",
    "GeneratorConfig",
    "LSTMTrajectoryModel",
    "N_EVENT_TYPES",
    "STATIC_DIM",
    "STATIC_FIELDS",
    "TERMINAL_EVENTS",
    "SyntheticLitigationGenerator",
    "TrainConfig",
    "auc_score",
    "build_wiring",
    "collate_timelines",
    "evaluate",
    "event_saliency",
    "featurize_timeline",
    "initialize_output_biases",
    "snapshot",
    "train_model",
    "wiring_report",
    "__version__",
]
