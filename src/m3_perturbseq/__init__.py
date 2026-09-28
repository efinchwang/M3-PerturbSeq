"""Clean implementation of the final M3 Perturb-seq model."""

from .model import (
    CONDITION_SLICE,
    LINE_ORDER,
    CenteredLineHead,
    PCAContextLinear,
    WinnerM3Model,
    build_winner_model,
)

__all__ = [
    "CONDITION_SLICE",
    "LINE_ORDER",
    "CenteredLineHead",
    "PCAContextLinear",
    "WinnerM3Model",
    "build_winner_model",
]
