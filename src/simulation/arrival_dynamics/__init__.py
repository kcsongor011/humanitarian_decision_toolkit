"""Arrival Dynamics short-term border-arrival simulation helpers."""

from .calibration import (
    build_diagnostics_table,
    calculate_candidate_diagnostics,
    summarize_candidate_windows,
)
from .generator import (
    PhaseConfig,
    default_phase_config,
    expected_arrival_path,
    generate_candidate,
)

__all__ = [
    "PhaseConfig",
    "build_diagnostics_table",
    "calculate_candidate_diagnostics",
    "default_phase_config",
    "expected_arrival_path",
    "generate_candidate",
    "summarize_candidate_windows",
]
