"""Synthetic 30-day arrival scenario generator skeleton.

The latent expected-arrival path represents escalation and stabilisation over
time. The variability mechanism represents stochastic day-to-day uncertainty
around that path; greater dispersion is not treated as solving the
changing-intensity problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for one latent scenario phase."""

    label: str
    start_day: int
    end_day: int
    expected_start: float
    expected_end: float
    variability_state: str
    dispersion: float

    def __post_init__(self) -> None:
        if self.start_day < 1:
            raise ValueError("start_day must be at least 1.")
        if self.end_day < self.start_day:
            raise ValueError("end_day must be greater than or equal to start_day.")
        if self.expected_start <= 0 or self.expected_end <= 0:
            raise ValueError("expected arrivals must be positive.")
        if self.dispersion <= 0:
            raise ValueError("dispersion must be positive.")


def default_phase_config() -> tuple[PhaseConfig, ...]:
    """Return the five agreed latent phases for the 30-day prototype episode."""

    return (
        PhaseConfig(
            label="early_visible_escalation",
            start_day=1,
            end_day=5,
            expected_start=35.0,
            expected_end=65.0,
            variability_state="moderate",
            dispersion=0.18,
        ),
        PhaseConfig(
            label="surge_becomes_apparent",
            start_day=6,
            end_day=12,
            expected_start=75.0,
            expected_end=170.0,
            variability_state="rising",
            dispersion=0.24,
        ),
        PhaseConfig(
            label="volatile_high_pressure",
            start_day=13,
            end_day=18,
            expected_start=150.0,
            expected_end=230.0,
            variability_state="high",
            dispersion=0.32,
        ),
        PhaseConfig(
            label="updated_planning_period",
            start_day=19,
            end_day=23,
            expected_start=175.0,
            expected_end=130.0,
            variability_state="moderating",
            dispersion=0.22,
        ),
        PhaseConfig(
            label="partial_stabilisation",
            start_day=24,
            end_day=30,
            expected_start=125.0,
            expected_end=95.0,
            variability_state="lower",
            dispersion=0.16,
        ),
    )


def expected_arrival_path(
    phases: Iterable[PhaseConfig] | None = None,
) -> pd.DataFrame:
    """Build the latent expected-arrival path across all configured phases."""

    phase_configs = default_phase_config() if phases is None else tuple(phases)
    _validate_phase_coverage(phase_configs)

    rows: list[dict[str, object]] = []
    for phase in phase_configs:
        days = np.arange(phase.start_day, phase.end_day + 1)
        expected_values = np.linspace(
            phase.expected_start,
            phase.expected_end,
            num=len(days),
        )
        for day, expected in zip(days, expected_values):
            rows.append(
                {
                    "day": int(day),
                    "phase": phase.label,
                    "latent_expected_arrivals": float(expected),
                    "latent_variability_state": phase.variability_state,
                    "_dispersion": phase.dispersion,
                }
            )

    return pd.DataFrame(rows)


def generate_candidate(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    phases: Iterable[PhaseConfig] | None = None,
) -> pd.DataFrame:
    """Generate one internal 30-day candidate scenario table.

    Pass either ``seed`` or ``rng`` for deterministic reproducibility. The
    latent expected path controls escalation and stabilisation; phase-specific
    dispersion controls stochastic uncertainty around that path.

    The rounded normal draw is provisional for this skeleton stage and is
    subject to calibration review. It is not the final modelling choice and is
    not a claim about the later reader-facing Poisson versus Negative Binomial
    forecasts.
    """

    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")

    active_rng = rng if rng is not None else np.random.default_rng(seed)
    path = expected_arrival_path(phases)
    arrivals = [
        _draw_arrival(expected=float(expected), dispersion=float(dispersion), rng=active_rng)
        for expected, dispersion in zip(
            path["latent_expected_arrivals"],
            path["_dispersion"],
        )
    ]

    candidate = path.drop(columns=["_dispersion"]).copy()
    candidate.insert(1, "arrivals", pd.Series(arrivals, dtype="int64"))
    return candidate


def _draw_arrival(
    *,
    expected: float,
    dispersion: float,
    rng: np.random.Generator,
) -> int:
    """Draw one non-negative integer arrival count around a latent mean."""

    standard_deviation = max(expected * dispersion, 1.0)
    draw = rng.normal(loc=expected, scale=standard_deviation)
    return max(0, int(round(draw)))


def _validate_phase_coverage(phases: tuple[PhaseConfig, ...]) -> None:
    expected_days = list(range(1, 31))
    actual_days: list[int] = []
    for phase in phases:
        actual_days.extend(range(phase.start_day, phase.end_day + 1))

    if actual_days != expected_days:
        raise ValueError("Phases must cover days 1 through 30 contiguously.")
