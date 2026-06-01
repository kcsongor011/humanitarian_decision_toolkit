"""Synthetic 30-day arrival scenario generator skeleton.

The latent expected-arrival path represents escalation and stabilisation over
time. The Poisson-lognormal variability mechanism represents stochastic
day-to-day uncertainty around that path; additional volatility is not treated
as solving the changing-intensity problem.
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
    lognormal_sigma: float

    def __post_init__(self) -> None:
        if self.start_day < 1:
            raise ValueError("start_day must be at least 1.")
        if self.end_day < self.start_day:
            raise ValueError("end_day must be greater than or equal to start_day.")
        if self.expected_start <= 0 or self.expected_end <= 0:
            raise ValueError("expected arrivals must be positive.")
        if self.lognormal_sigma < 0:
            raise ValueError("lognormal_sigma must be non-negative.")


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
            lognormal_sigma=0.18,
        ),
        PhaseConfig(
            label="surge_becomes_apparent",
            start_day=6,
            end_day=12,
            expected_start=75.0,
            expected_end=170.0,
            variability_state="rising",
            lognormal_sigma=0.24,
        ),
        PhaseConfig(
            label="volatile_high_pressure",
            start_day=13,
            end_day=18,
            expected_start=150.0,
            expected_end=230.0,
            variability_state="high",
            lognormal_sigma=0.32,
        ),
        PhaseConfig(
            label="updated_planning_period",
            start_day=19,
            end_day=23,
            expected_start=175.0,
            expected_end=130.0,
            variability_state="moderating",
            lognormal_sigma=0.22,
        ),
        PhaseConfig(
            label="partial_stabilisation",
            start_day=24,
            end_day=30,
            expected_start=125.0,
            expected_end=95.0,
            variability_state="lower",
            lognormal_sigma=0.16,
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
                    "_lognormal_sigma": phase.lognormal_sigma,
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
    ``lognormal_sigma`` controls mean-preserving multiplicative volatility
    around that path before a Poisson count draw.

    This synthetic-data mechanism keeps changing intensity separate from
    additional day-to-day volatility. It is not a claim that extra variability,
    or the later reader-facing Negative Binomial comparison, solves a
    trend/change-in-level forecasting problem.
    """

    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")

    active_rng = rng if rng is not None else np.random.default_rng(seed)
    path = expected_arrival_path(phases)
    arrivals = [
        _draw_arrival(
            expected=float(expected),
            lognormal_sigma=float(lognormal_sigma),
            rng=active_rng,
        )
        for expected, lognormal_sigma in zip(
            path["latent_expected_arrivals"],
            path["_lognormal_sigma"],
        )
    ]

    candidate = path.drop(columns=["_lognormal_sigma"]).copy()
    candidate.insert(1, "arrivals", pd.Series(arrivals, dtype="int64"))
    return candidate


def _draw_arrival(
    *,
    expected: float,
    lognormal_sigma: float,
    rng: np.random.Generator,
) -> int:
    """Draw one count from a mean-preserving Poisson-lognormal mechanism.

    ``lognormal_sigma`` controls additional multiplicative volatility around
    the latent expected-arrival path. When it is zero, the daily rate remains
    equal to ``expected`` and arrivals are ordinary Poisson count variation.
    """

    shock = rng.lognormal(mean=-0.5 * lognormal_sigma**2, sigma=lognormal_sigma)
    realised_rate = expected * shock
    return int(rng.poisson(lam=realised_rate))


def _validate_phase_coverage(phases: tuple[PhaseConfig, ...]) -> None:
    expected_days = list(range(1, 31))
    actual_days: list[int] = []
    for phase in phases:
        actual_days.extend(range(phase.start_day, phase.end_day + 1))

    if actual_days != expected_days:
        raise ValueError("Phases must cover days 1 through 30 contiguously.")
