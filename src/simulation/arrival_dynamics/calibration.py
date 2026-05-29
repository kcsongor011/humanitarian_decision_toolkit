"""Descriptive diagnostics for generated arrival-dynamics candidates.

These helpers inspect whether a generated synthetic episode supports the
planned teaching sequence. They do not define acceptance thresholds, rank
candidates, select a canonical scenario, or claim that any forecasting
distribution is correct.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .generator import generate_candidate


WINDOWS: tuple[tuple[str, int, int], ...] = (
    ("days_1_5", 1, 5),
    ("days_6_12", 6, 12),
    ("days_13_18", 13, 18),
    ("days_19_23", 19, 23),
    ("days_24_30", 24, 30),
)

DIAGNOSTIC_COLUMNS = [
    "candidate_seed",
    "early_mean_days_1_5",
    "early_net_change_days_1_5",
    "early_linear_slope_days_1_5",
    "baseline_total_forecast_days_6_12",
    "realised_total_days_6_12",
    "surge_excess_arrivals_days_6_12",
    "surge_realised_to_baseline_ratio_days_6_12",
    "surge_peak_days_6_12",
    "high_pressure_mean_days_13_18",
    "high_pressure_std_days_13_18",
    "high_pressure_cv_days_13_18",
    "high_pressure_peak_days_13_18",
    "partial_stabilisation_mean_days_24_30",
    "partial_stabilisation_std_days_24_30",
    "partial_stabilisation_cv_days_24_30",
    "high_to_stabilisation_mean_ratio",
    "high_minus_stabilisation_std",
    "high_to_stabilisation_cv_ratio",
]


def summarize_candidate_windows(candidate: pd.DataFrame) -> pd.DataFrame:
    """Summarise observed arrivals in the main decision-point windows.

    Standard deviation captures absolute day-to-day fluctuation in arrival
    numbers, which matters operationally. Coefficient of variation provides a
    descriptive comparison of relative volatility across periods with
    different typical arrival levels. Neither metric is an acceptance
    threshold or a claim that a forecasting distribution is correct.
    """

    _validate_candidate(candidate)

    rows: list[dict[str, float | int | str]] = []
    for window, start_day, end_day in WINDOWS:
        arrivals = _window_arrivals(candidate, start_day, end_day)
        mean_arrivals = float(arrivals.mean())
        std_arrivals = float(arrivals.std())
        rows.append(
            {
                "window": window,
                "start_day": start_day,
                "end_day": end_day,
                "n_days": int(arrivals.size),
                "total_arrivals": int(arrivals.sum()),
                "mean_arrivals": mean_arrivals,
                "std_arrivals": std_arrivals,
                "min_arrivals": int(arrivals.min()),
                "peak_arrivals": int(arrivals.max()),
                "coefficient_of_variation": _coefficient_of_variation(
                    mean_arrivals=mean_arrivals,
                    std_arrivals=std_arrivals,
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_candidate_diagnostics(
    candidate: pd.DataFrame,
    *,
    candidate_seed: int | None = None,
) -> pd.DataFrame:
    """Calculate one row of descriptive diagnostics for a candidate scenario.

    Early net change and early linear slope describe whether days 1-5 visibly
    suggest escalation without defining a success threshold. Changing latent
    expected-arrival levels produce escalation and stabilisation, while the
    Poisson-lognormal mechanism produces additional day-to-day variation around
    those levels. Standard deviation describes absolute fluctuation;
    coefficient of variation describes relative volatility. No metric is an
    acceptance threshold, score, ranking input, or claim that a forecasting
    distribution is correct.
    """

    _validate_candidate(candidate)

    early = _window_arrivals(candidate, 1, 5)
    surge = _window_arrivals(candidate, 6, 12)
    high_pressure = _window_arrivals(candidate, 13, 18)
    partial_stabilisation = _window_arrivals(candidate, 24, 30)

    early_mean = float(early.mean())
    baseline_total = early_mean * 7
    realised_total = int(surge.sum())
    high_pressure_mean = float(high_pressure.mean())
    high_pressure_std = float(high_pressure.std())
    high_pressure_cv = _coefficient_of_variation(
        mean_arrivals=high_pressure_mean,
        std_arrivals=high_pressure_std,
    )
    partial_stabilisation_mean = float(partial_stabilisation.mean())
    partial_stabilisation_std = float(partial_stabilisation.std())
    partial_stabilisation_cv = _coefficient_of_variation(
        mean_arrivals=partial_stabilisation_mean,
        std_arrivals=partial_stabilisation_std,
    )

    row = {
        "candidate_seed": candidate_seed,
        "early_mean_days_1_5": early_mean,
        "early_net_change_days_1_5": int(
            _arrival_on_day(candidate, 5) - _arrival_on_day(candidate, 1)
        ),
        "early_linear_slope_days_1_5": _linear_slope(
            candidate.loc[candidate["day"].between(1, 5), "day"],
            early,
        ),
        "baseline_total_forecast_days_6_12": baseline_total,
        "realised_total_days_6_12": realised_total,
        "surge_excess_arrivals_days_6_12": realised_total - baseline_total,
        "surge_realised_to_baseline_ratio_days_6_12": _ratio(
            numerator=realised_total,
            denominator=baseline_total,
        ),
        "surge_peak_days_6_12": int(surge.max()),
        "high_pressure_mean_days_13_18": high_pressure_mean,
        "high_pressure_std_days_13_18": high_pressure_std,
        "high_pressure_cv_days_13_18": high_pressure_cv,
        "high_pressure_peak_days_13_18": int(high_pressure.max()),
        "partial_stabilisation_mean_days_24_30": partial_stabilisation_mean,
        "partial_stabilisation_std_days_24_30": partial_stabilisation_std,
        "partial_stabilisation_cv_days_24_30": partial_stabilisation_cv,
        "high_to_stabilisation_mean_ratio": _ratio(
            numerator=high_pressure_mean,
            denominator=partial_stabilisation_mean,
        ),
        "high_minus_stabilisation_std": (
            high_pressure_std - partial_stabilisation_std
        ),
        "high_to_stabilisation_cv_ratio": _ratio(
            numerator=high_pressure_cv,
            denominator=partial_stabilisation_cv,
        ),
    }

    return pd.DataFrame([row])


def build_diagnostics_table(candidate_seeds: Iterable[int]) -> pd.DataFrame:
    """Generate a reproducible descriptive diagnostics table for given seeds."""

    seeds = list(candidate_seeds)
    if not seeds:
        return pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)

    rows = [
        calculate_candidate_diagnostics(
            generate_candidate(seed=seed),
            candidate_seed=seed,
        )
        for seed in seeds
    ]
    return pd.concat(rows, ignore_index=True)


def _validate_candidate(candidate: pd.DataFrame) -> None:
    required_columns = {"day", "arrivals"}
    missing_columns = required_columns.difference(candidate.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Candidate is missing required columns: {missing}.")

    days = candidate["day"].tolist()
    if days != list(range(1, 31)):
        raise ValueError("Candidate must contain contiguous days 1 through 30.")


def _window_arrivals(
    candidate: pd.DataFrame,
    start_day: int,
    end_day: int,
) -> pd.Series:
    return candidate.loc[
        candidate["day"].between(start_day, end_day),
        "arrivals",
    ]


def _arrival_on_day(candidate: pd.DataFrame, day: int) -> int:
    return int(candidate.loc[candidate["day"] == day, "arrivals"].iloc[0])


def _linear_slope(days: pd.Series, arrivals: pd.Series) -> float:
    return float(np.polyfit(days, arrivals, deg=1)[0])


def _coefficient_of_variation(
    *,
    mean_arrivals: float,
    std_arrivals: float,
) -> float:
    if mean_arrivals <= 0:
        return float("nan")
    return std_arrivals / mean_arrivals


def _ratio(*, numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator
