"""Reader-facing forecast helpers for Arrival Dynamics 1.

These helpers operate on public observed arrivals only. They do not import the
synthetic scenario generator or use latent generator fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


FORECAST_COLUMNS = [
    "day",
    "forecast_arrivals",
    "observed_mean_arrivals",
    "forecast_horizon_days",
    "forecast_total_arrivals",
]

SIMULATION_COLUMNS = [
    "simulation_id",
    "day",
    "simulated_arrivals",
]

UPDATED_SIMULATION_COLUMNS = [
    "model",
    "simulation_id",
    "day",
    "simulated_arrivals",
    "assumed_daily_mean",
    "assumed_daily_variance",
]

NEGATIVE_BINOMIAL_SIMULATION_COLUMNS = UPDATED_SIMULATION_COLUMNS + [
    "negative_binomial_n",
    "negative_binomial_p",
]

SUMMARY_COLUMNS = [
    "statistic",
    "value",
]

ARRIVAL_LEVEL_COLUMNS = [
    "observed_start_day",
    "observed_end_day",
    "observed_days",
    "updated_daily_arrival_level",
    "observed_sample_variance",
    "observed_sample_cv",
]

MODEL_SUMMARY_COLUMNS = [
    "model",
    "statistic",
    "value",
]


def mean_baseline_forecast(
    observed: pd.DataFrame,
    *,
    observed_start_day: int = 1,
    observed_end_day: int = 5,
    forecast_start_day: int = 6,
    forecast_end_day: int = 12,
) -> pd.DataFrame:
    """Calculate a constant mean-based forecast from observed arrivals."""

    observed_arrivals = _observed_window_arrivals(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    forecast_days = _forecast_days(
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
    )

    observed_mean = float(observed_arrivals.mean())
    horizon_days = len(forecast_days)
    forecast_total = observed_mean * horizon_days

    return pd.DataFrame(
        {
            "day": forecast_days,
            "forecast_arrivals": observed_mean,
            "observed_mean_arrivals": observed_mean,
            "forecast_horizon_days": horizon_days,
            "forecast_total_arrivals": forecast_total,
        },
        columns=FORECAST_COLUMNS,
    )


def simulate_poisson_forecast(
    observed: pd.DataFrame,
    *,
    observed_start_day: int = 1,
    observed_end_day: int = 5,
    forecast_start_day: int = 6,
    forecast_end_day: int = 12,
    n_simulations: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate Poisson future-arrival trajectories from observed arrivals."""

    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    observed_arrivals = _observed_window_arrivals(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    forecast_days = _forecast_days(
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
    )

    active_rng = rng if rng is not None else np.random.default_rng(seed)
    daily_rate = float(observed_arrivals.mean())
    draws = active_rng.poisson(
        lam=daily_rate,
        size=(n_simulations, len(forecast_days)),
    )

    return pd.DataFrame(
        {
            "simulation_id": np.repeat(
                np.arange(1, n_simulations + 1),
                len(forecast_days),
            ),
            "day": np.tile(forecast_days, n_simulations),
            "simulated_arrivals": draws.reshape(-1),
        },
        columns=SIMULATION_COLUMNS,
    )


def summarize_poisson_totals(
    simulations: pd.DataFrame,
    *,
    percentiles: tuple[float, ...] = (5, 25, 50, 75, 95),
) -> pd.DataFrame:
    """Summarise simulated total arrivals across the forecast horizon."""

    _validate_simulations(simulations)
    totals = simulations.groupby("simulation_id")["simulated_arrivals"].sum()

    rows = [
        {"statistic": "mean", "value": float(totals.mean())},
        {"statistic": "std", "value": float(totals.std())},
        {"statistic": "min", "value": float(totals.min())},
        {"statistic": "max", "value": float(totals.max())},
    ]
    for percentile in percentiles:
        rows.append(
            {
                "statistic": _percentile_label(percentile),
                "value": float(np.percentile(totals, percentile)),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def estimate_recent_arrival_level(
    observed: pd.DataFrame,
    *,
    observed_start_day: int = 6,
    observed_end_day: int = 12,
) -> pd.DataFrame:
    """Estimate an updated daily arrival level from recent observed arrivals."""

    stats = _recent_arrival_stats(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )

    return pd.DataFrame([stats], columns=ARRIVAL_LEVEL_COLUMNS)


def simulate_updated_poisson_forecast(
    observed: pd.DataFrame,
    *,
    observed_start_day: int = 6,
    observed_end_day: int = 12,
    forecast_start_day: int = 13,
    forecast_end_day: int = 18,
    n_simulations: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate a Poisson forecast around an updated observed arrival level."""

    active_rng = _simulation_rng(seed=seed, rng=rng, n_simulations=n_simulations)
    stats = _recent_arrival_stats(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    forecast_days = _forecast_days(
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
    )

    daily_mean = stats["updated_daily_arrival_level"]
    draws = active_rng.poisson(
        lam=daily_mean,
        size=(n_simulations, len(forecast_days)),
    )

    return pd.DataFrame(
        {
            "model": "poisson_updated_level",
            "simulation_id": np.repeat(
                np.arange(1, n_simulations + 1),
                len(forecast_days),
            ),
            "day": np.tile(forecast_days, n_simulations),
            "simulated_arrivals": draws.reshape(-1),
            "assumed_daily_mean": daily_mean,
            "assumed_daily_variance": daily_mean,
        },
        columns=UPDATED_SIMULATION_COLUMNS,
    )


def simulate_updated_negative_binomial_forecast(
    observed: pd.DataFrame,
    *,
    observed_start_day: int = 6,
    observed_end_day: int = 12,
    forecast_start_day: int = 13,
    forecast_end_day: int = 18,
    n_simulations: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Simulate an overdispersed forecast around an updated observed level.

    The updated mean addresses the changed arrival level visible in the recent
    observed window. The Negative Binomial comparison represents greater
    day-to-day variability and upper-tail risk around that level; it does not
    solve trend or changing-intensity problems.
    """

    active_rng = _simulation_rng(seed=seed, rng=rng, n_simulations=n_simulations)
    stats = _recent_arrival_stats(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    forecast_days = _forecast_days(
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
    )

    daily_mean = stats["updated_daily_arrival_level"]
    daily_variance = stats["observed_sample_variance"]
    n, p = _negative_binomial_parameters(mean=daily_mean, variance=daily_variance)
    draws = active_rng.negative_binomial(
        n=n,
        p=p,
        size=(n_simulations, len(forecast_days)),
    )

    return pd.DataFrame(
        {
            "model": "negative_binomial_updated_level",
            "simulation_id": np.repeat(
                np.arange(1, n_simulations + 1),
                len(forecast_days),
            ),
            "day": np.tile(forecast_days, n_simulations),
            "simulated_arrivals": draws.reshape(-1),
            "assumed_daily_mean": daily_mean,
            "assumed_daily_variance": daily_variance,
            "negative_binomial_n": n,
            "negative_binomial_p": p,
        },
        columns=NEGATIVE_BINOMIAL_SIMULATION_COLUMNS,
    )


def summarize_forecast_totals_by_model(
    simulations: pd.DataFrame,
    *,
    percentiles: tuple[float, ...] = (50, 80, 90, 95),
) -> pd.DataFrame:
    """Summarise simulated forecast-horizon totals for each model."""

    _validate_model_simulations(simulations)
    totals = (
        simulations.groupby(["model", "simulation_id"])["simulated_arrivals"]
        .sum()
        .reset_index(name="simulated_total_arrivals")
    )

    rows = []
    for model, model_totals in totals.groupby("model", sort=False):
        values = model_totals["simulated_total_arrivals"]
        rows.extend(
            [
                {"model": model, "statistic": "mean", "value": float(values.mean())},
                {"model": model, "statistic": "std", "value": float(values.std())},
                {"model": model, "statistic": "min", "value": float(values.min())},
                {"model": model, "statistic": "max", "value": float(values.max())},
            ]
        )
        for percentile in percentiles:
            rows.append(
                {
                    "model": model,
                    "statistic": _percentile_label(percentile),
                    "value": float(np.percentile(values, percentile)),
                }
            )

    return pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS)


def _observed_window_arrivals(
    observed: pd.DataFrame,
    *,
    observed_start_day: int,
    observed_end_day: int,
) -> pd.Series:
    _validate_day_window(
        start_day=observed_start_day,
        end_day=observed_end_day,
        window_name="observed",
    )
    _validate_observed_schema(observed)

    observed_window = observed.loc[
        observed["day"].between(observed_start_day, observed_end_day),
        ["day", "arrivals"],
    ].copy()
    expected_days = list(range(observed_start_day, observed_end_day + 1))

    if observed_window["day"].duplicated().any():
        raise ValueError("Observed window contains duplicate days.")
    if sorted(observed_window["day"].tolist()) != expected_days:
        raise ValueError("Observed window must contain each required day exactly once.")
    if (observed_window["arrivals"] < 0).any():
        raise ValueError("Observed arrivals must be non-negative.")

    return observed_window.sort_values("day")["arrivals"]


def _recent_arrival_stats(
    observed: pd.DataFrame,
    *,
    observed_start_day: int,
    observed_end_day: int,
) -> dict[str, float | int]:
    observed_arrivals = _observed_window_arrivals(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    daily_mean = float(observed_arrivals.mean())
    sample_variance = float(observed_arrivals.var(ddof=1))
    sample_std = float(observed_arrivals.std(ddof=1))
    sample_cv = sample_std / daily_mean if daily_mean > 0 else np.nan

    return {
        "observed_start_day": observed_start_day,
        "observed_end_day": observed_end_day,
        "observed_days": int(len(observed_arrivals)),
        "updated_daily_arrival_level": daily_mean,
        "observed_sample_variance": sample_variance,
        "observed_sample_cv": sample_cv,
    }


def _simulation_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
    n_simulations: int,
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Provide either seed or rng, not both.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")
    return rng if rng is not None else np.random.default_rng(seed)


def _negative_binomial_parameters(*, mean: float, variance: float) -> tuple[float, float]:
    if not np.isfinite(variance) or variance <= mean:
        raise ValueError(
            "Negative Binomial desired variance must be greater than the "
            "updated daily arrival level."
        )

    p = mean / variance
    n = mean**2 / (variance - mean)
    return n, p


def _forecast_days(*, forecast_start_day: int, forecast_end_day: int) -> list[int]:
    _validate_day_window(
        start_day=forecast_start_day,
        end_day=forecast_end_day,
        window_name="forecast",
    )
    return list(range(forecast_start_day, forecast_end_day + 1))


def _validate_observed_schema(observed: pd.DataFrame) -> None:
    required_columns = {"day", "arrivals"}
    missing_columns = required_columns.difference(observed.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Observed data is missing required columns: {missing}.")


def _validate_day_window(*, start_day: int, end_day: int, window_name: str) -> None:
    if start_day < 1:
        raise ValueError(f"{window_name}_start_day must be at least 1.")
    if end_day < start_day:
        raise ValueError(
            f"{window_name}_end_day must be greater than or equal to "
            f"{window_name}_start_day."
        )


def _validate_simulations(simulations: pd.DataFrame) -> None:
    required_columns = {"simulation_id", "simulated_arrivals"}
    missing_columns = required_columns.difference(simulations.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Simulations are missing required columns: {missing}.")
    if simulations.empty:
        raise ValueError("Simulations must not be empty.")
    if (simulations["simulated_arrivals"] < 0).any():
        raise ValueError("Simulated arrivals must be non-negative.")


def _validate_model_simulations(simulations: pd.DataFrame) -> None:
    _validate_simulations(simulations)
    if "model" not in simulations.columns:
        raise ValueError("Simulations are missing required columns: model.")


def _percentile_label(percentile: float) -> str:
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentiles must be between 0 and 100.")
    if float(percentile).is_integer():
        return f"p{int(percentile):02d}"
    return f"p{str(percentile).replace('.', '_')}"
