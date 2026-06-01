"""Reader-facing forecast helpers for Arrival Dynamics 1.

These helpers operate on public observed arrivals only. They do not import the
synthetic scenario generator or use latent generator fields.
"""

from __future__ import annotations

import math

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

CAUTIOUS_NEGATIVE_BINOMIAL_SIMULATION_COLUMNS = NEGATIVE_BINOMIAL_SIMULATION_COLUMNS + [
    "variance_rule",
    "recent_sample_sd",
    "sd_floor",
    "assumed_daily_sd",
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

CAUTIOUS_ARRIVAL_LEVEL_COLUMNS = [
    "observed_start_day",
    "observed_end_day",
    "observed_days",
    "updated_daily_arrival_level",
    "recent_sample_sd",
    "sd_floor_fraction",
    "sd_floor",
    "assumed_daily_sd",
    "assumed_daily_variance",
    "variance_rule",
]

PREPAREDNESS_SUMMARY_COLUMNS = [
    "model",
    "forecast_reference",
    "forecast_total_arrivals",
]

PREPAREDNESS_COLUMNS = [
    "model",
    "forecast_reference",
    "forecast_total_arrivals",
    "forecast_horizon_days",
    "average_daily_arrivals",
    "immediate_water_litres",
    "immediate_food_units",
    "protection_contacts",
    "protection_staff_days",
    "registration_workload",
    "medical_teams_per_day",
    "medical_team_days",
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


def estimate_cautious_arrival_level(
    observed: pd.DataFrame,
    *,
    observed_start_day: int,
    observed_end_day: int,
    sd_floor_fraction: float = 0.25,
) -> pd.DataFrame:
    """Estimate an updated level with a cautious standard-deviation floor."""

    if sd_floor_fraction < 0:
        raise ValueError("sd_floor_fraction must be non-negative.")

    stats = _recent_arrival_stats(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
    )
    daily_mean = float(stats["updated_daily_arrival_level"])
    recent_sample_sd = float(stats["observed_sample_variance"]) ** 0.5
    sd_floor = sd_floor_fraction * daily_mean
    assumed_daily_sd = max(recent_sample_sd, sd_floor)
    assumed_daily_variance = assumed_daily_sd**2
    _negative_binomial_parameters(
        mean=daily_mean,
        variance=assumed_daily_variance,
    )

    row = {
        "observed_start_day": stats["observed_start_day"],
        "observed_end_day": stats["observed_end_day"],
        "observed_days": stats["observed_days"],
        "updated_daily_arrival_level": daily_mean,
        "recent_sample_sd": recent_sample_sd,
        "sd_floor_fraction": sd_floor_fraction,
        "sd_floor": sd_floor,
        "assumed_daily_sd": assumed_daily_sd,
        "assumed_daily_variance": assumed_daily_variance,
        "variance_rule": "max_recent_sample_sd_or_fraction_of_mean",
    }

    return pd.DataFrame([row], columns=CAUTIOUS_ARRIVAL_LEVEL_COLUMNS)


def simulate_cautious_negative_binomial_forecast(
    observed: pd.DataFrame,
    *,
    observed_start_day: int,
    observed_end_day: int,
    forecast_start_day: int,
    forecast_end_day: int,
    n_simulations: int = 10_000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    sd_floor_fraction: float = 0.25,
) -> pd.DataFrame:
    """Simulate a cautious overdispersed forecast for an arbitrary horizon."""

    active_rng = _simulation_rng(seed=seed, rng=rng, n_simulations=n_simulations)
    level = estimate_cautious_arrival_level(
        observed,
        observed_start_day=observed_start_day,
        observed_end_day=observed_end_day,
        sd_floor_fraction=sd_floor_fraction,
    ).iloc[0]
    forecast_days = _forecast_days(
        forecast_start_day=forecast_start_day,
        forecast_end_day=forecast_end_day,
    )

    daily_mean = float(level["updated_daily_arrival_level"])
    daily_variance = float(level["assumed_daily_variance"])
    n, p = _negative_binomial_parameters(mean=daily_mean, variance=daily_variance)
    draws = active_rng.negative_binomial(
        n=n,
        p=p,
        size=(n_simulations, len(forecast_days)),
    )

    return pd.DataFrame(
        {
            "model": "cautious_negative_binomial_updated_level",
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
            "variance_rule": level["variance_rule"],
            "recent_sample_sd": float(level["recent_sample_sd"]),
            "sd_floor": float(level["sd_floor"]),
            "assumed_daily_sd": float(level["assumed_daily_sd"]),
        },
        columns=CAUTIOUS_NEGATIVE_BINOMIAL_SIMULATION_COLUMNS,
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


def summarize_preparedness_forecast_totals(
    simulations: pd.DataFrame,
    *,
    percentiles: tuple[float, ...] = (50, 80, 95),
) -> pd.DataFrame:
    """Summarise forecast totals for direct preparedness translation."""

    _validate_model_simulations(simulations)
    totals = (
        simulations.groupby(["model", "simulation_id"])["simulated_arrivals"]
        .sum()
        .reset_index(name="simulated_total_arrivals")
    )

    rows = []
    for model, model_totals in totals.groupby("model", sort=False):
        values = model_totals["simulated_total_arrivals"]
        for percentile in percentiles:
            rows.append(
                {
                    "model": model,
                    "forecast_reference": _percentile_label(percentile),
                    "forecast_total_arrivals": float(
                        np.percentile(values, percentile)
                    ),
                }
            )

    return pd.DataFrame(rows, columns=PREPAREDNESS_SUMMARY_COLUMNS)


def translate_arrival_totals_to_preparedness(
    forecast_totals: pd.DataFrame,
    *,
    forecast_horizon_days: int,
    water_litres_per_arrival: float = 3,
    food_units_per_arrival: float = 1,
    protection_contacts_per_arrival: float = 1,
    protection_contacts_per_staff_day: int = 60,
    registrations_per_arrival: float = 1,
    medical_team_arrivals_per_day: int = 250,
    minimum_medical_teams: int = 1,
) -> pd.DataFrame:
    """Translate arrival-total percentiles into illustrative preparedness needs."""

    _validate_preparedness_assumptions(
        forecast_horizon_days=forecast_horizon_days,
        water_litres_per_arrival=water_litres_per_arrival,
        food_units_per_arrival=food_units_per_arrival,
        protection_contacts_per_arrival=protection_contacts_per_arrival,
        protection_contacts_per_staff_day=protection_contacts_per_staff_day,
        registrations_per_arrival=registrations_per_arrival,
        medical_team_arrivals_per_day=medical_team_arrivals_per_day,
        minimum_medical_teams=minimum_medical_teams,
    )
    _validate_preparedness_summary(forecast_totals)

    rows = []
    for row in forecast_totals.itertuples(index=False):
        forecast_total = float(row.forecast_total_arrivals)
        average_daily_arrivals = forecast_total / forecast_horizon_days
        water_litres = forecast_total * water_litres_per_arrival
        food_units = forecast_total * food_units_per_arrival
        protection_contacts = forecast_total * protection_contacts_per_arrival
        protection_staff_days = math.ceil(
            protection_contacts / protection_contacts_per_staff_day
        )
        registration_workload = forecast_total * registrations_per_arrival
        medical_teams_per_day = max(
            minimum_medical_teams,
            math.ceil(average_daily_arrivals / medical_team_arrivals_per_day),
        )

        rows.append(
            {
                "model": row.model,
                "forecast_reference": row.forecast_reference,
                "forecast_total_arrivals": forecast_total,
                "forecast_horizon_days": forecast_horizon_days,
                "average_daily_arrivals": average_daily_arrivals,
                "immediate_water_litres": water_litres,
                "immediate_food_units": food_units,
                "protection_contacts": protection_contacts,
                "protection_staff_days": protection_staff_days,
                "registration_workload": registration_workload,
                "medical_teams_per_day": medical_teams_per_day,
                "medical_team_days": forecast_horizon_days * medical_teams_per_day,
            }
        )

    return pd.DataFrame(rows, columns=PREPAREDNESS_COLUMNS)


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


def _validate_preparedness_summary(forecast_totals: pd.DataFrame) -> None:
    required_columns = {
        "model",
        "forecast_reference",
        "forecast_total_arrivals",
    }
    missing_columns = required_columns.difference(forecast_totals.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Forecast totals are missing required columns: {missing}.")
    if forecast_totals.empty:
        raise ValueError("Forecast totals must not be empty.")
    if (forecast_totals["forecast_total_arrivals"] < 0).any():
        raise ValueError("Forecast total arrivals must be non-negative.")


def _validate_preparedness_assumptions(
    *,
    forecast_horizon_days: int,
    water_litres_per_arrival: float,
    food_units_per_arrival: float,
    protection_contacts_per_arrival: float,
    protection_contacts_per_staff_day: int,
    registrations_per_arrival: float,
    medical_team_arrivals_per_day: int,
    minimum_medical_teams: int,
) -> None:
    if forecast_horizon_days <= 0:
        raise ValueError("forecast_horizon_days must be positive.")

    non_negative_parameters = {
        "water_litres_per_arrival": water_litres_per_arrival,
        "food_units_per_arrival": food_units_per_arrival,
        "protection_contacts_per_arrival": protection_contacts_per_arrival,
        "registrations_per_arrival": registrations_per_arrival,
    }
    for name, value in non_negative_parameters.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    positive_parameters = {
        "protection_contacts_per_staff_day": protection_contacts_per_staff_day,
        "medical_team_arrivals_per_day": medical_team_arrivals_per_day,
        "minimum_medical_teams": minimum_medical_teams,
    }
    for name, value in positive_parameters.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _percentile_label(percentile: float) -> str:
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentiles must be between 0 and 100.")
    if float(percentile).is_integer():
        return f"p{int(percentile):02d}"
    return f"p{str(percentile).replace('.', '_')}"
