"""Operations and backlog helpers for Arrival Dynamics 1."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


DAILY_BACKLOG_COLUMNS = [
    "day",
    "arrivals",
    "officers",
    "daily_registration_capacity",
    "backlog_start",
    "processed_registrations",
    "backlog_end",
    "backlog_person_days",
    "reception_capacity_threshold",
    "reception_capacity_exceeded",
]

SIMULATED_BACKLOG_COLUMNS = [
    "model",
    "simulation_id",
    "day",
    "simulated_arrivals",
    "officers",
    "daily_registration_capacity",
    "backlog_start",
    "processed_registrations",
    "backlog_end",
    "backlog_person_days",
    "reception_capacity_threshold",
    "reception_capacity_exceeded",
]

BACKLOG_RISK_COLUMNS = [
    "model",
    "officers",
    "daily_registration_capacity",
    "n_simulations",
    "probability_any_backlog",
    "probability_reception_capacity_exceeded",
    "median_max_backlog",
    "p90_max_backlog",
    "probability_backlog_persists_at_horizon_end",
    "median_backlog_person_days",
    "p90_backlog_person_days",
]

EXTENDED_STAY_COLUMNS = BACKLOG_RISK_COLUMNS + [
    "water_litres_per_backlog_person_day",
    "food_units_per_backlog_person_day",
    "median_extended_stay_water_litres",
    "p90_extended_stay_water_litres",
    "median_extended_stay_food_units",
    "p90_extended_stay_food_units",
]


def calculate_daily_registration_capacity(
    *,
    officers: int,
    operating_hours_per_day: float = 8,
    registration_minutes_per_person: float = 15,
) -> int:
    """Calculate daily registration capacity for a staffing level."""

    _validate_positive_integer(officers, "officers")
    _validate_positive_number(operating_hours_per_day, "operating_hours_per_day")
    _validate_positive_number(
        registration_minutes_per_person,
        "registration_minutes_per_person",
    )

    capacity_per_officer = math.floor(
        (operating_hours_per_day * 60) / registration_minutes_per_person
    )
    return officers * capacity_per_officer


def calculate_daily_backlog(
    arrivals: pd.DataFrame,
    *,
    officers: int,
    initial_backlog: int = 0,
    reception_capacity_threshold: int = 60,
    operating_hours_per_day: float = 8,
    registration_minutes_per_person: float = 15,
) -> pd.DataFrame:
    """Calculate recursive registration backlog for one realised arrival path."""

    _validate_backlog_parameters(
        officers=officers,
        initial_backlog=initial_backlog,
        reception_capacity_threshold=reception_capacity_threshold,
        operating_hours_per_day=operating_hours_per_day,
        registration_minutes_per_person=registration_minutes_per_person,
    )
    _validate_single_path_arrivals(arrivals)

    daily_capacity = calculate_daily_registration_capacity(
        officers=officers,
        operating_hours_per_day=operating_hours_per_day,
        registration_minutes_per_person=registration_minutes_per_person,
    )
    path = arrivals.loc[:, ["day", "arrivals"]].copy().sort_values("day")
    rows = _calculate_path_backlog_rows(
        days=path["day"].tolist(),
        arrivals=path["arrivals"].tolist(),
        officers=officers,
        daily_capacity=daily_capacity,
        initial_backlog=initial_backlog,
        reception_capacity_threshold=reception_capacity_threshold,
    )

    return pd.DataFrame(rows, columns=DAILY_BACKLOG_COLUMNS)


def simulate_backlog_by_staffing(
    simulations: pd.DataFrame,
    *,
    staffing_options: tuple[int, ...] = (3, 4, 5, 6, 7),
    initial_backlog: int = 0,
    reception_capacity_threshold: int = 60,
    operating_hours_per_day: float = 8,
    registration_minutes_per_person: float = 15,
) -> pd.DataFrame:
    """Apply the daily backlog recursion across simulated forecast paths."""

    _validate_staffing_options(staffing_options)
    _validate_non_negative_integer(initial_backlog, "initial_backlog")
    _validate_non_negative_integer(
        reception_capacity_threshold,
        "reception_capacity_threshold",
    )
    _validate_positive_number(operating_hours_per_day, "operating_hours_per_day")
    _validate_positive_number(
        registration_minutes_per_person,
        "registration_minutes_per_person",
    )
    _validate_simulation_arrivals(simulations)

    working = simulations.copy()
    if "model" not in working.columns:
        working["model"] = "simulation"

    rows: list[dict[str, object]] = []
    group_columns = ["model", "simulation_id"]
    for (model, simulation_id), path in working.groupby(group_columns, sort=False):
        sorted_path = path.sort_values("day")
        for officers in staffing_options:
            daily_capacity = calculate_daily_registration_capacity(
                officers=officers,
                operating_hours_per_day=operating_hours_per_day,
                registration_minutes_per_person=registration_minutes_per_person,
            )
            path_rows = _calculate_path_backlog_rows(
                days=sorted_path["day"].tolist(),
                arrivals=sorted_path["simulated_arrivals"].tolist(),
                officers=officers,
                daily_capacity=daily_capacity,
                initial_backlog=initial_backlog,
                reception_capacity_threshold=reception_capacity_threshold,
            )
            for row in path_rows:
                rows.append(
                    {
                        "model": model,
                        "simulation_id": simulation_id,
                        "day": row["day"],
                        "simulated_arrivals": row["arrivals"],
                        "officers": row["officers"],
                        "daily_registration_capacity": row[
                            "daily_registration_capacity"
                        ],
                        "backlog_start": row["backlog_start"],
                        "processed_registrations": row["processed_registrations"],
                        "backlog_end": row["backlog_end"],
                        "backlog_person_days": row["backlog_person_days"],
                        "reception_capacity_threshold": row[
                            "reception_capacity_threshold"
                        ],
                        "reception_capacity_exceeded": row[
                            "reception_capacity_exceeded"
                        ],
                    }
                )

    return pd.DataFrame(rows, columns=SIMULATED_BACKLOG_COLUMNS)


def summarize_backlog_risk(simulated_backlog: pd.DataFrame) -> pd.DataFrame:
    """Summarise backlog risk across simulation paths by model and staffing."""

    _validate_simulated_backlog(simulated_backlog)

    path_rows = []
    group_columns = ["model", "officers", "simulation_id"]
    for (model, officers, simulation_id), path in simulated_backlog.groupby(
        group_columns,
        sort=False,
    ):
        sorted_path = path.sort_values("day")
        path_rows.append(
            {
                "model": model,
                "officers": officers,
                "simulation_id": simulation_id,
                "daily_registration_capacity": int(
                    sorted_path["daily_registration_capacity"].iloc[0]
                ),
                "any_backlog": bool((sorted_path["backlog_end"] > 0).any()),
                "reception_capacity_exceeded": bool(
                    sorted_path["reception_capacity_exceeded"].any()
                ),
                "max_backlog": float(sorted_path["backlog_end"].max()),
                "backlog_persists_at_horizon_end": bool(
                    sorted_path["backlog_end"].iloc[-1] > 0
                ),
                "backlog_person_days": float(
                    sorted_path["backlog_person_days"].sum()
                ),
            }
        )

    path_summary = pd.DataFrame(path_rows)
    rows = []
    for (model, officers), group in path_summary.groupby(
        ["model", "officers"],
        sort=False,
    ):
        rows.append(
            {
                "model": model,
                "officers": int(officers),
                "daily_registration_capacity": int(
                    group["daily_registration_capacity"].iloc[0]
                ),
                "n_simulations": int(group["simulation_id"].nunique()),
                "probability_any_backlog": float(group["any_backlog"].mean()),
                "probability_reception_capacity_exceeded": float(
                    group["reception_capacity_exceeded"].mean()
                ),
                "median_max_backlog": _percentile(group["max_backlog"], 50),
                "p90_max_backlog": _percentile(group["max_backlog"], 90),
                "probability_backlog_persists_at_horizon_end": float(
                    group["backlog_persists_at_horizon_end"].mean()
                ),
                "median_backlog_person_days": _percentile(
                    group["backlog_person_days"],
                    50,
                ),
                "p90_backlog_person_days": _percentile(
                    group["backlog_person_days"],
                    90,
                ),
            }
        )

    return pd.DataFrame(rows, columns=BACKLOG_RISK_COLUMNS)


def calculate_extended_stay_needs(
    backlog_person_days: float,
    *,
    water_litres_per_backlog_person_day: float = 12,
    food_units_per_backlog_person_day: float = 1,
) -> dict[str, float]:
    """Calculate extended-stay water and food needs from backlog person-days."""

    _validate_non_negative_number(backlog_person_days, "backlog_person_days")
    _validate_positive_number(
        water_litres_per_backlog_person_day,
        "water_litres_per_backlog_person_day",
    )
    _validate_positive_number(
        food_units_per_backlog_person_day,
        "food_units_per_backlog_person_day",
    )

    return {
        "extended_stay_water_litres": (
            backlog_person_days * water_litres_per_backlog_person_day
        ),
        "extended_stay_food_units": (
            backlog_person_days * food_units_per_backlog_person_day
        ),
    }


def summarize_extended_stay_needs_by_staffing(
    backlog_risk_summary: pd.DataFrame,
    *,
    water_litres_per_backlog_person_day: float = 12,
    food_units_per_backlog_person_day: float = 1,
) -> pd.DataFrame:
    """Append median and p90 extended-stay needs to backlog risk summaries."""

    _validate_backlog_risk_summary(backlog_risk_summary)
    _validate_positive_number(
        water_litres_per_backlog_person_day,
        "water_litres_per_backlog_person_day",
    )
    _validate_positive_number(
        food_units_per_backlog_person_day,
        "food_units_per_backlog_person_day",
    )

    summary = backlog_risk_summary.copy()
    summary["water_litres_per_backlog_person_day"] = (
        water_litres_per_backlog_person_day
    )
    summary["food_units_per_backlog_person_day"] = food_units_per_backlog_person_day
    summary["median_extended_stay_water_litres"] = (
        summary["median_backlog_person_days"] * water_litres_per_backlog_person_day
    )
    summary["p90_extended_stay_water_litres"] = (
        summary["p90_backlog_person_days"] * water_litres_per_backlog_person_day
    )
    summary["median_extended_stay_food_units"] = (
        summary["median_backlog_person_days"] * food_units_per_backlog_person_day
    )
    summary["p90_extended_stay_food_units"] = (
        summary["p90_backlog_person_days"] * food_units_per_backlog_person_day
    )

    if "model" in summary.columns:
        columns = EXTENDED_STAY_COLUMNS
    else:
        columns = [column for column in EXTENDED_STAY_COLUMNS if column != "model"]

    return summary.loc[:, columns]


def _calculate_path_backlog_rows(
    *,
    days: list[int],
    arrivals: list[int],
    officers: int,
    daily_capacity: int,
    initial_backlog: int,
    reception_capacity_threshold: int,
) -> list[dict[str, object]]:
    backlog_start = initial_backlog
    rows: list[dict[str, object]] = []

    for day, arrival_count in zip(days, arrivals):
        available = backlog_start + int(arrival_count)
        processed = min(available, daily_capacity)
        backlog_end = available - processed
        rows.append(
            {
                "day": int(day),
                "arrivals": int(arrival_count),
                "officers": officers,
                "daily_registration_capacity": daily_capacity,
                "backlog_start": int(backlog_start),
                "processed_registrations": int(processed),
                "backlog_end": int(backlog_end),
                "backlog_person_days": int(backlog_end),
                "reception_capacity_threshold": reception_capacity_threshold,
                "reception_capacity_exceeded": (
                    backlog_end > reception_capacity_threshold
                ),
            }
        )
        backlog_start = backlog_end

    return rows


def _validate_single_path_arrivals(arrivals: pd.DataFrame) -> None:
    required_columns = {"day", "arrivals"}
    _validate_required_columns(arrivals, required_columns, "Arrivals")
    if arrivals.empty:
        raise ValueError("Arrivals must not be empty.")
    if arrivals["day"].duplicated().any():
        raise ValueError("Arrivals must contain unique days.")
    if (arrivals["arrivals"] < 0).any():
        raise ValueError("Arrivals must be non-negative.")


def _validate_simulation_arrivals(simulations: pd.DataFrame) -> None:
    required_columns = {"simulation_id", "day", "simulated_arrivals"}
    _validate_required_columns(simulations, required_columns, "Simulations")
    if simulations.empty:
        raise ValueError("Simulations must not be empty.")

    grouping_columns = ["simulation_id", "day"]
    if "model" in simulations.columns:
        grouping_columns = ["model"] + grouping_columns
    if simulations.duplicated(subset=grouping_columns).any():
        raise ValueError(
            "Simulations must contain unique days within each model and "
            "simulation_id path."
        )
    if (simulations["simulated_arrivals"] < 0).any():
        raise ValueError("Simulated arrivals must be non-negative.")


def _validate_simulated_backlog(simulated_backlog: pd.DataFrame) -> None:
    if simulated_backlog.empty:
        raise ValueError("Simulated backlog must not be empty.")

    required_columns = set(SIMULATED_BACKLOG_COLUMNS)
    _validate_required_columns(
        simulated_backlog,
        required_columns,
        "Simulated backlog",
    )
    if simulated_backlog.duplicated(
        subset=["model", "officers", "simulation_id", "day"]
    ).any():
        raise ValueError(
            "Simulated backlog must contain unique days within each model, "
            "officers, and simulation_id path."
        )


def _validate_backlog_risk_summary(backlog_risk_summary: pd.DataFrame) -> None:
    if backlog_risk_summary.empty:
        raise ValueError("Backlog risk summary must not be empty.")

    required_columns = set(BACKLOG_RISK_COLUMNS)
    if "model" not in backlog_risk_summary.columns:
        required_columns.remove("model")
    _validate_required_columns(
        backlog_risk_summary,
        required_columns,
        "Backlog risk summary",
    )
    for column in ("median_backlog_person_days", "p90_backlog_person_days"):
        if (backlog_risk_summary[column] < 0).any():
            raise ValueError(f"{column} must be non-negative.")


def _validate_backlog_parameters(
    *,
    officers: int,
    initial_backlog: int,
    reception_capacity_threshold: int,
    operating_hours_per_day: float,
    registration_minutes_per_person: float,
) -> None:
    _validate_positive_integer(officers, "officers")
    _validate_non_negative_integer(initial_backlog, "initial_backlog")
    _validate_non_negative_integer(
        reception_capacity_threshold,
        "reception_capacity_threshold",
    )
    _validate_positive_number(operating_hours_per_day, "operating_hours_per_day")
    _validate_positive_number(
        registration_minutes_per_person,
        "registration_minutes_per_person",
    )


def _validate_staffing_options(staffing_options: tuple[int, ...]) -> None:
    if not staffing_options:
        raise ValueError("staffing_options must not be empty.")
    for officers in staffing_options:
        _validate_positive_integer(officers, "staffing_options")


def _validate_required_columns(
    frame: pd.DataFrame,
    required_columns: set[str],
    frame_name: str,
) -> None:
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{frame_name} is missing required columns: {missing}.")


def _validate_positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_positive_number(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative_number(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _percentile(values: pd.Series, percentile: float) -> float:
    return float(np.percentile(values, percentile))
