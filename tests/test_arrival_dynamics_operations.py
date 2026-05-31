from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from operations.arrival_dynamics import (  # noqa: E402
    calculate_daily_backlog,
    calculate_daily_registration_capacity,
    calculate_extended_stay_needs,
    simulate_backlog_by_staffing,
    summarize_backlog_risk,
    summarize_extended_stay_needs_by_staffing,
)


def test_daily_registration_capacity_uses_locked_assumptions():
    assert calculate_daily_registration_capacity(officers=1) == 32
    assert calculate_daily_registration_capacity(officers=3) == 96
    assert calculate_daily_registration_capacity(officers=7) == 224


def test_daily_backlog_calculates_recursive_path():
    arrivals = pd.DataFrame(
        {
            "day": [1, 2, 3],
            "arrivals": [100, 50, 20],
        }
    )

    backlog = calculate_daily_backlog(arrivals, officers=2)

    assert backlog.to_dict("records") == [
        {
            "day": 1,
            "arrivals": 100,
            "officers": 2,
            "daily_registration_capacity": 64,
            "backlog_start": 0,
            "processed_registrations": 64,
            "backlog_end": 36,
            "backlog_person_days": 36,
            "reception_capacity_threshold": 60,
            "reception_capacity_exceeded": False,
        },
        {
            "day": 2,
            "arrivals": 50,
            "officers": 2,
            "daily_registration_capacity": 64,
            "backlog_start": 36,
            "processed_registrations": 64,
            "backlog_end": 22,
            "backlog_person_days": 22,
            "reception_capacity_threshold": 60,
            "reception_capacity_exceeded": False,
        },
        {
            "day": 3,
            "arrivals": 20,
            "officers": 2,
            "daily_registration_capacity": 64,
            "backlog_start": 22,
            "processed_registrations": 42,
            "backlog_end": 0,
            "backlog_person_days": 0,
            "reception_capacity_threshold": 60,
            "reception_capacity_exceeded": False,
        },
    ]


def test_daily_backlog_applies_initial_backlog_and_strict_threshold():
    arrivals = pd.DataFrame({"day": [1, 2], "arrivals": [124, 1]})

    backlog = calculate_daily_backlog(
        arrivals,
        officers=2,
        initial_backlog=0,
        reception_capacity_threshold=60,
    )

    assert backlog["backlog_start"].tolist() == [0, 60]
    assert backlog["backlog_end"].tolist() == [60, 0]
    assert backlog["reception_capacity_exceeded"].tolist() == [False, False]

    exceeded = calculate_daily_backlog(arrivals, officers=2, initial_backlog=1)
    assert exceeded["backlog_end"].tolist()[0] == 61
    assert exceeded["reception_capacity_exceeded"].tolist()[0] is True


def test_simulate_backlog_by_staffing_preserves_paths_and_models():
    simulations = pd.DataFrame(
        {
            "model": ["model_a", "model_a", "model_a", "model_a"],
            "simulation_id": [1, 1, 2, 2],
            "day": [1, 2, 1, 2],
            "simulated_arrivals": [100, 20, 50, 200],
        }
    )

    backlog = simulate_backlog_by_staffing(
        simulations,
        staffing_options=(2, 3),
    )

    assert set(backlog["model"]) == {"model_a"}
    assert set(backlog["simulation_id"]) == {1, 2}
    assert set(backlog["officers"]) == {2, 3}
    assert len(backlog) == 8

    path = backlog.loc[
        (backlog["simulation_id"] == 2) & (backlog["officers"] == 2)
    ].sort_values("day")
    assert path["backlog_start"].tolist() == [0, 0]
    assert path["backlog_end"].tolist() == [0, 136]
    assert path["reception_capacity_exceeded"].tolist() == [False, True]


def test_simulate_backlog_by_staffing_adds_placeholder_model_when_missing():
    simulations = pd.DataFrame(
        {
            "simulation_id": [1],
            "day": [1],
            "simulated_arrivals": [10],
        }
    )

    backlog = simulate_backlog_by_staffing(simulations, staffing_options=(3,))

    assert backlog["model"].tolist() == ["simulation"]


def test_summarize_backlog_risk_aggregates_over_simulation_paths():
    simulated_backlog = pd.DataFrame(
        {
            "model": ["m"] * 6,
            "simulation_id": [1, 1, 2, 2, 3, 3],
            "day": [1, 2, 1, 2, 1, 2],
            "simulated_arrivals": [0, 0, 0, 0, 0, 0],
            "officers": [3] * 6,
            "daily_registration_capacity": [96] * 6,
            "backlog_start": [0, 0, 0, 30, 0, 70],
            "processed_registrations": [0, 0, 0, 0, 0, 0],
            "backlog_end": [0, 0, 30, 20, 70, 50],
            "backlog_person_days": [0, 0, 30, 20, 70, 50],
            "reception_capacity_threshold": [60] * 6,
            "reception_capacity_exceeded": [False, False, False, False, True, False],
        }
    )

    summary = summarize_backlog_risk(simulated_backlog).iloc[0]

    assert summary["model"] == "m"
    assert summary["officers"] == 3
    assert summary["n_simulations"] == 3
    assert summary["probability_any_backlog"] == pytest.approx(2 / 3)
    assert summary["probability_reception_capacity_exceeded"] == pytest.approx(1 / 3)
    assert summary["median_max_backlog"] == pytest.approx(30)
    assert summary["p90_max_backlog"] == pytest.approx(62)
    assert summary["probability_backlog_persists_at_horizon_end"] == pytest.approx(
        2 / 3
    )
    assert summary["median_backlog_person_days"] == pytest.approx(50)
    assert summary["p90_backlog_person_days"] == pytest.approx(106)


def test_extended_stay_needs_use_locked_rates_and_preserve_model():
    needs = calculate_extended_stay_needs(10)

    assert needs == {
        "extended_stay_water_litres": 120,
        "extended_stay_food_units": 10,
    }

    risk_summary = pd.DataFrame(
        {
            "model": ["m1", "m2"],
            "officers": [3, 3],
            "daily_registration_capacity": [96, 96],
            "n_simulations": [2, 2],
            "probability_any_backlog": [1.0, 0.5],
            "probability_reception_capacity_exceeded": [0.5, 0.0],
            "median_max_backlog": [20.0, 10.0],
            "p90_max_backlog": [40.0, 30.0],
            "probability_backlog_persists_at_horizon_end": [0.5, 0.0],
            "median_backlog_person_days": [10.0, 5.0],
            "p90_backlog_person_days": [30.0, 20.0],
        }
    )

    summary = summarize_extended_stay_needs_by_staffing(risk_summary)

    assert summary["model"].tolist() == ["m1", "m2"]
    assert summary["median_extended_stay_water_litres"].tolist() == [120.0, 60.0]
    assert summary["p90_extended_stay_water_litres"].tolist() == [360.0, 240.0]
    assert summary["median_extended_stay_food_units"].tolist() == [10.0, 5.0]
    assert summary["p90_extended_stay_food_units"].tolist() == [30.0, 20.0]


def test_extended_stay_summary_can_handle_model_less_input():
    risk_summary = pd.DataFrame(
        {
            "officers": [3],
            "daily_registration_capacity": [96],
            "n_simulations": [2],
            "probability_any_backlog": [1.0],
            "probability_reception_capacity_exceeded": [0.5],
            "median_max_backlog": [20.0],
            "p90_max_backlog": [40.0],
            "probability_backlog_persists_at_horizon_end": [0.5],
            "median_backlog_person_days": [10.0],
            "p90_backlog_person_days": [30.0],
        }
    )

    summary = summarize_extended_stay_needs_by_staffing(risk_summary)

    assert "model" not in summary.columns
    assert summary["median_extended_stay_water_litres"].tolist() == [120.0]


def test_helpers_do_not_mutate_inputs():
    arrivals = pd.DataFrame({"day": [2, 1], "arrivals": [10, 20]})
    simulations = pd.DataFrame(
        {
            "model": ["m"],
            "simulation_id": [1],
            "day": [1],
            "simulated_arrivals": [10],
        }
    )
    arrivals_original = arrivals.copy(deep=True)
    simulations_original = simulations.copy(deep=True)

    backlog = calculate_daily_backlog(arrivals, officers=3)
    simulated_backlog = simulate_backlog_by_staffing(simulations, staffing_options=(3,))
    risk = summarize_backlog_risk(simulated_backlog)
    summarize_extended_stay_needs_by_staffing(risk)

    pd.testing.assert_frame_equal(arrivals, arrivals_original)
    pd.testing.assert_frame_equal(simulations, simulations_original)
    assert backlog["day"].tolist() == [1, 2]


def test_validation_rejects_invalid_single_path_inputs():
    with pytest.raises(ValueError, match="missing required columns"):
        calculate_daily_backlog(pd.DataFrame({"day": [1]}), officers=3)
    with pytest.raises(ValueError, match="unique days"):
        calculate_daily_backlog(
            pd.DataFrame({"day": [1, 1], "arrivals": [1, 2]}),
            officers=3,
        )
    with pytest.raises(ValueError, match="non-negative"):
        calculate_daily_backlog(
            pd.DataFrame({"day": [1], "arrivals": [-1]}),
            officers=3,
        )
    with pytest.raises(ValueError, match="positive integer"):
        calculate_daily_backlog(pd.DataFrame({"day": [1], "arrivals": [1]}), officers=0)


def test_validation_rejects_invalid_simulation_inputs():
    valid = pd.DataFrame(
        {
            "model": ["m"],
            "simulation_id": [1],
            "day": [1],
            "simulated_arrivals": [1],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        simulate_backlog_by_staffing(pd.DataFrame({"day": [1]}))
    with pytest.raises(ValueError, match="unique days"):
        simulate_backlog_by_staffing(pd.concat([valid, valid], ignore_index=True))
    with pytest.raises(ValueError, match="non-negative"):
        simulate_backlog_by_staffing(
            pd.DataFrame(
                {
                    "simulation_id": [1],
                    "day": [1],
                    "simulated_arrivals": [-1],
                }
            )
        )
    with pytest.raises(ValueError, match="must not be empty"):
        simulate_backlog_by_staffing(valid, staffing_options=())
    with pytest.raises(ValueError, match="positive integer"):
        simulate_backlog_by_staffing(valid, staffing_options=(0,))


def test_validation_rejects_invalid_summary_and_resource_inputs():
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_backlog_risk(pd.DataFrame())
    with pytest.raises(ValueError, match="non-negative"):
        calculate_extended_stay_needs(-1)
    with pytest.raises(ValueError, match="positive"):
        calculate_extended_stay_needs(1, water_litres_per_backlog_person_day=0)
    with pytest.raises(ValueError, match="must not be empty"):
        summarize_extended_stay_needs_by_staffing(pd.DataFrame())
