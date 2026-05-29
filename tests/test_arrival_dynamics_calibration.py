from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation.arrival_dynamics import (  # noqa: E402
    build_diagnostics_table,
    calculate_candidate_diagnostics,
    generate_candidate,
    summarize_candidate_windows,
)


EXPECTED_DIAGNOSTIC_COLUMNS = [
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


def _manual_candidate() -> pd.DataFrame:
    arrivals = [
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        180,
        210,
        240,
        210,
        180,
        170,
        160,
        150,
        140,
        130,
        100,
        110,
        120,
        130,
        120,
        110,
        100,
    ]
    return pd.DataFrame({"day": range(1, 31), "arrivals": arrivals})


def test_summarize_candidate_windows_calculates_manual_window_metrics():
    candidate = _manual_candidate()

    summary = summarize_candidate_windows(candidate)
    early = summary.loc[summary["window"] == "days_1_5"].iloc[0]
    high_pressure = summary.loc[summary["window"] == "days_13_18"].iloc[0]

    early_arrivals = candidate.loc[candidate["day"].between(1, 5), "arrivals"]
    high_arrivals = candidate.loc[candidate["day"].between(13, 18), "arrivals"]

    assert early["start_day"] == 1
    assert early["end_day"] == 5
    assert early["n_days"] == 5
    assert early["total_arrivals"] == early_arrivals.sum()
    assert early["mean_arrivals"] == pytest.approx(early_arrivals.mean())
    assert early["std_arrivals"] == pytest.approx(early_arrivals.std())
    assert early["min_arrivals"] == early_arrivals.min()
    assert early["peak_arrivals"] == early_arrivals.max()
    assert early["coefficient_of_variation"] == pytest.approx(
        early_arrivals.std() / early_arrivals.mean()
    )

    assert high_pressure["peak_arrivals"] == high_arrivals.max()
    assert high_pressure["coefficient_of_variation"] == pytest.approx(
        high_arrivals.std() / high_arrivals.mean()
    )


def test_calculate_candidate_diagnostics_calculates_manual_metrics():
    candidate = _manual_candidate()

    diagnostics = calculate_candidate_diagnostics(
        candidate,
        candidate_seed=999,
    ).iloc[0]

    early = candidate.loc[candidate["day"].between(1, 5), "arrivals"]
    surge = candidate.loc[candidate["day"].between(6, 12), "arrivals"]
    high_pressure = candidate.loc[candidate["day"].between(13, 18), "arrivals"]
    stabilisation = candidate.loc[candidate["day"].between(24, 30), "arrivals"]

    baseline_total = early.mean() * 7
    high_pressure_cv = high_pressure.std() / high_pressure.mean()
    stabilisation_cv = stabilisation.std() / stabilisation.mean()

    assert diagnostics["candidate_seed"] == 999
    assert diagnostics["early_mean_days_1_5"] == pytest.approx(early.mean())
    assert diagnostics["early_net_change_days_1_5"] == 40
    assert diagnostics["early_linear_slope_days_1_5"] == pytest.approx(10.0)
    assert diagnostics["baseline_total_forecast_days_6_12"] == pytest.approx(
        baseline_total
    )
    assert diagnostics["realised_total_days_6_12"] == surge.sum()
    assert diagnostics["surge_excess_arrivals_days_6_12"] == pytest.approx(
        surge.sum() - baseline_total
    )
    assert diagnostics["surge_realised_to_baseline_ratio_days_6_12"] == (
        pytest.approx(surge.sum() / baseline_total)
    )
    assert diagnostics["surge_peak_days_6_12"] == surge.max()
    assert diagnostics["high_pressure_mean_days_13_18"] == pytest.approx(
        high_pressure.mean()
    )
    assert diagnostics["high_pressure_std_days_13_18"] == pytest.approx(
        high_pressure.std()
    )
    assert diagnostics["high_pressure_cv_days_13_18"] == pytest.approx(
        high_pressure_cv
    )
    assert diagnostics["high_pressure_peak_days_13_18"] == high_pressure.max()
    assert diagnostics["partial_stabilisation_mean_days_24_30"] == pytest.approx(
        stabilisation.mean()
    )
    assert diagnostics["partial_stabilisation_std_days_24_30"] == pytest.approx(
        stabilisation.std()
    )
    assert diagnostics["partial_stabilisation_cv_days_24_30"] == pytest.approx(
        stabilisation_cv
    )
    assert diagnostics["high_to_stabilisation_mean_ratio"] == pytest.approx(
        high_pressure.mean() / stabilisation.mean()
    )
    assert diagnostics["high_minus_stabilisation_std"] == pytest.approx(
        high_pressure.std() - stabilisation.std()
    )
    assert diagnostics["high_to_stabilisation_cv_ratio"] == pytest.approx(
        high_pressure_cv / stabilisation_cv
    )


def test_build_diagnostics_table_is_deterministic_for_fixed_seeds():
    first = build_diagnostics_table([101, 202])
    second = build_diagnostics_table([101, 202])

    pd.testing.assert_frame_equal(first, second)
    assert first["candidate_seed"].tolist() == [101, 202]


def test_diagnostics_table_has_expected_schema():
    diagnostics = build_diagnostics_table([101])

    assert list(diagnostics.columns) == EXPECTED_DIAGNOSTIC_COLUMNS


def test_window_summary_has_expected_schema_without_duplicate_peak_columns():
    summary = summarize_candidate_windows(_manual_candidate())

    assert summary["window"].tolist() == [
        "days_1_5",
        "days_6_12",
        "days_13_18",
        "days_19_23",
        "days_24_30",
    ]
    assert "peak_arrivals" in summary.columns
    assert "max_arrivals" not in summary.columns


def test_diagnostics_do_not_mutate_candidate_data():
    candidate = generate_candidate(seed=42)
    original = candidate.copy(deep=True)

    summarize_candidate_windows(candidate)
    calculate_candidate_diagnostics(candidate, candidate_seed=42)

    pd.testing.assert_frame_equal(candidate, original)


def test_diagnostics_do_not_include_selection_columns_or_file_output_arguments():
    diagnostics = build_diagnostics_table([101, 202])
    summary = summarize_candidate_windows(_manual_candidate())
    prohibited_terms = {
        "score",
        "rank",
        "pass",
        "fail",
        "accepted",
        "selected",
        "canonical",
    }

    for columns in (diagnostics.columns, summary.columns):
        for column in columns:
            assert not any(term in column.lower() for term in prohibited_terms)

    for function in (
        summarize_candidate_windows,
        calculate_candidate_diagnostics,
        build_diagnostics_table,
    ):
        assert "output_path" not in inspect.signature(function).parameters


def test_candidate_validation_requires_required_columns_and_day_coverage():
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_candidate_windows(pd.DataFrame({"day": range(1, 31)}))

    with pytest.raises(ValueError, match="days 1 through 30"):
        calculate_candidate_diagnostics(
            pd.DataFrame({"day": range(2, 32), "arrivals": np.arange(30)})
        )
