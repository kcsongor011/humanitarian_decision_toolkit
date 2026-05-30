from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from forecasting.arrival_dynamics import (  # noqa: E402
    mean_baseline_forecast,
    simulate_poisson_forecast,
    summarize_poisson_totals,
)


CANONICAL_CSV_PATH = (
    PROJECT_ROOT / "data" / "processed" / "arrival_dynamics_short_term_master.csv"
)


def _canonical() -> pd.DataFrame:
    return pd.read_csv(CANONICAL_CSV_PATH)


def test_mean_baseline_forecast_uses_canonical_days_one_to_five():
    forecast = mean_baseline_forecast(_canonical())

    assert forecast["day"].tolist() == list(range(6, 13))
    assert forecast["forecast_arrivals"].tolist() == pytest.approx([45.2] * 7)
    assert forecast["observed_mean_arrivals"].tolist() == pytest.approx([45.2] * 7)
    assert forecast["forecast_horizon_days"].tolist() == [7] * 7
    assert forecast["forecast_total_arrivals"].tolist() == pytest.approx(
        [316.4] * 7
    )


def test_poisson_forecast_outputs_only_forecast_horizon_days():
    simulations = simulate_poisson_forecast(
        _canonical(),
        n_simulations=4,
        seed=123,
    )

    assert simulations["day"].tolist() == list(range(6, 13)) * 4
    assert simulations["simulation_id"].tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
    ]
    assert len(simulations) == 28


def test_forecasts_do_not_change_when_hidden_arrivals_change():
    canonical = _canonical()
    changed_future = canonical.copy(deep=True)
    changed_future.loc[changed_future["day"] >= 6, "arrivals"] = 9999

    baseline = mean_baseline_forecast(canonical)
    changed_baseline = mean_baseline_forecast(changed_future)
    poisson = simulate_poisson_forecast(canonical, n_simulations=20, seed=456)
    changed_poisson = simulate_poisson_forecast(
        changed_future,
        n_simulations=20,
        seed=456,
    )

    pd.testing.assert_frame_equal(baseline, changed_baseline)
    pd.testing.assert_frame_equal(poisson, changed_poisson)


def test_extra_latent_columns_do_not_affect_forecasts():
    canonical = _canonical()
    public_only = canonical[["day", "arrivals"]].copy()

    baseline = mean_baseline_forecast(public_only)
    latent_baseline = mean_baseline_forecast(canonical)
    poisson = simulate_poisson_forecast(public_only, n_simulations=20, seed=789)
    latent_poisson = simulate_poisson_forecast(canonical, n_simulations=20, seed=789)

    pd.testing.assert_frame_equal(baseline, latent_baseline)
    pd.testing.assert_frame_equal(poisson, latent_poisson)


def test_observed_window_validation_rejects_missing_incomplete_or_invalid_data():
    with pytest.raises(ValueError, match="missing required columns"):
        mean_baseline_forecast(pd.DataFrame({"day": range(1, 6)}))

    with pytest.raises(ValueError, match="each required day"):
        mean_baseline_forecast(
            pd.DataFrame({"day": [1, 2, 3, 4], "arrivals": [1, 2, 3, 4]})
        )

    with pytest.raises(ValueError, match="duplicate days"):
        mean_baseline_forecast(
            pd.DataFrame(
                {
                    "day": [1, 2, 3, 4, 4, 5],
                    "arrivals": [1, 2, 3, 4, 5, 6],
                }
            )
        )

    with pytest.raises(ValueError, match="non-negative"):
        simulate_poisson_forecast(
            pd.DataFrame({"day": range(1, 6), "arrivals": [1, 2, -3, 4, 5]})
        )


def test_forecast_helpers_reject_invalid_day_windows():
    canonical = _canonical()

    with pytest.raises(ValueError, match="observed_end_day"):
        mean_baseline_forecast(
            canonical,
            observed_start_day=5,
            observed_end_day=1,
        )

    with pytest.raises(ValueError, match="forecast_end_day"):
        simulate_poisson_forecast(
            canonical,
            forecast_start_day=12,
            forecast_end_day=6,
        )


def test_poisson_forecast_seed_controls_reproducibility():
    first = simulate_poisson_forecast(_canonical(), n_simulations=50, seed=101)
    second = simulate_poisson_forecast(_canonical(), n_simulations=50, seed=101)
    different = simulate_poisson_forecast(_canonical(), n_simulations=50, seed=102)

    pd.testing.assert_frame_equal(first, second)
    assert not first["simulated_arrivals"].equals(different["simulated_arrivals"])


def test_poisson_forecast_rng_controls_reproducibility():
    first = simulate_poisson_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(202),
    )
    second = simulate_poisson_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(202),
    )

    pd.testing.assert_frame_equal(first, second)


def test_poisson_forecast_rejects_seed_and_rng_together_or_nonpositive_runs():
    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_poisson_forecast(
            _canonical(),
            seed=1,
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="n_simulations must be positive"):
        simulate_poisson_forecast(_canonical(), n_simulations=0)


def test_summarize_poisson_totals_matches_grouped_totals_and_percentiles():
    simulations = pd.DataFrame(
        {
            "simulation_id": [1, 1, 2, 2, 3, 3],
            "day": [6, 7, 6, 7, 6, 7],
            "simulated_arrivals": [10, 20, 30, 40, 50, 60],
        }
    )

    summary = summarize_poisson_totals(simulations, percentiles=(5, 50, 95))
    totals = simulations.groupby("simulation_id")["simulated_arrivals"].sum()

    assert summary["statistic"].tolist() == [
        "mean",
        "std",
        "min",
        "max",
        "p05",
        "p50",
        "p95",
    ]
    assert summary.loc[
        summary["statistic"] == "mean",
        "value",
    ].iloc[0] == pytest.approx(totals.mean())
    assert summary.loc[
        summary["statistic"] == "std",
        "value",
    ].iloc[0] == pytest.approx(totals.std())
    assert summary.loc[summary["statistic"] == "min", "value"].iloc[0] == totals.min()
    assert summary.loc[summary["statistic"] == "max", "value"].iloc[0] == totals.max()
    assert summary.loc[summary["statistic"] == "p50", "value"].iloc[0] == pytest.approx(
        np.percentile(totals, 50)
    )


def test_summarize_poisson_totals_rejects_invalid_percentiles():
    simulations = simulate_poisson_forecast(_canonical(), n_simulations=5, seed=404)

    with pytest.raises(ValueError, match="between 0 and 100"):
        summarize_poisson_totals(simulations, percentiles=(-1,))

    with pytest.raises(ValueError, match="between 0 and 100"):
        summarize_poisson_totals(simulations, percentiles=(101,))


def test_forecast_helpers_do_not_mutate_input_data():
    canonical = _canonical()
    original = canonical.copy(deep=True)

    mean_baseline_forecast(canonical)
    simulations = simulate_poisson_forecast(canonical, n_simulations=10, seed=303)
    summarize_poisson_totals(simulations)

    pd.testing.assert_frame_equal(canonical, original)
