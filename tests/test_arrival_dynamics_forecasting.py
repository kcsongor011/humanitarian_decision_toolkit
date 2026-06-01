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
    estimate_cautious_arrival_level,
    estimate_recent_arrival_level,
    mean_baseline_forecast,
    simulate_cautious_negative_binomial_forecast,
    simulate_poisson_forecast,
    simulate_updated_negative_binomial_forecast,
    simulate_updated_poisson_forecast,
    summarize_forecast_totals_by_model,
    summarize_preparedness_forecast_totals,
    summarize_poisson_totals,
    translate_arrival_totals_to_preparedness,
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


def test_estimate_recent_arrival_level_uses_canonical_days_six_to_twelve():
    level = estimate_recent_arrival_level(_canonical())

    assert level.to_dict("records") == [
        {
            "observed_start_day": 6,
            "observed_end_day": 12,
            "observed_days": 7,
            "updated_daily_arrival_level": pytest.approx(122.57142857142857),
            "observed_sample_variance": pytest.approx(1353.2857142857144),
            "observed_sample_cv": pytest.approx(0.30012729969206725),
        }
    ]


def test_updated_forecasts_output_only_default_forecast_horizon_days():
    poisson = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=4,
        seed=123,
    )
    negative_binomial = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=4,
        seed=123,
    )

    assert poisson["day"].tolist() == list(range(13, 19)) * 4
    assert negative_binomial["day"].tolist() == list(range(13, 19)) * 4
    assert poisson["simulation_id"].tolist() == [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6
    assert negative_binomial["simulation_id"].tolist() == (
        [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6
    )
    assert len(poisson) == 24
    assert len(negative_binomial) == 24
    assert poisson["model"].unique().tolist() == ["poisson_updated_level"]
    assert negative_binomial["model"].unique().tolist() == [
        "negative_binomial_updated_level"
    ]


def test_updated_negative_binomial_parameters_match_mean_variance_conversion():
    simulations = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=2,
        seed=321,
    )
    mu = 122.57142857142857
    var = 1353.2857142857144

    assert simulations["assumed_daily_mean"].unique().tolist() == pytest.approx([mu])
    assert simulations["assumed_daily_variance"].unique().tolist() == pytest.approx(
        [var]
    )
    assert simulations["negative_binomial_p"].unique().tolist() == pytest.approx(
        [mu / var]
    )
    assert simulations["negative_binomial_n"].unique().tolist() == pytest.approx(
        [mu**2 / (var - mu)]
    )


def test_updated_forecast_seed_controls_reproducibility():
    poisson_first = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=50,
        seed=101,
    )
    poisson_second = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=50,
        seed=101,
    )
    poisson_different = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=50,
        seed=102,
    )
    nb_first = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=50,
        seed=201,
    )
    nb_second = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=50,
        seed=201,
    )
    nb_different = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=50,
        seed=202,
    )

    pd.testing.assert_frame_equal(poisson_first, poisson_second)
    pd.testing.assert_frame_equal(nb_first, nb_second)
    assert not poisson_first["simulated_arrivals"].equals(
        poisson_different["simulated_arrivals"]
    )
    assert not nb_first["simulated_arrivals"].equals(nb_different["simulated_arrivals"])


def test_updated_forecast_rng_controls_reproducibility():
    poisson_first = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(202),
    )
    poisson_second = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(202),
    )
    nb_first = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(303),
    )
    nb_second = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=20,
        rng=np.random.default_rng(303),
    )

    pd.testing.assert_frame_equal(poisson_first, poisson_second)
    pd.testing.assert_frame_equal(nb_first, nb_second)


def test_updated_forecasts_reject_seed_and_rng_together_or_nonpositive_runs():
    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_updated_poisson_forecast(
            _canonical(),
            seed=1,
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="n_simulations must be positive"):
        simulate_updated_poisson_forecast(_canonical(), n_simulations=0)

    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_updated_negative_binomial_forecast(
            _canonical(),
            seed=1,
            rng=np.random.default_rng(1),
        )

    with pytest.raises(ValueError, match="n_simulations must be positive"):
        simulate_updated_negative_binomial_forecast(_canonical(), n_simulations=0)


def test_updated_negative_binomial_rejects_variance_not_greater_than_mean():
    low_variance_observed = pd.DataFrame(
        {
            "day": range(6, 13),
            "arrivals": [100, 100, 100, 100, 100, 100, 100],
        }
    )

    with pytest.raises(ValueError, match="variance must be greater"):
        simulate_updated_negative_binomial_forecast(low_variance_observed)


def test_updated_forecasts_do_not_change_when_post_day_twelve_arrivals_change():
    canonical = _canonical()
    changed_future = canonical.copy(deep=True)
    changed_future.loc[changed_future["day"] >= 13, "arrivals"] = 9999

    poisson = simulate_updated_poisson_forecast(
        canonical,
        n_simulations=20,
        seed=456,
    )
    changed_poisson = simulate_updated_poisson_forecast(
        changed_future,
        n_simulations=20,
        seed=456,
    )
    negative_binomial = simulate_updated_negative_binomial_forecast(
        canonical,
        n_simulations=20,
        seed=654,
    )
    changed_negative_binomial = simulate_updated_negative_binomial_forecast(
        changed_future,
        n_simulations=20,
        seed=654,
    )

    pd.testing.assert_frame_equal(poisson, changed_poisson)
    pd.testing.assert_frame_equal(negative_binomial, changed_negative_binomial)


def test_latent_columns_do_not_affect_updated_forecasts():
    canonical = _canonical()
    public_only = canonical[["day", "arrivals"]].copy()

    level = estimate_recent_arrival_level(public_only)
    latent_level = estimate_recent_arrival_level(canonical)
    poisson = simulate_updated_poisson_forecast(
        public_only,
        n_simulations=20,
        seed=789,
    )
    latent_poisson = simulate_updated_poisson_forecast(
        canonical,
        n_simulations=20,
        seed=789,
    )
    negative_binomial = simulate_updated_negative_binomial_forecast(
        public_only,
        n_simulations=20,
        seed=987,
    )
    latent_negative_binomial = simulate_updated_negative_binomial_forecast(
        canonical,
        n_simulations=20,
        seed=987,
    )

    pd.testing.assert_frame_equal(level, latent_level)
    pd.testing.assert_frame_equal(poisson, latent_poisson)
    pd.testing.assert_frame_equal(negative_binomial, latent_negative_binomial)


def test_summarize_forecast_totals_by_model_returns_comparable_percentiles():
    poisson = simulate_updated_poisson_forecast(
        _canonical(),
        n_simulations=10,
        seed=123,
    )
    negative_binomial = simulate_updated_negative_binomial_forecast(
        _canonical(),
        n_simulations=10,
        seed=456,
    )
    simulations = pd.concat([poisson, negative_binomial], ignore_index=True)

    summary = summarize_forecast_totals_by_model(
        simulations,
        percentiles=(50, 80, 90, 95),
    )

    assert set(summary["model"]) == {
        "poisson_updated_level",
        "negative_binomial_updated_level",
    }
    for model in summary["model"].unique():
        model_summary = summary.loc[summary["model"] == model, "statistic"].tolist()
        assert model_summary == ["mean", "std", "min", "max", "p50", "p80", "p90", "p95"]


def test_cautious_arrival_level_uses_variance_floor_for_later_windows():
    days_13_18 = estimate_cautious_arrival_level(
        _canonical(),
        observed_start_day=13,
        observed_end_day=18,
    ).iloc[0]
    days_19_23 = estimate_cautious_arrival_level(
        _canonical(),
        observed_start_day=19,
        observed_end_day=23,
    ).iloc[0]

    assert days_13_18["updated_daily_arrival_level"] == pytest.approx(
        174.33333333333334
    )
    assert days_13_18["recent_sample_sd"] == pytest.approx(34.662179730942485)
    assert days_13_18["sd_floor"] == pytest.approx(43.583333333333336)
    assert days_13_18["assumed_daily_sd"] == pytest.approx(43.583333333333336)
    assert days_13_18["assumed_daily_variance"] == pytest.approx(
        1899.5069444444446
    )

    assert days_19_23["updated_daily_arrival_level"] == pytest.approx(136.0)
    assert days_19_23["recent_sample_sd"] == pytest.approx(19.40360791247031)
    assert days_19_23["sd_floor"] == pytest.approx(34.0)
    assert days_19_23["assumed_daily_sd"] == pytest.approx(34.0)
    assert days_19_23["assumed_daily_variance"] == pytest.approx(1156.0)


def test_cautious_negative_binomial_parameters_match_variance_floor_rule():
    simulations = simulate_cautious_negative_binomial_forecast(
        _canonical(),
        observed_start_day=13,
        observed_end_day=18,
        forecast_start_day=19,
        forecast_end_day=23,
        n_simulations=2,
        seed=321,
    )
    mu = 174.33333333333334
    variance = 1899.5069444444446

    assert simulations["day"].tolist() == list(range(19, 24)) * 2
    assert simulations["model"].unique().tolist() == [
        "cautious_negative_binomial_updated_level"
    ]
    assert simulations["assumed_daily_mean"].unique().tolist() == pytest.approx([mu])
    assert simulations["assumed_daily_variance"].unique().tolist() == pytest.approx(
        [variance]
    )
    assert simulations["negative_binomial_p"].unique().tolist() == pytest.approx(
        [mu / variance]
    )
    assert simulations["negative_binomial_n"].unique().tolist() == pytest.approx(
        [mu**2 / (variance - mu)]
    )
    assert simulations["variance_rule"].unique().tolist() == [
        "max_recent_sample_sd_or_fraction_of_mean"
    ]


def test_cautious_negative_binomial_reproducibility_and_validation():
    canonical = _canonical()

    first = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        seed=101,
    )
    second = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        seed=101,
    )
    rng_first = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        rng=np.random.default_rng(202),
    )
    rng_second = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        rng=np.random.default_rng(202),
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(rng_first, rng_second)

    with pytest.raises(ValueError, match="either seed or rng"):
        simulate_cautious_negative_binomial_forecast(
            canonical,
            observed_start_day=19,
            observed_end_day=23,
            forecast_start_day=24,
            forecast_end_day=30,
            seed=1,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="n_simulations must be positive"):
        simulate_cautious_negative_binomial_forecast(
            canonical,
            observed_start_day=19,
            observed_end_day=23,
            forecast_start_day=24,
            forecast_end_day=30,
            n_simulations=0,
        )
    with pytest.raises(ValueError, match="sd_floor_fraction must be non-negative"):
        estimate_cautious_arrival_level(
            canonical,
            observed_start_day=19,
            observed_end_day=23,
            sd_floor_fraction=-0.1,
        )


def test_cautious_forecasts_do_not_peek_at_future_or_latent_columns():
    canonical = _canonical()
    changed_after_day_18 = canonical.copy(deep=True)
    changed_after_day_18.loc[changed_after_day_18["day"] >= 19, "arrivals"] = 9999
    changed_after_day_23 = canonical.copy(deep=True)
    changed_after_day_23.loc[changed_after_day_23["day"] >= 24, "arrivals"] = 9999

    forecast_19_23 = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=13,
        observed_end_day=18,
        forecast_start_day=19,
        forecast_end_day=23,
        n_simulations=20,
        seed=456,
    )
    changed_forecast_19_23 = simulate_cautious_negative_binomial_forecast(
        changed_after_day_18,
        observed_start_day=13,
        observed_end_day=18,
        forecast_start_day=19,
        forecast_end_day=23,
        n_simulations=20,
        seed=456,
    )
    forecast_24_30 = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        seed=654,
    )
    changed_forecast_24_30 = simulate_cautious_negative_binomial_forecast(
        changed_after_day_23,
        observed_start_day=19,
        observed_end_day=23,
        forecast_start_day=24,
        forecast_end_day=30,
        n_simulations=20,
        seed=654,
    )

    pd.testing.assert_frame_equal(forecast_19_23, changed_forecast_19_23)
    pd.testing.assert_frame_equal(forecast_24_30, changed_forecast_24_30)

    public_only = canonical[["day", "arrivals"]].copy()
    public_level = estimate_cautious_arrival_level(
        public_only,
        observed_start_day=13,
        observed_end_day=18,
    )
    latent_level = estimate_cautious_arrival_level(
        canonical,
        observed_start_day=13,
        observed_end_day=18,
    )
    pd.testing.assert_frame_equal(public_level, latent_level)


def test_preparedness_summary_hands_off_cleanly_to_translation():
    simulations = pd.DataFrame(
        {
            "model": ["example_model"] * 9,
            "simulation_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "day": [1, 2, 3] * 3,
            "simulated_arrivals": [10, 20, 30, 20, 40, 60, 30, 60, 90],
        }
    )

    summary = summarize_preparedness_forecast_totals(simulations)
    preparedness = translate_arrival_totals_to_preparedness(
        summary,
        forecast_horizon_days=3,
    )

    assert summary["forecast_reference"].tolist() == ["p50", "p80", "p95"]
    assert summary["forecast_total_arrivals"].tolist() == pytest.approx(
        [120.0, 156.0, 174.0]
    )
    assert preparedness["forecast_reference"].tolist() == ["p50", "p80", "p95"]
    assert preparedness["forecast_total_arrivals"].tolist() == pytest.approx(
        [120.0, 156.0, 174.0]
    )
    assert preparedness["average_daily_arrivals"].tolist() == pytest.approx(
        [40.0, 52.0, 58.0]
    )
    assert preparedness["immediate_water_litres"].tolist() == pytest.approx(
        [360.0, 468.0, 522.0]
    )
    assert preparedness["immediate_food_units"].tolist() == pytest.approx(
        [120.0, 156.0, 174.0]
    )
    assert preparedness["protection_contacts"].tolist() == pytest.approx(
        [120.0, 156.0, 174.0]
    )
    assert preparedness["protection_staff_days"].tolist() == [2, 3, 3]
    assert preparedness["registration_workload"].tolist() == pytest.approx(
        [120.0, 156.0, 174.0]
    )
    assert preparedness["medical_teams_per_day"].tolist() == [1, 1, 1]
    assert preparedness["medical_team_days"].tolist() == [3, 3, 3]


def test_preparedness_translation_uses_custom_assumptions():
    summary = pd.DataFrame(
        {
            "model": ["example_model"],
            "forecast_reference": ["p95"],
            "forecast_total_arrivals": [600.0],
        }
    )

    preparedness = translate_arrival_totals_to_preparedness(
        summary,
        forecast_horizon_days=3,
        water_litres_per_arrival=4,
        food_units_per_arrival=1.5,
        protection_contacts_per_arrival=0.5,
        protection_contacts_per_staff_day=100,
        registrations_per_arrival=0.75,
        medical_team_arrivals_per_day=150,
        minimum_medical_teams=2,
    ).iloc[0]

    assert preparedness["average_daily_arrivals"] == pytest.approx(200.0)
    assert preparedness["immediate_water_litres"] == pytest.approx(2400.0)
    assert preparedness["immediate_food_units"] == pytest.approx(900.0)
    assert preparedness["protection_contacts"] == pytest.approx(300.0)
    assert preparedness["protection_staff_days"] == 3
    assert preparedness["registration_workload"] == pytest.approx(450.0)
    assert preparedness["medical_teams_per_day"] == 2
    assert preparedness["medical_team_days"] == 6


def test_preparedness_translation_validates_inputs():
    valid_summary = pd.DataFrame(
        {
            "model": ["example_model"],
            "forecast_reference": ["p50"],
            "forecast_total_arrivals": [500.0],
        }
    )

    with pytest.raises(ValueError, match="forecast_horizon_days must be positive"):
        translate_arrival_totals_to_preparedness(
            valid_summary,
            forecast_horizon_days=0,
        )
    with pytest.raises(ValueError, match="missing required columns"):
        translate_arrival_totals_to_preparedness(
            pd.DataFrame({"forecast_reference": ["p50"]}),
            forecast_horizon_days=5,
        )
    with pytest.raises(ValueError, match="non-negative"):
        translate_arrival_totals_to_preparedness(
            pd.DataFrame(
                {
                    "model": ["example_model"],
                    "forecast_reference": ["p50"],
                    "forecast_total_arrivals": [-1.0],
                }
            ),
            forecast_horizon_days=5,
        )
    for parameter_name in (
        "water_litres_per_arrival",
        "food_units_per_arrival",
        "protection_contacts_per_arrival",
        "registrations_per_arrival",
    ):
        with pytest.raises(ValueError, match=f"{parameter_name} must be non-negative"):
            translate_arrival_totals_to_preparedness(
                valid_summary,
                forecast_horizon_days=5,
                **{parameter_name: -0.1},
            )
    for parameter_name in (
        "protection_contacts_per_staff_day",
        "medical_team_arrivals_per_day",
        "minimum_medical_teams",
    ):
        with pytest.raises(ValueError, match=f"{parameter_name} must be positive"):
            translate_arrival_totals_to_preparedness(
                valid_summary,
                forecast_horizon_days=5,
                **{parameter_name: 0},
            )


def test_forecast_helpers_do_not_mutate_input_data():
    canonical = _canonical()
    original = canonical.copy(deep=True)

    mean_baseline_forecast(canonical)
    simulations = simulate_poisson_forecast(canonical, n_simulations=10, seed=303)
    summarize_poisson_totals(simulations)
    estimate_recent_arrival_level(canonical)
    updated_poisson = simulate_updated_poisson_forecast(
        canonical,
        n_simulations=10,
        seed=404,
    )
    updated_negative_binomial = simulate_updated_negative_binomial_forecast(
        canonical,
        n_simulations=10,
        seed=505,
    )
    cautious_negative_binomial = simulate_cautious_negative_binomial_forecast(
        canonical,
        observed_start_day=13,
        observed_end_day=18,
        forecast_start_day=19,
        forecast_end_day=23,
        n_simulations=10,
        seed=606,
    )
    summarize_forecast_totals_by_model(
        pd.concat([updated_poisson, updated_negative_binomial], ignore_index=True)
    )
    preparedness_summary = summarize_preparedness_forecast_totals(
        cautious_negative_binomial
    )
    translate_arrival_totals_to_preparedness(
        preparedness_summary,
        forecast_horizon_days=5,
    )

    pd.testing.assert_frame_equal(canonical, original)
