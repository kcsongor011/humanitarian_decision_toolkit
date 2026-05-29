from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation.arrival_dynamics import (  # noqa: E402
    default_phase_config,
    expected_arrival_path,
    generate_candidate,
)


REQUIRED_COLUMNS = {
    "day",
    "arrivals",
    "phase",
    "latent_expected_arrivals",
    "latent_variability_state",
}


def test_default_phase_config_covers_days_one_to_thirty_contiguously():
    phases = default_phase_config()

    covered_days = []
    for phase in phases:
        covered_days.extend(range(phase.start_day, phase.end_day + 1))

    assert len(phases) == 5
    assert covered_days == list(range(1, 31))


def test_expected_arrival_path_has_required_latent_schema():
    path = expected_arrival_path()

    assert len(path) == 30
    assert list(path["day"]) == list(range(1, 31))
    assert {
        "day",
        "phase",
        "latent_expected_arrivals",
        "latent_variability_state",
    }.issubset(path.columns)


def test_expected_arrival_path_rejects_empty_phase_iterable():
    with pytest.raises(ValueError, match="cover days 1 through 30"):
        expected_arrival_path([])


def test_generate_candidate_has_required_schema_and_day_coverage():
    candidate = generate_candidate(seed=42)

    assert REQUIRED_COLUMNS.issubset(candidate.columns)
    assert len(candidate) == 30
    assert list(candidate["day"]) == list(range(1, 31))


def test_generate_candidate_arrivals_are_non_negative_integers():
    candidate = generate_candidate(seed=42)

    assert pd.api.types.is_integer_dtype(candidate["arrivals"])
    assert (candidate["arrivals"] >= 0).all()


def test_phase_labels_appear_on_intended_day_ranges():
    candidate = generate_candidate(seed=42)

    expected_ranges = {
        "early_visible_escalation": range(1, 6),
        "surge_becomes_apparent": range(6, 13),
        "volatile_high_pressure": range(13, 19),
        "updated_planning_period": range(19, 24),
        "partial_stabilisation": range(24, 31),
    }

    for label, days in expected_ranges.items():
        actual_days = candidate.loc[candidate["phase"] == label, "day"].tolist()
        assert actual_days == list(days)


def test_same_seed_produces_identical_candidate():
    first = generate_candidate(seed=123)
    second = generate_candidate(seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_preserve_structure_but_can_change_arrivals():
    first = generate_candidate(seed=123)
    second = generate_candidate(seed=124)

    assert first.drop(columns=["arrivals"]).equals(second.drop(columns=["arrivals"]))
    assert not first["arrivals"].equals(second["arrivals"])


def test_rng_can_control_reproducibility():
    first = generate_candidate(rng=np.random.default_rng(456))
    second = generate_candidate(rng=np.random.default_rng(456))

    pd.testing.assert_frame_equal(first, second)


def test_seed_and_rng_together_raise_value_error():
    with pytest.raises(ValueError, match="either seed or rng"):
        generate_candidate(seed=1, rng=np.random.default_rng(1))
