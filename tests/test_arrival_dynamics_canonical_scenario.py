from pathlib import Path
import hashlib
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulation.arrival_dynamics import PhaseConfig, generate_candidate  # noqa: E402


CANONICAL_CSV_PATH = (
    PROJECT_ROOT / "data" / "processed" / "arrival_dynamics_short_term_master.csv"
)
CANONICAL_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "arrival_dynamics_short_term_master_metadata.json"
)

EXPECTED_COLUMNS = [
    "day",
    "arrivals",
    "phase",
    "latent_expected_arrivals",
    "latent_variability_state",
]

EXPECTED_ARRIVALS = [
    25,
    40,
    44,
    48,
    69,
    96,
    106,
    101,
    83,
    136,
    188,
    148,
    151,
    234,
    167,
    193,
    164,
    137,
    163,
    126,
    121,
    150,
    120,
    121,
    123,
    121,
    114,
    96,
    111,
    92,
]


def _read_metadata() -> dict:
    return json.loads(CANONICAL_METADATA_PATH.read_text(encoding="utf-8"))


def test_canonical_scenario_files_exist():
    assert CANONICAL_CSV_PATH.exists()
    assert CANONICAL_METADATA_PATH.exists()


def test_canonical_csv_has_expected_schema_days_and_arrivals():
    canonical = pd.read_csv(CANONICAL_CSV_PATH)

    assert list(canonical.columns) == EXPECTED_COLUMNS
    assert len(canonical) == 30
    assert canonical["day"].tolist() == list(range(1, 31))
    assert canonical["arrivals"].tolist() == EXPECTED_ARRIVALS


def test_canonical_metadata_has_reproducibility_fields():
    metadata = _read_metadata()
    phase_config = metadata["generator_parameters"]["phase_config"]

    assert metadata["selected_seed"] == 3660
    assert metadata["generator_mechanism"]
    assert metadata["generator_source_commit"]
    assert len(phase_config) == 5
    assert metadata["csv_sha256"]


def test_canonical_csv_sha256_matches_metadata():
    metadata = _read_metadata()
    csv_bytes = CANONICAL_CSV_PATH.read_bytes().replace(b"\r\n", b"\n")
    csv_digest = hashlib.sha256(csv_bytes).hexdigest()

    assert csv_digest == metadata["csv_sha256"]


def test_metadata_phase_config_reconstructs_canonical_csv():
    metadata = _read_metadata()
    phase_config = [
        PhaseConfig(**phase)
        for phase in metadata["generator_parameters"]["phase_config"]
    ]
    generated = generate_candidate(
        seed=metadata["selected_seed"],
        phases=phase_config,
    )
    canonical = pd.read_csv(CANONICAL_CSV_PATH)

    exact_columns = [
        "day",
        "arrivals",
        "phase",
        "latent_variability_state",
    ]
    pd.testing.assert_frame_equal(
        generated[exact_columns],
        canonical[exact_columns],
    )
    pd.testing.assert_series_equal(
        generated["latent_expected_arrivals"],
        canonical["latent_expected_arrivals"],
        check_exact=False,
        check_names=False,
    )
