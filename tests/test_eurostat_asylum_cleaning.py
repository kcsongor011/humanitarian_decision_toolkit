from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data.eurostat_asylum import clean_italy_first_time_asylum_applications  # noqa: E402


RAW_EXCEL_PATH = PROJECT_ROOT / "data" / "raw" / "eu_fta.xlsx"


def test_clean_italy_first_time_asylum_applications_outputs_expected_series(tmp_path):
    output_csv_path = tmp_path / "italy_first_time_asylum_monthly.csv"

    cleaned = clean_italy_first_time_asylum_applications(
        RAW_EXCEL_PATH,
        output_csv_path,
    )

    assert list(cleaned.columns) == ["date", "applications"]
    assert cleaned["date"].iloc[0] == pd.Timestamp("2008-01-01")
    assert cleaned["date"].iloc[-1] == pd.Timestamp("2026-02-01")
    assert not cleaned["applications"].isna().any()
    assert cleaned["date"].tolist() == pd.date_range(
        "2008-01-01",
        "2026-02-01",
        freq="MS",
    ).tolist()
    assert pd.api.types.is_integer_dtype(cleaned["applications"])
    assert (cleaned["applications"] >= 0).all()
    assert pd.Timestamp("2026-03-01") not in set(cleaned["date"])
    assert pd.Timestamp("2026-04-01") not in set(cleaned["date"])

    reloaded = pd.read_csv(output_csv_path, parse_dates=["date"])
    assert pd.api.types.is_integer_dtype(reloaded["applications"])
    pd.testing.assert_frame_equal(cleaned, reloaded)
