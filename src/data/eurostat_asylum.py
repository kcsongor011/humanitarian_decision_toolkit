"""Eurostat asylum data cleaning helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = ["date", "applications"]
START_DATE = pd.Timestamp("2008-01-01")
END_DATE = pd.Timestamp("2026-02-01")


def clean_italy_first_time_asylum_applications(
    raw_excel_path: str | Path,
    output_csv_path: str | Path,
) -> pd.DataFrame:
    """Clean Italy monthly first-time asylum applications from Eurostat Excel."""

    raw_excel_path = Path(raw_excel_path)
    output_csv_path = Path(output_csv_path)
    if not raw_excel_path.exists():
        raise FileNotFoundError(
            f"Raw Eurostat Excel file not found: {raw_excel_path}. "
            "Place it at data/raw/eu_fta.xlsx."
        )

    raw = pd.read_excel(raw_excel_path, sheet_name="Data")
    country_column = raw.columns[0]
    italy_rows = raw.loc[raw[country_column].astype(str).str.strip().eq("Italy")]
    if len(italy_rows) != 1:
        raise ValueError("Expected exactly one Italy row in Eurostat Data sheet.")

    italy = italy_rows.iloc[0]
    rows = []
    for column in raw.columns[1:]:
        month = _parse_month_column(column)
        if month is None:
            continue
        rows.append({"date": month, "applications": italy[column]})

    cleaned = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    cleaned["applications"] = (
        cleaned["applications"].replace(":", pd.NA).pipe(pd.to_numeric, errors="coerce")
    )
    cleaned = cleaned.loc[
        cleaned["date"].between(START_DATE, END_DATE),
        OUTPUT_COLUMNS,
    ].sort_values("date")

    _validate_cleaned_monthly_series(cleaned)
    cleaned["applications"] = cleaned["applications"].astype("int64")
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_csv_path, index=False)
    return cleaned.reset_index(drop=True)


def _parse_month_column(column: object) -> pd.Timestamp | None:
    if not isinstance(column, str):
        return None
    parsed = pd.to_datetime(column.strip(), format="%Y-%m", errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).replace(day=1)


def _validate_cleaned_monthly_series(cleaned: pd.DataFrame) -> None:
    expected_dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    if cleaned.empty:
        raise ValueError("Cleaned Italy asylum application series is empty.")
    if cleaned["date"].duplicated().any():
        raise ValueError("Cleaned Italy asylum application series has duplicate months.")
    if not cleaned["date"].reset_index(drop=True).equals(pd.Series(expected_dates)):
        raise ValueError("Cleaned Italy asylum application series has missing months.")
    if cleaned["applications"].isna().any():
        raise ValueError("Cleaned Italy asylum application series has missing values.")
    if (cleaned["applications"] < 0).any():
        raise ValueError("Applications must be non-negative.")
    if not (cleaned["applications"] % 1 == 0).all():
        raise ValueError("Applications must be integer-like counts.")
