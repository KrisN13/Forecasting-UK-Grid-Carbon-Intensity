"""
pipeline.py
-----------
Automated data ingestion for the Carbon Intensity forecasting project.

Data sources
~~~~~~~~~~~~
- Carbon Intensity API  (api.carbonintensity.org.uk) — free, no auth
- Open-Meteo Archive API (archive-api.open-meteo.com)  — free, no auth

Public API
~~~~~~~~~~
    update_carbon_parquet(path)      — fetch new carbon + mix records, append to parquet
    update_temperature_parquet(path) — fetch new temperature records, append to parquet
    update_all()                     — run both updates in one call

    fetch_carbon_intensity(start, end) -> pd.DataFrame   (raw, 30-min)
    fetch_generation_mix(start, end)   -> pd.DataFrame   (raw, 30-min)
    fetch_temperature(start, end)      -> pd.DataFrame   (hourly)

Usage
~~~~~
    # First run — backfills from PIPELINE_START to today
    from carbon.pipeline import update_all
    update_all()

    # Subsequent runs — appends only new records
    update_all()
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from .config import (
    CARBON_PARQUET,
    TEMP_PARQUET,
)

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------

PIPELINE_START = "2022-01-01"   # earliest date to backfill to

CI_API_BASE  = "https://api.carbonintensity.org.uk"
TEMP_API_BASE = "https://archive-api.open-meteo.com/v1/archive"

# UK centroid (used for temperature — representative national average)
TEMP_LAT = 52.5
TEMP_LON  = -1.5

# API chunk size — Carbon Intensity API is stable up to 30 days per request
CHUNK_DAYS = 14

# Fuel → column name mapping (Carbon Intensity API uses lowercase)
FUEL_MAP = {
    "gas":     "GAS",
    "coal":    "COAL",
    "nuclear": "NUCLEAR",
    "wind":    "WIND",
    "hydro":   "HYDRO",
    "solar":   "SOLAR",
    "biomass": "BIOMASS",
    "imports": "OTHER",   # imports treated as OTHER to match existing schema
    "other":   "OTHER",
    "storage": "STORAGE",
}

# Fossil fuels (for deriving FOSSIL column)
FOSSIL_FUELS = {"GAS", "COAL"}

# Low-carbon fuels
LOW_CARBON_FUELS = {"NUCLEAR", "WIND", "HYDRO", "SOLAR", "BIOMASS", "STORAGE"}

# Zero-carbon fuels
ZERO_CARBON_FUELS = {"WIND", "HYDRO", "SOLAR"}

# Renewable fuels (for RENEWABLE column)
RENEWABLE_FUELS = {"WIND", "HYDRO", "SOLAR", "BIOMASS"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_chunks(
    start: str | pd.Timestamp,
    end:   str | pd.Timestamp,
    chunk_days: int = CHUNK_DAYS,
) -> list[tuple[str, str]]:
    """Split a date range into chunks for API requests."""
    start = pd.Timestamp(start, tz="UTC")
    end   = pd.Timestamp(end,   tz="UTC")
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days), end)
        chunks.append((
            current.strftime("%Y-%m-%dT%H:%MZ"),
            chunk_end.strftime("%Y-%m-%dT%H:%MZ"),
        ))
        current = chunk_end
    return chunks


def _get_last_timestamp(path: str | Path) -> pd.Timestamp | None:
    """Return the last datetime index value in a parquet file, or None."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df.index.max()


# ---------------------------------------------------------------------------
# Carbon Intensity API — intensity endpoint
# ---------------------------------------------------------------------------

def fetch_carbon_intensity(
    start: str,
    end:   str,
) -> pd.DataFrame:
    """
    Fetch actual carbon intensity (gCO₂/kWh) for a date range.

    Returns 30-minute resolution DataFrame with columns:
        DATETIME (index, UTC), CARBON_INTENSITY
    """
    chunks = _date_chunks(start, end)
    records = []

    for from_ts, to_ts in chunks:
        url = f"{CI_API_BASE}/intensity/{from_ts}/{to_ts}"
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()

        for entry in resp.json().get("data", []):
            actual = entry.get("intensity", {}).get("actual")
            if actual is not None:
                records.append({
                    "DATETIME":         entry["from"],
                    "CARBON_INTENSITY": float(actual),
                })
        time.sleep(0.3)   # be polite to the API

    if not records:
        return pd.DataFrame(columns=["CARBON_INTENSITY"])

    df = pd.DataFrame(records)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
    df = df.set_index("DATETIME").sort_index()
    return df


# ---------------------------------------------------------------------------
# Carbon Intensity API — generation mix endpoint
# ---------------------------------------------------------------------------

def fetch_generation_mix(
    start: str,
    end:   str,
) -> pd.DataFrame:
    """
    Fetch generation mix percentages for a date range.

    Returns 30-minute resolution DataFrame with fuel columns mapped to
    the project schema (GAS, COAL, NUCLEAR, WIND, HYDRO, SOLAR, BIOMASS,
    STORAGE, OTHER, FOSSIL, RENEWABLE, LOW_CARBON, ZERO_CARBON, GENERATION).
    """
    chunks = _date_chunks(start, end)
    records = []

    for from_ts, to_ts in chunks:
        url = f"{CI_API_BASE}/generation/{from_ts}/{to_ts}"
        resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
        resp.raise_for_status()

        for entry in resp.json().get("data", []):
            row = {"DATETIME": entry["from"]}
            for item in entry.get("generationmix", []):
                fuel = item["fuel"].lower()
                col  = FUEL_MAP.get(fuel)
                if col:
                    row[col] = row.get(col, 0.0) + float(item["perc"])
            records.append(row)
        time.sleep(0.3)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).fillna(0.0)
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], utc=True)
    df = df.set_index("DATETIME").sort_index()

    # Ensure all mix columns exist
    for col in ["GAS", "COAL", "NUCLEAR", "WIND", "HYDRO",
                "SOLAR", "BIOMASS", "STORAGE", "OTHER"]:
        if col not in df.columns:
            df[col] = 0.0

    # Derived columns
    df["FOSSIL"]     = df[list(FOSSIL_FUELS & set(df.columns))].sum(axis=1)
    df["RENEWABLE"]  = df[list(RENEWABLE_FUELS & set(df.columns))].sum(axis=1)
    df["LOW_CARBON"] = df[list(LOW_CARBON_FUELS & set(df.columns))].sum(axis=1)
    df["ZERO_CARBON"]= df[list(ZERO_CARBON_FUELS & set(df.columns))].sum(axis=1)
    df["GENERATION"] = df[["GAS", "COAL", "NUCLEAR", "WIND", "HYDRO",
                            "SOLAR", "BIOMASS", "STORAGE", "OTHER"]].sum(axis=1)

    return df


# ---------------------------------------------------------------------------
# Combine + resample to hourly
# ---------------------------------------------------------------------------

def _build_carbon_dataframe(
    df_intensity: pd.DataFrame,
    df_mix:       pd.DataFrame,
) -> pd.DataFrame:
    """
    Join intensity and generation mix, then resample from 30-min to hourly.
    """
    df = df_intensity.join(df_mix, how="inner")
    df = df.resample("h").mean()
    df = df.dropna(subset=["CARBON_INTENSITY"])
    return df


# ---------------------------------------------------------------------------
# Open-Meteo temperature
# ---------------------------------------------------------------------------

def fetch_temperature(
    start: str,
    end:   str,
    lat:   float = TEMP_LAT,
    lon:   float = TEMP_LON,
) -> pd.DataFrame:
    """
    Fetch hourly 2-metre temperature (°C) for the UK from Open-Meteo.

    Parameters
    ----------
    start, end : Date strings ``'YYYY-MM-DD'``.
    lat, lon   : Coordinates (default: UK centroid).

    Returns
    -------
    pd.DataFrame with column ``temp_2m``, hourly UTC index.
    """
    # Open-Meteo requires date-only strings
    start_date = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_date   = pd.Timestamp(end).strftime("%Y-%m-%d")

    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          "temperature_2m",
        "start_date":      start_date,
        "end_date":        end_date,
        "timezone":        "UTC",
    }

    resp = requests.get(TEMP_API_BASE, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame({
        "DATETIME": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temp_2m":  data["hourly"]["temperature_2m"],
    })
    df = df.set_index("DATETIME").sort_index()
    return df


# ---------------------------------------------------------------------------
# Update parquet files (append-only)
# ---------------------------------------------------------------------------

def update_carbon_parquet(
    path: str | Path = CARBON_PARQUET,
    start: str = PIPELINE_START,
) -> pd.DataFrame:
    """
    Fetch new carbon intensity + generation mix records and append to parquet.

    On first run (no parquet exists): backfills from ``start`` to today.
    On subsequent runs: fetches only records after the last saved timestamp.

    Returns the updated DataFrame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_ts = _get_last_timestamp(path)

    if last_ts is None:
        fetch_start = start
        print(f"No existing data — backfilling from {start}")
    else:
        fetch_start = (last_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%MZ")
        print(f"Last record: {last_ts}  — fetching new records from {fetch_start}")

    fetch_end = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    if pd.Timestamp(fetch_start, tz="UTC") >= pd.Timestamp(fetch_end, tz="UTC"):
        print("Already up to date.")
        return pd.read_parquet(path)

    print("Fetching carbon intensity...")
    df_intensity = fetch_carbon_intensity(fetch_start, fetch_end)

    print("Fetching generation mix...")
    df_mix = fetch_generation_mix(fetch_start, fetch_end)

    if df_intensity.empty or df_mix.empty:
        print("No new data returned from API.")
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    df_new = _build_carbon_dataframe(df_intensity, df_mix)
    print(f"  {len(df_new):,} new hourly records")

    if last_ts is not None:
        df_existing = pd.read_parquet(path)
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined = df_combined.sort_index()
    else:
        df_combined = df_new

    df_combined.to_parquet(path)
    print(f"Saved → {path}  ({len(df_combined):,} total rows)")
    return df_combined


def update_temperature_parquet(
    path: str | Path = TEMP_PARQUET,
    start: str = PIPELINE_START,
) -> pd.DataFrame:
    """
    Fetch new UK temperature records from Open-Meteo and append to parquet.

    On first run: backfills from ``start`` to today.
    On subsequent runs: fetches only records after the last saved timestamp.

    Returns the updated DataFrame.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_ts = _get_last_timestamp(path)

    if last_ts is None:
        fetch_start = start
        print(f"No existing temperature data — backfilling from {start}")
    else:
        fetch_start = (last_ts + pd.Timedelta(hours=1)).strftime("%Y-%m-%d")
        print(f"Last temperature record: {last_ts}  — fetching from {fetch_start}")

    fetch_end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if pd.Timestamp(fetch_start) >= pd.Timestamp(fetch_end):
        print("Temperature data already up to date.")
        return pd.read_parquet(path)

    print("Fetching temperature from Open-Meteo...")
    df_new = fetch_temperature(fetch_start, fetch_end)
    print(f"  {len(df_new):,} new hourly temperature records")

    if last_ts is not None:
        df_existing = pd.read_parquet(path)
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined = df_combined.sort_index()
    else:
        df_combined = df_new

    df_combined.to_parquet(path)
    print(f"Saved → {path}  ({len(df_combined):,} total rows)")
    return df_combined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def update_all(start: str = PIPELINE_START) -> None:
    """
    Fetch and append all new carbon intensity, generation mix, and
    temperature records in one call.

    Safe to run repeatedly — only fetches records newer than what's
    already saved.

    Example
    -------
    >>> from carbon.pipeline import update_all
    >>> update_all()
    """
    print("=" * 50)
    print("Carbon Intensity Pipeline")
    print("=" * 50)
    update_carbon_parquet(start=start)
    print()
    update_temperature_parquet(start=start)
    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    update_all()
