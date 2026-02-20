"""
IPHEX (Integrated Precipitation and Hydrology Experiment) campaign data parser.

Campaign: IPHEX
Data Format: Whitespace-delimited text files with .iphex extension.
  - Column header line starts with 'Time' and contains 'Air_Temp'.
  - One units line follows the header; data starts on the line after that.
  - Filename encodes start timestamp: YYYY_MM_DD_HH_MM_SS.iphex
  - 'Time' column is seconds since midnight UTC on the flight date.

Required variables
------------------
- FrostPoint: chilled mirror hygrometer measurement (°C)
- Air_Temp: ambient air temperature (°C)
- STATIC_PR: static pressure (hPa)

Si derivation
-------------
Uses the notebook-validated Tetens formula:
    e  = 6.112 * exp(22.46 * Tf / (272.62 + Tf))
    ei = 6.112 * exp(22.46 * Ta / (272.62 + Ta))
    Si = e / ei - 1
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


IPHEX_INVALID_VALUES = {
    999999.9999,
    999.9999999,
    9999.999999,
    99999.99999,
    99999999999,
    9.9999e30,
    9.999e30,
    -9999,
    -9999.99,
    -7777,
    -7777.77,
    -8888,
    -8888.88,
}


IPHEX_REFERENCE_STATS = {
    "min": -1.0000,
    "q1": -0.3599,
    "median": -0.1299,
    "q3": 0.0168,
    "max": 1.6811,
    "mean": -0.1615,
    "std": 0.3688,
}


def _es_ice_tetens(temp_c: pd.Series) -> pd.Series:
    t = pd.to_numeric(temp_c, errors="coerce")
    return 6.112 * np.exp((22.46 * t) / (272.62 + t))


def _compute_si_from_frostpoint(frost_point_c: pd.Series, air_temp_c: pd.Series) -> pd.Series:
    e = _es_ice_tetens(frost_point_c)
    ei = _es_ice_tetens(air_temp_c)
    si = (e / ei) - 1.0
    si[~np.isfinite(si)] = np.nan
    return si


def _find_data_start(lines: list[str]) -> tuple[Optional[int], Optional[int]]:
    """
    Locate the column-header line and the first data line.

    The column-header line is the first line that starts with 'Time' and
    also contains 'Air_Temp'.  One units line follows it, then data.

    Returns (header_line_idx, data_start_line_idx).
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Time") and "Air_Temp" in stripped:
            return i, i + 2  # units line at i+1, data at i+2
    return None, None


def _extract_date_from_filename(filepath: Path) -> Optional[datetime]:
    """Extract base date from filename YYYY_MM_DD_HH_MM_SS.iphex."""
    parts = filepath.stem.split("_")
    if len(parts) >= 3:
        try:
            return datetime(
                int(parts[0]), int(parts[1]), int(parts[2])
            )
        except (ValueError, IndexError):
            pass
    return None


def _distribution_stats(si: pd.Series) -> Optional[dict]:
    fp = pd.to_numeric(si, errors="coerce").dropna()
    if fp.empty:
        return None
    return {
        "min": float(np.min(fp)),
        "q1": float(np.percentile(fp, 25)),
        "median": float(np.percentile(fp, 50)),
        "q3": float(np.percentile(fp, 75)),
        "max": float(np.max(fp)),
        "mean": float(np.mean(fp)),
        "std": float(np.std(fp)),
    }


def _print_distribution_check(si: pd.Series) -> None:
    stats = _distribution_stats(si)
    if stats is None:
        print("  Distribution check: no valid Si values")
        return

    print("  IPHEX Si statistics:")
    print(f"    Minimum: {stats['min']:.4f}")
    print(f"    25th Percentile (Q1): {stats['q1']:.4f}")
    print(f"    Median (Q2): {stats['median']:.4f}")
    print(f"    75th Percentile (Q3): {stats['q3']:.4f}")
    print(f"    Maximum: {stats['max']:.4f}")
    print(f"    Mean: {stats['mean']:.4f}")
    print(f"    Standard Deviation: {stats['std']:.4f}")

    delta = {
        k: abs(stats[k] - IPHEX_REFERENCE_STATS[k]) for k in IPHEX_REFERENCE_STATS
    }
    aligned = (
        delta["q1"] <= 0.20
        and delta["median"] <= 0.20
        and delta["q3"] <= 0.20
        and delta["mean"] <= 0.20
        and delta["std"] <= 0.20
    )
    if aligned:
        print("  Distribution check: PASS (aligned with notebook reference shape)")
    else:
        print("  Distribution check: WARN (deviates from notebook reference stats)")


def _coerce_and_mask(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.mask(s.isin(IPHEX_INVALID_VALUES), np.nan)
    return s


def load_iphex_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a single IPHEX .iphex file and derive Si from FrostPoint."""
    filepath = Path(filepath)

    with open(filepath, "r") as f:
        lines = f.readlines()

    header_idx, data_start = _find_data_start(lines)
    if header_idx is None:
        raise ValueError(
            f"Could not find column header ('Time ... Air_Temp ...') in {filepath.name}"
        )

    columns = lines[header_idx].strip().split()

    na_values = [
        "999999.9999", "999.9999999", "9999.999999", "99999.99999",
        "99999999999", "9.999E+30", "9.9999E+30",
    ]

    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=data_start,
        names=columns,
        na_values=na_values,
        engine="python",
        on_bad_lines="skip",
    )

    if df.empty:
        return df

    for required in ("FrostPoint", "Air_Temp", "STATIC_PR"):
        if required not in df.columns:
            raise KeyError(
                f"Missing required IPHEX column '{required}' in {filepath.name}. "
                f"Available: {list(df.columns)}"
            )

    df["FrostPoint"] = _coerce_and_mask(df["FrostPoint"])
    df["Air_Temp"] = _coerce_and_mask(df["Air_Temp"])
    df["STATIC_PR"] = _coerce_and_mask(df["STATIC_PR"])

    # Unit guards: convert Kelvin → Celsius if median looks Kelvin-range
    med_t = np.nanmedian(df["Air_Temp"].to_numpy(dtype=float))
    if np.isfinite(med_t) and med_t > 150:
        df["Air_Temp"] = df["Air_Temp"] - 273.15
    med_fp = np.nanmedian(df["FrostPoint"].to_numpy(dtype=float))
    if np.isfinite(med_fp) and med_fp > 150:
        df["FrostPoint"] = df["FrostPoint"] - 273.15

    # Convert Pa → hPa if pressure looks like Pascals
    med_p = np.nanmedian(df["STATIC_PR"].to_numpy(dtype=float))
    if np.isfinite(med_p) and 2000 < med_p < 120000:
        df["STATIC_PR"] = df["STATIC_PR"] / 100.0

    # Physical bounds
    df.loc[(df["Air_Temp"] < -95) | (df["Air_Temp"] > 60), "Air_Temp"] = np.nan
    df.loc[(df["FrostPoint"] < -120) | (df["FrostPoint"] > 40), "FrostPoint"] = np.nan
    df.loc[(df["STATIC_PR"] < 50) | (df["STATIC_PR"] > 1100), "STATIC_PR"] = np.nan
    df.loc[df["FrostPoint"] > (df["Air_Temp"] + 20), "FrostPoint"] = np.nan

    # Si from frost point
    df["Si"] = _compute_si_from_frostpoint(df["FrostPoint"], df["Air_Temp"])
    df.loc[(df["Si"] < -1.0) | (df["Si"] > 5.0), "Si"] = np.nan

    # Timestamps: Time column is seconds-since-midnight on the flight date
    base_date = _extract_date_from_filename(filepath)
    if "Time" in df.columns and base_date is not None:
        time_s = pd.to_numeric(df["Time"], errors="coerce")
        df["Timestamp"] = pd.Timestamp(base_date) + pd.to_timedelta(time_s, unit="s")
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    else:
        df["Timestamp"] = pd.NaT

    # Position columns (IPHEX uses POS_Lat, POS_Lon, POS_Alt)
    for col, lo, hi in [
        ("POS_Lat", -90.0, 90.0),
        ("POS_Lon", -180.0, 180.0),
        ("POS_Alt", -500.0, 25000.0),
    ]:
        if col in df.columns:
            df[col] = _coerce_and_mask(df[col])
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df["source_file"] = filepath.name
    df["Campaign"] = "IPHEX"

    _print_distribution_check(df["Si"])

    return df


def load_iphex(data_dir: Union[str, Path], pattern: str = "*.iphex") -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file() and "Combined" not in f.name]

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_iphex_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if not dfs:
        raise ValueError(f"No valid IPHEX files were parsed in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "IPHEX"
    return combined


def extract_iphex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Return standardized columns for combined campaign output."""
    return pd.DataFrame(
        {
            "Timestamp": df.get("Timestamp", pd.NaT),
            "Tair_C": df.get("Air_Temp", np.nan),
            "Si": df.get("Si", np.nan),
            "Lat": df.get("POS_Lat", np.nan),
            "Lon": df.get("POS_Lon", np.nan),
            "Alt_m": df.get("POS_Alt", np.nan),
            "Campaign": df.get("Campaign", "IPHEX"),
            "source_file": df.get("source_file", ""),
        }
    )
