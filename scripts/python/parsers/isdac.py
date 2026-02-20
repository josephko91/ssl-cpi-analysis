"""
ISDAC (Indirect and Semi-Direct Aerosol Campaign) campaign data parser.

Campaign: ISDAC — 2008 Convair 580, Fairbanks Alaska
Data Format: Comma-delimited .txt files (STRAPP bulk microphysics)
  - Filename pattern: 001F##_HHMMSS.txt
  - Column header line starts with 'HH,MM,SS'
  - Two lines follow the header (units row + dashes separator) before data
  - duplicate HH,MM,SS columns: first set = UTC time, second = DAS time
  - Flight date embedded in file header: "Flight Date: YYYY/Mon/DD"

Key variables
-------------
- RSTem   : Port Static Temperature (°C)                → Tair
- NPres   : Corrected Static Air Pressure (mb = hPa)   → Pressure
- ReHuI   : Relative Humidity w.r.t. Ice (%)           → Si source
- LicFro  : LiCor Frost Point (°C)                     → available for cross-check
- MastrLAT: Master GPS Latitude (° N)
- MasterLON: Master GPS Longitude (° E)
- PreAlt  : Pressure Altitude (m)

Si derivation
-------------
Primary: Si = ReHuI / 100 - 1   (direct from on-board RH_ice sensor)

Invalid-value sentinels
-----------------------
Large negative fill values of the form -8, -88, -888, -8.888, -88.8, etc.
(all digits 8 or 9, possibly with a decimal).  A regex pattern is used to
catch all variants:  ^-?[89]+(.\\d+)?$
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Invalid-value detection
# ---------------------------------------------------------------------------

# Regex: matches strings like -8, -88, -888, -8.8, -8.888, -88.8, -9999, etc.
_ISDAC_FILL_RE = re.compile(r"^-?[89]+(\.\d+)?$")

# Numeric values that are clearly fill (large magnitude negatives common in STRAPP)
_ISDAC_FILL_FLOATS: set[float] = {
    -8.0, -88.0, -888.0, -8888.0,
    -8.8, -88.8, -888.8,
    -8.88, -88.88, -888.88,
    -8.888, -88.888,
    -8.8888,
    -3000.0,   # altitude-specific fill
    -9999.0, -9999.99,
    -7777.0, -7777.77,
    99999.0, 999999.0,
    9.999e30, 9.9999e30,
}


def _is_fill_string(s: str) -> bool:
    """Return True if a string token represents a fill value."""
    return bool(_ISDAC_FILL_RE.match(s.strip()))


def _mask_fills(series: pd.Series) -> pd.Series:
    """Convert to float and replace all known fill values with NaN."""
    s = pd.to_numeric(series, errors="coerce")
    # Mask numeric fill values
    s = s.where(~s.isin(_ISDAC_FILL_FLOATS), other=np.nan)
    return s


# ---------------------------------------------------------------------------
# Header / date parsing helpers
# ---------------------------------------------------------------------------

def _find_header_row(lines: list[str]) -> Optional[int]:
    """
    Return the index of the column-names line.
    It is the first line whose stripped text starts with 'HH,MM,SS'.
    """
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("HH,MM,SS"):
            return i
    return None


def _parse_flight_date(lines: list[str]) -> Optional[datetime]:
    """
    Extract the flight date from a header line like:
        'Flight Date: 2008/Mar/31    Aircraft: CONVAIR 580'
    Returns a UTC midnight datetime if found, else None.
    """
    date_re = re.compile(r"Flight\s+Date\s*:\s*(\d{4}/\w{3}/\d{1,2})", re.IGNORECASE)
    for line in lines:
        m = date_re.search(line)
        if m:
            try:
                return datetime.strptime(m.group(1), "%Y/%b/%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def _make_unique_columns(names: list[str]) -> list[str]:
    """Append _1, _2, … suffixes to duplicate column names."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for name in names:
        name = name.strip() or "unnamed"
        if name in seen:
            seen[name] += 1
            result.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


# ---------------------------------------------------------------------------
# File-level loader
# ---------------------------------------------------------------------------

def load_isdac_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single ISDAC STRAPP comma-delimited .txt file.

    Steps
    -----
    1. Locate the column-names line (`HH,MM,SS ...`).
    2. Parse column names; deduplicate duplicates.
    3. Skip header_row + 3 lines (col names + units + dashes separator).
    4. Parse fill values and apply physical-range filters.
    5. Derive Si from ReHuI.
    6. Build UTC timestamps from HH / MM / SS + flight date.

    Returns a DataFrame with columns including: Timestamp, RSTem, NPres,
    ReHuI, LicFro, Si, MastrLAT, MasterLON, PreAlt, source_file, Campaign.
    """
    filepath = Path(filepath)

    with open(filepath, "r", errors="replace") as fh:
        lines = fh.readlines()

    header_row = _find_header_row(lines)
    if header_row is None:
        raise ValueError(
            f"Could not find column header ('HH,MM,SS ...') in {filepath.name}"
        )

    flight_date = _parse_flight_date(lines[:header_row])

    # Parse and deduplicate column names
    raw_cols = [c.strip() for c in lines[header_row].split(",")]
    columns = _make_unique_columns(raw_cols)

    # data starts at header_row + 3 (skip units line + dashes separator)
    data_start = header_row + 3

    df = pd.read_csv(
        filepath,
        skiprows=data_start,
        names=columns,
        na_values=["", " ", "NA", "NaN"],
        skipinitialspace=True,
        on_bad_lines="skip",
        low_memory=False,
    )

    if df.empty:
        return df

    # -------------------------------------------------------------------
    # Apply fill-value masking to all numeric columns
    # -------------------------------------------------------------------
    for col in df.columns:
        if col in ("source_file", "Campaign"):
            continue
        df[col] = _mask_fills(df[col])

    # -------------------------------------------------------------------
    # Check required columns present
    # -------------------------------------------------------------------
    for required in ("ReHuI", "RSTem", "NPres"):
        if required not in df.columns:
            raise KeyError(
                f"Missing required ISDAC column '{required}' in {filepath.name}. "
                f"Available: {list(df.columns)}"
            )

    # -------------------------------------------------------------------
    # Physical-range filters
    # -------------------------------------------------------------------
    # Temperature: RSTem in °C; check for accidental Kelvin
    med_t = np.nanmedian(df["RSTem"].to_numpy(dtype=float))
    if np.isfinite(med_t) and med_t > 150:
        df["RSTem"] = df["RSTem"] - 273.15
    df.loc[(df["RSTem"] < -95) | (df["RSTem"] > 60), "RSTem"] = np.nan

    # Pressure: NPres in mb (= hPa); check for accidental Pa
    med_p = np.nanmedian(df["NPres"].to_numpy(dtype=float))
    if np.isfinite(med_p) and 2000 < med_p < 120000:
        df["NPres"] = df["NPres"] / 100.0
    df.loc[(df["NPres"] < 50) | (df["NPres"] > 1100), "NPres"] = np.nan

    # Relative humidity w.r.t. ice: expect [0, 200]%
    df.loc[(df["ReHuI"] < -20) | (df["ReHuI"] > 250), "ReHuI"] = np.nan

    # Frost point (if present): expect [-120, 40] °C
    if "LicFro" in df.columns:
        fp_vals = df["LicFro"].to_numpy(dtype=float)
        if np.any(np.isfinite(fp_vals)):
            med_fp = np.nanmedian(fp_vals)
            if np.isfinite(med_fp) and med_fp > 150:
                df["LicFro"] = df["LicFro"] - 273.15
        df.loc[(df["LicFro"] < -120) | (df["LicFro"] > 40), "LicFro"] = np.nan

    # Altitude: PreAlt in m
    if "PreAlt" in df.columns:
        df.loc[(df["PreAlt"] < -500) | (df["PreAlt"] > 25000), "PreAlt"] = np.nan

    # Latitude / Longitude
    if "MastrLAT" in df.columns:
        df.loc[(df["MastrLAT"] < -90) | (df["MastrLAT"] > 90), "MastrLAT"] = np.nan
    if "MasterLON" in df.columns:
        df.loc[(df["MasterLON"] < -180) | (df["MasterLON"] > 180), "MasterLON"] = np.nan

    # -------------------------------------------------------------------
    # Si derivation: Si = ReHuI / 100 - 1
    # -------------------------------------------------------------------
    df["Si"] = df["ReHuI"] / 100.0 - 1.0
    df.loc[~np.isfinite(df["Si"].to_numpy(dtype=float)), "Si"] = np.nan
    # Physical plausibility: allow a slight margin beyond [-1, 1] for genuine
    # supersaturation extremes, but cap extreme outliers
    df.loc[(df["Si"] < -1.0) | (df["Si"] > 5.0), "Si"] = np.nan

    # -------------------------------------------------------------------
    # Build UTC timestamps from HH / MM / SS columns
    # -------------------------------------------------------------------
    time_cols_present = all(c in df.columns for c in ("HH", "MM", "SS"))
    if time_cols_present and flight_date is not None:
        hh = pd.to_numeric(df["HH"], errors="coerce").fillna(0).astype(int)
        mm = pd.to_numeric(df["MM"], errors="coerce").fillna(0).astype(int)
        ss = pd.to_numeric(df["SS"], errors="coerce").fillna(0).astype(int)
        total_seconds = hh * 3600 + mm * 60 + ss
        # flight_date already carries tzinfo=UTC, so pd.Timestamp is tz-aware;
        # use tz_localize only if the result is naive, otherwise it is already UTC.
        base = pd.Timestamp(flight_date)  # tz-aware UTC
        timestamps = base + pd.to_timedelta(total_seconds, unit="s")
        if timestamps.dt.tz is None:
            timestamps = timestamps.dt.tz_localize("UTC")
        df["Timestamp"] = timestamps
    else:
        df["Timestamp"] = pd.NaT

    df["source_file"] = filepath.name
    df["Campaign"] = "ISDAC"

    return df


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------

def load_isdac(data_dir: Union[str, Path], pattern: str = "*.txt") -> pd.DataFrame:
    """
    Load all ISDAC STRAPP files from *data_dir* matching *pattern*.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the .txt data files.
    pattern : str
        Glob pattern for file discovery.  Defaults to ``*.txt``.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with a 'Campaign' column set to 'ISDAC'.
    """
    data_dir = Path(data_dir)
    files = sorted(f for f in data_dir.glob(pattern) if f.is_file())

    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}"
        )

    dfs: list[pd.DataFrame] = []
    for filepath in files:
        # Skip files with 'Combined' in the name (aggregate files)
        if "Combined" in filepath.name:
            continue
        try:
            dfs.append(load_isdac_file(filepath))
        except Exception as exc:
            print(f"Warning: Could not load {filepath.name}: {exc}")

    if not dfs:
        raise ValueError(f"No valid ISDAC files were parsed from {data_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "ISDAC"
    return combined


# ---------------------------------------------------------------------------
# Standardised extractor
# ---------------------------------------------------------------------------

def extract_isdac_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a standardised DataFrame with columns used across all campaigns.

    Columns
    -------
    Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign, source_file
    """
    return pd.DataFrame(
        {
            "Timestamp": df.get("Timestamp", pd.NaT),
            "Tair_C": df.get("RSTem", np.nan),
            "Si": df.get("Si", np.nan),
            "Lat": df.get("MastrLAT", np.nan),
            "Lon": df.get("MasterLON", np.nan),
            "Alt_m": df.get("PreAlt", np.nan),
            "Campaign": df.get("Campaign", "ISDAC"),
            "source_file": df.get("source_file", ""),
        }
    )
