"""
POSIDON (Profiling of Winter Storms) campaign data parser.

Campaign: POSIDON — 2016 NASA WB-57, Guam / western Pacific
Data Format: NASA ICARTT (.ict) files, organised by instrument subdirectory
  - Subdirectories: DLH-H2O/, MMS/, NOAA-H2O/
  - Time columns: DLH-H2O → Time_UTC, MMS → TIME_UTC, NOAA-H2O → NW_UTC_s
  - Column names are taken from the last ICARTT header line
  - Data columns prefixed with the instrument stem (e.g. MMS-1HZ, DLH-H2O)

Key variables
-------------
- DLH-H2O_H2O_ppmv   : Diode Laser Hygometer water vapour (ppmv)
- MMS-1HZ_T          : Raw ambient temperature (×0.01 → Kelvin)
- MMS-1HZ_P          : Raw ambient static pressure (×0.01 → hPa)
- MMS-1HZ_G_LAT      : GPS latitude (°N)
- MMS-1HZ_G_LONG     : GPS longitude (°E)
- MMS-1HZ_G_ALT      : GPS altitude (m)
- NOAA-H2O_NW_WV_H2O_ppm : NOAA TDL water vapour (ppmv) — available but not
                           used as primary Si source due to inlet artifacts

Correction factors (applied during Si calculation)
--------------------------------------------------
coef_temp     = 0.01   (MMS-1HZ_T raw → Kelvin)
coef_pressure = 0.01   (MMS-1HZ_P raw → hPa)

Si derivation (primary: DLH)
-----------------------------
  P = MMS-1HZ_P × 0.01          [hPa]
  T = MMS-1HZ_T × 0.01          [K]
  e   = (DLH-H2O_H2O_ppmv / 1e6) × P
  e_s = 6.112 × exp(22.46 × (T − 273.15) / (T − 0.55))   [hPa]
  Si  = e / e_s − 1
  Physical validity filter applied before Si: P > 0, DLH > 0

Invalid-value sentinels
-----------------------
Explicit flags: −9999, −9999.99, −7777, −7777.77, −8888, −8888.88
DLH values ≤ 0 are treated as invalid after flag masking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COEF_TEMP = 0.01       # raw MMS-1HZ_T → Kelvin
COEF_PRESSURE = 0.01   # raw MMS-1HZ_P → hPa

# Explicit ICARTT fill / sentinel values
_MISSING_FLAGS: list[float] = [
    -9999.0, -9999.99,
    -8888.0, -8888.88,
    -7777.0, -7777.77,
]

# Time column per instrument directory
_TIME_COLS: dict[str, str] = {
    "NOAA-H2O": "NW_UTC_s",
    "MMS":      "TIME_UTC",
    "DLH-H2O":  "Time_UTC",
}


# ---------------------------------------------------------------------------
# ICARTT parsing helpers
# ---------------------------------------------------------------------------

def _parse_ict_file(filepath: Path) -> pd.DataFrame:
    """Parse a single NASA ICARTT (.ict) file for the POSIDON campaign.

    Returns a DataFrame with *datetime_utc* as the first column and all data
    columns prefixed with the instrument stem (e.g. ``MMS-1HZ_``).
    """
    with open(filepath, "r", errors="replace") as fh:
        first_line = fh.readline().strip()
        n_header = int(first_line.split(",")[0])
        fh.seek(0)
        header_lines = [fh.readline().rstrip("\n") for _ in range(n_header)]

    # --- Identify flight date ---
    flight_date: Optional[pd.Timestamp] = None
    for line in header_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                if 1990 < yr < 2100 and 1 <= mo <= 12 and 1 <= dy <= 31:
                    flight_date = pd.Timestamp(year=yr, month=mo, day=dy, tz="UTC")
                    break
            except (ValueError, OverflowError):
                continue

    if flight_date is None:
        raise ValueError(
            f"Could not find flight date in ICARTT header of '{filepath.name}'"
        )

    # --- Column names from last header line ---
    columns = [c.strip() for c in header_lines[-1].split(",") if c.strip()]

    # --- Instrument directory → time column ---
    instrument_dir = filepath.parent.name
    time_col = _TIME_COLS.get(instrument_dir)
    if time_col is None:
        raise ValueError(
            f"Unknown instrument directory '{instrument_dir}' in '{filepath}'. "
            f"Expected one of: {list(_TIME_COLS.keys())}"
        )
    if time_col not in columns:
        raise ValueError(
            f"Time column '{time_col}' not found in '{filepath.name}'. "
            f"Available: {columns}"
        )

    # --- Read CSV data ---
    df = pd.read_csv(
        filepath,
        skiprows=n_header,
        names=columns,
        na_values=["", " ", "NA", "NaN"],
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()

    # --- Replace explicit fill values with NaN ---
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == time_col:
            continue
        df[col] = df[col].replace(_MISSING_FLAGS, np.nan)

    # --- Build UTC timestamps ---
    df["datetime_utc"] = flight_date + pd.to_timedelta(
        df[time_col].astype(float), unit="s"
    )

    # --- Prefix data columns with instrument stem ---
    file_prefix = filepath.stem.split("_")[0]
    data_cols = [c for c in df.columns if c not in (time_col, "datetime_utc")]
    df = df.rename(columns={c: f"{file_prefix}_{c}" for c in data_cols})

    keep = ["datetime_utc"] + [f"{file_prefix}_{c}" for c in data_cols]
    return df[keep]


def _combine_ict_files(
    file_list: list[Path],
    time_tolerance: str = "1s",
) -> pd.DataFrame:
    """Load and merge all POSIDON ICT files across instruments.

    Files from the same instrument are concatenated; then instruments are
    joined with ``merge_asof`` (±*time_tolerance*) on *datetime_utc*.
    """
    dfs: list[pd.DataFrame] = []
    for fp in file_list:
        try:
            dfs.append(_parse_ict_file(fp))
        except Exception as exc:
            print(f"[POSIDON] Warning: skipping '{fp.name}': {exc}")

    if not dfs:
        raise ValueError("No POSIDON ICT files could be loaded.")

    # Group by instrument prefix (first segment of first data column name)
    instrument_dfs: dict[str, list[pd.DataFrame]] = {}
    for df in dfs:
        data_cols = [c for c in df.columns if c != "datetime_utc"]
        if not data_cols:
            continue
        prefix = data_cols[0].split("_")[0]
        instrument_dfs.setdefault(prefix, []).append(df)

    # Concatenate within each instrument (multiple flight days)
    merged_instruments: dict[str, pd.DataFrame] = {}
    for prefix, df_list in instrument_dfs.items():
        merged_instruments[prefix] = (
            pd.concat(df_list, ignore_index=True)
            .sort_values("datetime_utc")
            .reset_index(drop=True)
        )

    # Merge across instruments using merge_asof
    inst_names = sorted(merged_instruments.keys())
    combined = merged_instruments[inst_names[0]].set_index("datetime_utc")
    for name in inst_names[1:]:
        right = merged_instruments[name].set_index("datetime_utc")
        combined = pd.merge_asof(
            combined.sort_index(),
            right.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(time_tolerance),
            direction="nearest",
        )

    return combined.reset_index()


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_posidon(
    data_dir: Union[str, Path],
    pattern: str = "*.ict",
    time_tolerance: str = "1s",
) -> pd.DataFrame:
    """Load all POSIDON ICT files from *data_dir* and return a merged DataFrame.

    The function searches *data_dir* recursively for files matching *pattern*,
    parses them as NASA ICARTT files, applies MMS correction factors, computes
    supersaturation with respect to ice (Si) from DLH measurements, and
    returns a tidy DataFrame.

    Parameters
    ----------
    data_dir:
        Root directory containing instrument subdirectories
        (``DLH-H2O/``, ``MMS/``, ``NOAA-H2O/``).
    pattern:
        Glob pattern for ICT files (default ``*.ict``).
    time_tolerance:
        Maximum time gap for ``merge_asof`` across instruments (default ``"1S"``).

    Returns
    -------
    pd.DataFrame
        Columns include ``datetime_utc``, ``MMS-1HZ_T``, ``MMS-1HZ_P``,
        ``MMS-1HZ_G_LAT``, ``MMS-1HZ_G_LONG``, ``MMS-1HZ_G_ALT``,
        ``DLH-H2O_H2O_ppmv``, ``T_K``, ``P_hPa``, ``Si``.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under '{data_dir}'"
        )

    combined = _combine_ict_files(files, time_tolerance=time_tolerance)

    # --- Apply correction factors ---
    if "MMS-1HZ_T" in combined.columns:
        combined["T_K"] = combined["MMS-1HZ_T"] * COEF_TEMP
    else:
        combined["T_K"] = np.nan

    if "MMS-1HZ_P" in combined.columns:
        combined["P_hPa"] = combined["MMS-1HZ_P"] * COEF_PRESSURE
    else:
        combined["P_hPa"] = np.nan

    # --- Calculate Si from DLH ---
    dlh_col = "DLH-H2O_H2O_ppmv"
    valid = (
        combined["P_hPa"].notna()
        & (combined["P_hPa"] > 0)
        & combined["T_K"].notna()
        & combined[dlh_col].notna()
        & (combined[dlh_col] > 0)
    ) if dlh_col in combined.columns else pd.Series(False, index=combined.index)

    si = pd.Series(np.nan, index=combined.index)
    if valid.any():
        T = combined.loc[valid, "T_K"]
        P = combined.loc[valid, "P_hPa"]
        ppmv = combined.loc[valid, dlh_col]
        e_s = 6.112 * np.exp((22.46 * (T - 273.15)) / (T - 0.55))
        e = (ppmv / 1e6) * P
        si.loc[valid] = (e / e_s) - 1

    # Clip to physically meaningful range
    combined["Si"] = si.clip(-1.0, 1.0)

    return combined


# ---------------------------------------------------------------------------
# Standard-output extractor
# ---------------------------------------------------------------------------

def extract_posidon_standard(
    df: pd.DataFrame,
    campaign: str = "POSIDON",
) -> pd.DataFrame:
    """Extract a standardised set of columns from a loaded POSIDON DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        ``Timestamp``, ``Tair_K``, ``Tair_C``, ``Pressure_hPa``,
        ``Si``, ``Lat``, ``Lon``, ``Alt_m``, ``Campaign``
    """
    out = pd.DataFrame()
    out["Timestamp"] = df["datetime_utc"]

    # Temperature
    out["Tair_K"] = df.get("T_K", np.nan)
    out["Tair_C"] = out["Tair_K"] - 273.15 if "T_K" in df.columns else np.nan

    # Pressure
    out["Pressure_hPa"] = df.get("P_hPa", np.nan)

    # Supersaturation
    out["Si"] = df.get("Si", np.nan)

    # Position
    out["Lat"]   = df.get("MMS-1HZ_G_LAT",  np.nan)
    out["Lon"]   = df.get("MMS-1HZ_G_LONG", np.nan)
    out["Alt_m"] = df.get("MMS-1HZ_G_ALT",  np.nan)

    out["Campaign"] = campaign

    return out.reset_index(drop=True)
