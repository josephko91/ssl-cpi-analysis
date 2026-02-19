"""
ATTREX (Airborne Tropical TRopopause EXperiment) campaign data parser.

Campaign: ATTREX Global Hawk aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/attrex/id4/GHawk
Data Format: NASA ICARTT (.ict) files

Water vapor instruments: NOAA-H2O, DLH (Diode Laser Hygrometer)
Meteorological data: MMS (Meteorological Measurement System)

Notes
-----
ATTREX data is split across multiple instruments in separate .ict files.
Temperature and pressure come from the MMS instrument, while water vapor
comes from DLH-H2O and/or NOAA-H2O. These must be merged across instruments
(via merge_asof on datetime_utc) before Si can be computed.

MMS raw values for temperature and pressure are scaled by 0.01
(i.e., stored as integers; multiply by 0.01 to get Kelvin / hPa).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional
from collections import defaultdict

from .utils import si_from_ppmv


# ---------------------------------------------------------------------------
# Instrument-specific time column names
# ---------------------------------------------------------------------------
ATTREX_TIME_COLS: Dict[str, str] = {
    "DLH-H2O": "Time_UTC",
    "NOAA-H2O": "NW_UTC_s",
    "MMS": "TIME_UTC",
}

# Fallback patterns (checked in order) when the canonical name is missing
_TIME_FALLBACKS = ["Time_Start", "Time_Mid", "time"]

# Missing value flags used across ATTREX .ict files
ATTREX_MISSING_FLAGS = [-9999, -9999.99, -7777, -7777.77, -8888, -8888.88]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _replace_invalid_values(value: str) -> float:
    """Replace invalid / missing-flag values during CSV parsing."""
    try:
        val = float(value)
        if val in ATTREX_MISSING_FLAGS:
            return np.nan
        return val
    except (ValueError, TypeError):
        return np.nan


def _infer_instrument(filepath: Path) -> str:
    """
    Infer the instrument name from the parent directory or filename.

    The ATTREX archive is typically organized as::

        ATTREX/
            DLH-H2O/
                DLH-H2O_*.ict
            NOAA-H2O/
                NOAA-H2O_*.ict
            MMS/ (or MMS-1HZ/)
                MMS-1HZ_*.ict

    Returns one of the keys in ATTREX_TIME_COLS, or the raw parent name.
    """
    parent = filepath.parent.name
    stem = filepath.stem.upper()
    for key in ATTREX_TIME_COLS:
        if key.upper() in parent.upper() or key.upper().replace("-", "") in stem.replace("-", ""):
            return key
    # Check for MMS-1HZ variant
    if "MMS" in parent.upper() or "MMS" in stem:
        return "MMS"
    return parent


def _parse_ict_file(
    filepath: Path,
    time_col: str = "Time_Start",
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Parse a NASA ICARTT format (.ict) file.

    Parameters
    ----------
    filepath : Path
        Path to the .ict file.
    time_col : str
        Preferred time column name for this instrument.
    prefix : str, optional
        If provided, non-time data columns are prefixed (e.g. ``"DLH-H2O_"``).

    Returns
    -------
    pd.DataFrame
        Parsed data with a ``datetime_utc`` column.
    """
    with open(filepath, "r") as f:
        first_line = f.readline().strip()
        n_header_lines = int(first_line.split(",")[0])

        f.seek(0)
        header_lines = [f.readline().strip() for _ in range(n_header_lines)]

    # --- flight date from header (ICARTT: line 7 is "YYYY, MM, DD, …") ---
    flight_date = None
    for line in header_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                if 1990 < year < 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    flight_date = pd.Timestamp(year=year, month=month, day=day)
                    break
            except ValueError:
                continue

    if flight_date is None:
        raise ValueError(f"Could not find flight date in header of {filepath.name}")

    # --- column names from last header line ---
    col_line = header_lines[-1]
    columns = [col.strip() for col in col_line.split(",") if col.strip()]

    # --- read data ---
    df = pd.read_csv(
        filepath,
        skiprows=n_header_lines,
        names=columns,
        converters={col: _replace_invalid_values for col in columns},
        skipinitialspace=True,
        on_bad_lines="skip",
    )
    df.columns = df.columns.str.strip()

    # --- replace remaining missing-flag values ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(ATTREX_MISSING_FLAGS, np.nan)

    # --- create datetime_utc from the time column ---
    used_time_col = None
    if time_col in df.columns:
        used_time_col = time_col
    else:
        # Try fallbacks
        for fb in _TIME_FALLBACKS:
            if fb in df.columns:
                used_time_col = fb
                break
        if used_time_col is None:
            # Try any column with 'time' or 'utc' in its name
            time_candidates = [c for c in df.columns if "time" in c.lower() or "utc" in c.lower()]
            if time_candidates:
                used_time_col = time_candidates[0]

    if used_time_col is not None:
        df["datetime_utc"] = flight_date + pd.to_timedelta(df[used_time_col], unit="s")
    else:
        # Use the first column as seconds-of-day
        df["datetime_utc"] = flight_date + pd.to_timedelta(df.iloc[:, 0], unit="s")

    # --- optionally prefix data columns (not datetime_utc) ---
    if prefix:
        rename_map = {}
        for c in df.columns:
            if c == "datetime_utc":
                continue
            # Don't double-prefix
            if not c.startswith(prefix):
                rename_map[c] = f"{prefix}_{c}"
        df.rename(columns=rename_map, inplace=True)

    df["source_file"] = filepath.name
    return df


# ---------------------------------------------------------------------------
# Cross-instrument merge
# ---------------------------------------------------------------------------

def _combine_ict_files(
    files: List[Path],
    time_tolerance: str = "1s",
) -> pd.DataFrame:
    """
    Parse all .ict files, group by instrument, and merge via ``merge_asof``.

    Parameters
    ----------
    files : list of Path
        All .ict files found under the ATTREX data directory.
    time_tolerance : str
        Tolerance for ``merge_asof`` (default ``'1s'``).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns from all instruments and ``datetime_utc``.
    """
    # Group files by instrument
    instrument_files: Dict[str, List[Path]] = defaultdict(list)
    for f in files:
        inst = _infer_instrument(f)
        instrument_files[inst].append(f)

    # Parse and concatenate within each instrument
    instrument_dfs: Dict[str, pd.DataFrame] = {}
    for inst, inst_files in instrument_files.items():
        time_col = ATTREX_TIME_COLS.get(inst, "Time_Start")
        prefix = inst
        parsed = []
        for fp in sorted(inst_files):
            try:
                parsed.append(_parse_ict_file(fp, time_col=time_col, prefix=prefix))
            except Exception as e:
                print(f"  Warning: Could not parse {fp.name} ({inst}): {e}")
        if parsed:
            combined = pd.concat(parsed, ignore_index=True)
            combined.sort_values("datetime_utc", inplace=True)
            combined.reset_index(drop=True, inplace=True)
            instrument_dfs[inst] = combined

    if not instrument_dfs:
        raise ValueError("No ATTREX instrument files could be parsed.")

    # Merge across instruments using merge_asof on datetime_utc
    instruments = list(instrument_dfs.keys())
    merged = instrument_dfs[instruments[0]].copy()

    for inst in instruments[1:]:
        right = instrument_dfs[inst]
        # Drop duplicate non-key columns before merging
        overlap = set(merged.columns) & set(right.columns) - {"datetime_utc"}
        right_clean = right.drop(columns=[c for c in overlap if c != "source_file"], errors="ignore")
        if "source_file" in right_clean.columns and "source_file" in merged.columns:
            right_clean = right_clean.rename(columns={"source_file": f"source_file_{inst}"})
        merged = pd.merge_asof(
            merged.sort_values("datetime_utc"),
            right_clean.sort_values("datetime_utc"),
            on="datetime_utc",
            tolerance=pd.Timedelta(time_tolerance),
            direction="nearest",
        )

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_attrex(
    data_dir: Union[str, Path],
    pattern: str = "*.ict",
) -> pd.DataFrame:
    """
    Load all ATTREX ICT files, merge across instruments, and compute Si.

    The function:
    1. Recursively finds all ``.ict`` files under *data_dir*.
    2. Groups them by instrument (DLH-H2O, NOAA-H2O, MMS, …).
    3. Merges across instruments via ``merge_asof`` on ``datetime_utc``.
    4. Applies MMS scaling (× 0.01 for temperature and pressure).
    5. Computes Si from water-vapor mixing ratio, temperature, and pressure.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing ATTREX instrument sub-folders.
    pattern : str, optional
        Glob pattern for matching files (default: ``"*.ict"``).

    Returns
    -------
    pd.DataFrame
        Combined data from all instruments with ``Si``, ``T_C``, ``Timestamp``.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    print(f"  Found {len(files)} .ict files in {data_dir}")

    # --- merge across instruments ---
    df = _combine_ict_files(files, time_tolerance="1s")

    # --- apply MMS scaling (raw integers → physical units) ---
    # Per ATTREX ICARTT documentation, MMS-1HZ stores T and P as scaled integers:
    #   T_raw  [integer units of 0.01 K]  →  T_K  = T_raw * 0.01
    #   P_raw  [integer units of 0.01 hPa] →  P_hPa = P_raw * 0.01
    # At tropopause altitude (~70–100 hPa), raw P values are ~7000–10000, so a
    # median-based threshold of > 10000 would FAIL to detect the scaling need.
    # We therefore always apply × 0.01 when an MMS column is found.

    # Temperature: look for a prefixed MMS column containing "_T" as a word boundary
    mms_t_col = next(
        (c for c in df.columns if "MMS" in c.upper() and c.upper().endswith("_T")),
        None,
    )
    # Pressure: same logic for "_P"
    mms_p_col = next(
        (c for c in df.columns if "MMS" in c.upper() and c.upper().endswith("_P")),
        None,
    )

    if mms_t_col is not None:
        # Always apply × 0.01 (documented MMS-1HZ scaling; raw ~22000–27000 → 220–270 K)
        df["T"] = df[mms_t_col] * 0.01
        print(f"  Scaled T: {mms_t_col} × 0.01  (sample median raw = {df[mms_t_col].median():.0f})")
    else:
        # Fallback: look for any column named T or starting with T_
        t_fallback = next((c for c in df.columns if c == "T" or c.startswith("T_")), None)
        if t_fallback:
            if df[t_fallback].median() > 200:
                df["T"] = df[t_fallback]
            else:
                df["T"] = df[t_fallback] + 273.15

    if mms_p_col is not None:
        # Always apply × 0.01 (documented MMS-1HZ scaling; raw ~5000–10000 → 50–100 hPa)
        df["P"] = df[mms_p_col] * 0.01
        print(f"  Scaled P: {mms_p_col} × 0.01  (sample median raw = {df[mms_p_col].median():.0f})")
    else:
        p_fallback = next((c for c in df.columns if c == "P" or c.startswith("P_")), None)
        if p_fallback:
            df["P"] = df[p_fallback]

    # --- sanitize T and P: set physically impossible values to NaN ---
    # ATTREX flies near the tropical tropopause (~15-19 km altitude).
    # Physically reasonable ranges:
    #   Temperature: 150–350 K  (tropopause can be ~190 K)
    #   Pressure:    10–1100 hPa (tropopause ~70-100 hPa)
    if "T" in df.columns:
        invalid_t = (df["T"] < 150) | (df["T"] > 350) | df["T"].isna()
        n_bad_t = invalid_t.sum()
        if n_bad_t > 0:
            print(f"  Masking {n_bad_t:,} invalid T values (outside 150–350 K)")
            df.loc[invalid_t, "T"] = np.nan

    if "P" in df.columns:
        invalid_p = (df["P"] <= 0) | (df["P"] > 1100) | df["P"].isna()
        n_bad_p = invalid_p.sum()
        if n_bad_p > 0:
            print(f"  Masking {n_bad_p:,} invalid P values (outside 0–1100 hPa)")
            df.loc[invalid_p, "P"] = np.nan

    # --- find water vapor column(s) ---
    # Prefer DLH, fall back to NOAA
    h2o_cols = [c for c in df.columns if "h2o" in c.lower() and "ppm" in c.lower()]
    dlh_h2o = next((c for c in h2o_cols if "dlh" in c.lower()), None)
    noaa_h2o = next((c for c in h2o_cols if "noaa" in c.lower() or "nw" in c.lower()), None)
    h2o_col = dlh_h2o or noaa_h2o or (h2o_cols[0] if h2o_cols else None)

    # Sanitize H2O: water vapor mixing ratio must be > 0
    for wv_col in [dlh_h2o, noaa_h2o]:
        if wv_col and wv_col in df.columns:
            invalid_wv = (df[wv_col] <= 0) | df[wv_col].isna()
            n_bad_wv = invalid_wv.sum()
            if n_bad_wv > 0:
                print(f"  Masking {n_bad_wv:,} invalid {wv_col} values (≤ 0 or NaN)")
                df.loc[invalid_wv, wv_col] = np.nan

    # --- compute Si (only where T, P, and H2O are all valid) ---
    if h2o_col and "T" in df.columns and "P" in df.columns:
        # Build a validity mask: all three inputs must be finite and positive
        valid = df[h2o_col].notna() & df["T"].notna() & df["P"].notna()
        n_valid = valid.sum()
        print(f"  {n_valid:,} rows with valid T, P, and {h2o_col} for Si calculation")

        df["Si"] = np.nan
        if n_valid > 0:
            df.loc[valid, "Si"] = si_from_ppmv(
                df.loc[valid, h2o_col], df.loc[valid, "T"], df.loc[valid, "P"]
            )
        print(f"  Computed Si using H2O={h2o_col}, T from {'MMS' if mms_t_col else 'fallback'}, "
              f"P from {'MMS' if mms_p_col else 'fallback'}")

        # Also compute Si from the second H2O source if available
        h2o_alt = noaa_h2o if h2o_col == dlh_h2o else dlh_h2o
        if h2o_alt and h2o_alt in df.columns:
            valid_alt = df[h2o_alt].notna() & df["T"].notna() & df["P"].notna()
            df["Si_alt"] = np.nan
            if valid_alt.sum() > 0:
                df.loc[valid_alt, "Si_alt"] = si_from_ppmv(
                    df.loc[valid_alt, h2o_alt], df.loc[valid_alt, "T"], df.loc[valid_alt, "P"]
                )
            print(f"  Also computed Si_alt using H2O={h2o_alt} ({valid_alt.sum():,} valid rows)")
    else:
        missing = []
        if not h2o_col:
            missing.append("H2O (ppmv)")
        if "T" not in df.columns:
            missing.append("Temperature")
        if "P" not in df.columns:
            missing.append("Pressure")
        print(f"  WARNING: Cannot compute Si — missing columns: {', '.join(missing)}")
        print(f"  Available columns: {df.columns.tolist()}")
        df["Si"] = np.nan

    # --- final Si sanity check: values outside [-1, 10] are suspect ---
    if "Si" in df.columns:
        extreme = (df["Si"].abs() > 10) & df["Si"].notna()
        n_extreme = extreme.sum()
        if n_extreme > 0:
            print(f"  Masking {n_extreme:,} extreme Si values (|Si| > 10)")
            df.loc[extreme, "Si"] = np.nan
    if "Si_alt" in df.columns:
        extreme_alt = (df["Si_alt"].abs() > 10) & df["Si_alt"].notna()
        if extreme_alt.sum() > 0:
            df.loc[extreme_alt, "Si_alt"] = np.nan

    # --- temperature in Celsius ---
    if "T" in df.columns:
        df["T_C"] = df["T"] - 273.15

    # --- timestamp ---
    if "datetime_utc" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["datetime_utc"], utc=True)

    df["Campaign"] = "ATTREX"

    return df


def extract_attrex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from ATTREX data.

    Parameters
    ----------
    df : pd.DataFrame
        Data loaded by :func:`load_attrex`.

    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns (may be prefixed with instrument name)
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower()), None)

    # Consolidate source_file columns
    src_cols = [c for c in df.columns if c.startswith("source_file")]
    if src_cols:
        source = df[src_cols[0]].astype(str)
        for sc in src_cols[1:]:
            source = source + "+" + df[sc].astype(str)
    else:
        source = ""

    return pd.DataFrame({
        "Timestamp": df.get("Timestamp", pd.NaT),
        "Tair_C": df.get("T_C", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df[lat_col] if lat_col else np.nan,
        "Lon": df[lon_col] if lon_col else np.nan,
        "Alt_m": df[alt_col] if alt_col else np.nan,
        "Campaign": df.get("Campaign", "ATTREX"),
        "source_file": source,
    })
