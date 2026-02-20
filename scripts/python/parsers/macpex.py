"""
MACPEX (Mid-latitude Airborne Cirrus Properties Experiment) campaign data parser.

Campaign: MACPEX — WB-57 aircraft, April–May 2011
Data Source: https://espoarchive.nasa.gov/archive/browse/macpex
Data Format: NASA ICARTT (.ict) files, one instrument per sub-folder

Water vapor instruments
-----------------------
DLH  — Diode Laser Hygrometer, column ``DLH-H2O_H2O_ppmv``  ← DEFAULT
         Best overall data coverage (≈70 %), recommended for Si.
HWV  — Harvard Water Vapor instrument, column ``HWV_H2O``
         Good quality but lower coverage (≈56 %) than DLH.
JLH  — JPL Laser Hygrometer, column ``JLH_H2O(v)_ppmv``
         Limited coverage (≈13 %); use for cross-validation only.

Meteorological data
-------------------
Temperature and pressure come from the MMS-MetData instrument.
The raw ICT values are stored as scaled integers:
  - T_raw  [0.01 K units]  →  T_K  = T_raw × 0.01   (Kelvin)
  - P_raw  [0.1 hPa units] →  P_hPa = P_raw × 0.1   (hPa)

Si derivation
-------------
Uses the Tetens-style formula consistent with the MACPEX notebook analysis:
    e_s = 6.112 × exp( 22.46 × (T_K − 273.15) / (T_K − 0.55) )   [hPa]
    e   = (wv_ppmv / 1e6) × P_hPa                                   [hPa]
    Si  = e / e_s − 1

This is equivalent to si_from_ppmv() in utils.py, which is used internally.

Physical validity ranges applied before Si computation
------------------------------------------------------
  Temperature (K): 150–350
  Pressure (hPa):  10–1100
  Water vapor (ppmv): > 0
  Si (final): −1.0 to +1.0  (cirrus/UT regime; values outside this range
                               are almost certainly instrument artefacts)

Missing-value flags masked to NaN (from MACPEX ICT headers and notebook)
-----------------------------------------------------------------------
  -9999, -9999.99, -9999.0
  -8888, -8888.88
  -7777, -7777.77
  -999.99
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from .utils import si_from_ppmv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: All known missing-value flag are turned into NaN before any computation.
MACPEX_MISSING_FLAGS: list = [
    -9999, -9999.0, -9999.99,
    -8888, -8888.88,
    -7777, -7777.77,
    -999.99,
]

#: Instrument folder → time-column name mapping.
#: Built from a survey of MACPEX ICT headers; add entries as needed.
#: Folder names may vary between data mirror sites (e.g. "DLH" vs "DLH-H2O"),
#: so we list all known variants.  The parser also falls back to any column
#: containing 'time' or 'utc' if the explicit lookup misses.
MACPEX_TIME_COLS: Dict[str, str] = {
    # --- canonical short folder names (as on ESPO archive) ---
    "DLH":            "Time_Start",
    "HWV":            "Time_Start",
    "JLH":            "Time_Start",
    "MMS-Met":        "Time_Start",
    "CIMS":           "Time_Start",
    "CLH":            "Time_Start",
    "ULH":            "Time_Start",
    # --- long folder-name variants (some mirrors use these) ---
    "DLH-H2O":        "Time_Start",
    "MMS-MetData":    "Time_Start",
    "CIMS-H2O":       "Time_Start",
    "CLH-Enhanced":   "Time_Start",
}
_TIME_FALLBACKS = ["Time_Start", "Time_Mid", "Time_Stop", "time_utc", "Time_UTC"]

#: MMS-MetData raw-integer scaling factors (documented in MACPEX ICT headers).
MMS_T_SCALE = 0.01   # raw integer × 0.01 → Kelvin
MMS_P_SCALE = 0.10   # raw integer × 0.10 → hPa

#: Preferred water-vapor source column names after instrument-prefix renaming.
#: These names must cover both short and long folder-name variants (the prefix
#: applied during parsing equals the instrument sub-folder name).
WV_SOURCE_COLS: Dict[str, List[str]] = {
    "DLH": [
        "DLH_H2O_ppmv",           # short folder name  (DLH/)
        "DLH-H2O_H2O_ppmv",       # long folder name   (DLH-H2O/)
        "H2O_ppmv",               # fallback (un-prefixed)
    ],
    "HWV": [
        "HWV_H2O",                # same for both variants
        "H2O",                    # fallback
    ],
    "JLH": [
        "JLH_H2O(v)_ppmv",       # same for both variants
        "JLH-H2O_H2O(v)_ppmv",   # possible long variant
        "H2O(v)_ppmv",            # fallback
    ],
}

WvSource = Literal["DLH", "HWV", "JLH"]

#: Physical validity bounds applied before Si calculation.
T_BOUNDS_K    = (150.0, 350.0)   # MACPEX WB-57 tropopause region
P_BOUNDS_HPA  = (10.0,  1100.0)
WV_MIN_PPMV   = 0.0
SI_BOUNDS     = (-1.0, 1.0)      # cirrus UT regime; outside = artefact


# ---------------------------------------------------------------------------
# Low-level ICT parsing helpers
# ---------------------------------------------------------------------------

def _parse_ict_file(filepath: Path) -> pd.DataFrame:
    """
    Parse a single NASA ICARTT (.ict) file.

    Reads the ICARTT header to determine:
    - The number of header lines (first token of first line).
    - The flight date (first line matching YYYY, MM, DD[, …]).
    - The column names (last header line, comma-separated).

    Returns a DataFrame with a ``datetime_utc`` column constructed from
    elapsed-seconds-since-midnight and the parsed flight date.  All data
    columns are numeric; missing-flag values are replaced with NaN.

    Parameters
    ----------
    filepath : Path
        Path to the .ict file.

    Returns
    -------
    pd.DataFrame
    """
    with open(filepath, "r") as fh:
        raw = fh.read()

    lines = raw.splitlines()

    # --- ICARTT header length ---
    try:
        n_header = int(lines[0].split(",")[0].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Cannot determine header length from first line of {filepath.name}: "
            f"'{lines[0]}'"
        ) from exc

    header_lines = lines[:n_header]

    # --- flight date from header ---
    # ICARTT line format: "YYYY, MM, DD, YYYY, MM, DD"   (flight date, rev date)
    flight_date: Optional[pd.Timestamp] = None
    for line in header_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                if 1990 < yr < 2100 and 1 <= mo <= 12 and 1 <= dy <= 31:
                    flight_date = pd.Timestamp(year=yr, month=mo, day=dy)
                    break
            except ValueError:
                continue

    if flight_date is None:
        raise ValueError(
            f"Could not find flight date in ICARTT header of {filepath.name}"
        )

    # --- column names from last header line ---
    columns = [c.strip() for c in header_lines[-1].split(",") if c.strip()]

    # --- determine time column for this instrument ---
    instrument = filepath.parent.name
    time_col = MACPEX_TIME_COLS.get(instrument)
    if time_col is None or time_col not in columns:
        # Fall back to any recognisable time column
        time_col = next(
            (fb for fb in _TIME_FALLBACKS if fb in columns), None
        )
        if time_col is None:
            time_col = next(
                (c for c in columns if "time" in c.lower() or "utc" in c.lower()),
                columns[0],   # last resort: first column
            )

    # --- parse data block ---
    df = pd.read_csv(
        filepath,
        skiprows=n_header,
        names=columns,
        skipinitialspace=True,
        on_bad_lines="skip",
        engine="python",
    )
    df.columns = df.columns.str.strip()

    # --- coerce all data columns to float ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- replace all known missing-value flags with NaN ---
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(MACPEX_MISSING_FLAGS, np.nan)

    # --- build datetime_utc ---
    if time_col in df.columns:
        elapsed = pd.to_numeric(df[time_col], errors="coerce")
        df["datetime_utc"] = flight_date + pd.to_timedelta(elapsed, unit="s")
    else:
        df["datetime_utc"] = pd.NaT

    # --- rename data columns with instrument prefix so they survive merging ---
    prefix = instrument  # e.g. "DLH-H2O", "MMS-MetData"
    rename_map = {
        c: f"{prefix}_{c}"
        for c in df.columns
        if c not in (time_col, "datetime_utc", "source_file")
        and not c.startswith(prefix)
    }
    df.rename(columns=rename_map, inplace=True)

    df["source_file"] = filepath.name
    return df


# ---------------------------------------------------------------------------
# Multi-instrument merge
# ---------------------------------------------------------------------------

def _load_and_merge(
    files: List[Path],
    time_tolerance: str = "1s",
) -> pd.DataFrame:
    """
    Parse all .ict files, group by instrument folder, concatenate within each
    instrument, then merge across instruments via ``merge_asof``.

    Parameters
    ----------
    files : list of Path
    time_tolerance : str
        Time tolerance for merge_asof (default ``'1s'``).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``datetime_utc`` as the key column.
    """
    by_instrument: Dict[str, List[pd.DataFrame]] = defaultdict(list)

    for fp in files:
        try:
            parsed = _parse_ict_file(fp)
            by_instrument[fp.parent.name].append(parsed)
        except Exception as exc:
            print(f"  Warning: Could not parse {fp.name}: {exc}")

    if not by_instrument:
        raise ValueError("No MACPEX ICT files could be parsed.")

    instrument_dfs: Dict[str, pd.DataFrame] = {}
    for inst, dfs in by_instrument.items():
        combined = pd.concat(dfs, ignore_index=True)
        combined.sort_values("datetime_utc", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        instrument_dfs[inst] = combined
        print(f"  Instrument '{inst}': {len(combined):,} rows, "
              f"{len(combined.columns)} cols")

    # Merge across instruments with merge_asof on datetime_utc
    # Start with MMS-MetData (navigation/met state) as the left frame so every
    # row has valid T and P whenever possible.
    instruments = list(instrument_dfs.keys())
    primary = "MMS-MetData" if "MMS-MetData" in instruments else instruments[0]
    other = [i for i in instruments if i != primary]

    merged = instrument_dfs[primary].copy()

    for inst in other:
        right = instrument_dfs[inst]
        # Drop source_file from right to avoid collision; rename it instead
        if "source_file" in right.columns:
            right = right.rename(columns={"source_file": f"source_file_{inst}"})
        # Drop any columns that already exist in merged (except datetime_utc)
        overlap = set(merged.columns) & set(right.columns) - {"datetime_utc"}
        right = right.drop(columns=list(overlap), errors="ignore")

        merged = pd.merge_asof(
            merged.sort_values("datetime_utc"),
            right.sort_values("datetime_utc"),
            on="datetime_utc",
            tolerance=pd.Timedelta(time_tolerance),
            direction="nearest",
        )

    return merged


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in *df*, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_mms_col(df: pd.DataFrame, suffix: str) -> Optional[str]:
    """
    Find a MMS column by its known suffix.

    Tries short (``MMS-Met_{suffix}``) and long (``MMS-MetData_{suffix}``)
    canonical names first, then any column whose name ends with
    ``_{suffix}`` and contains ``MMS``.
    """
    # Try known canonical names (short folder, long folder)
    for prefix in ("MMS-Met", "MMS-MetData"):
        canonical = f"{prefix}_{suffix}"
        if canonical in df.columns:
            return canonical
    # Broader search
    return next(
        (c for c in df.columns
         if "mms" in c.lower() and c.upper().endswith(f"_{suffix.upper()}")),
        None,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_macpex(
    data_dir: Union[str, Path],
    pattern: str = "*.ict",
    wv_source: WvSource = "DLH",
    time_tolerance: str = "1s",
) -> pd.DataFrame:
    """
    Load all MACPEX ICT files, merge across instruments, and compute Si.

    The function:

    1. Recursively finds all ``.ict`` files under *data_dir*.
    2. Groups them by instrument sub-folder (DLH-H2O, HWV, JLH, MMS-MetData, …).
    3. Concatenates files within each instrument group then merges all groups
       via ``merge_asof`` on ``datetime_utc`` (tolerance = *time_tolerance*).
    4. Applies MMS-MetData raw-integer scaling:
       - Temperature: raw × 0.01 → Kelvin
       - Pressure:    raw × 0.10 → hPa
    5. Replaces all MACPEX missing-value flags (−9999, −8888, −999.99, …)
       with NaN in both raw columns and after scaling.
    6. Applies physical-validity masks (T, P, water vapor) before computing Si.
    7. Computes Si from *wv_source* water vapor (ppmv), T (K), and P (hPa):
       Si = e / e_s − 1  where e_s is saturation vapor pressure over ice.
    8. Clips final Si to [−1.0, +1.0]; values outside this range in the cirrus
       / upper-troposphere regime are almost certainly instrument artefacts.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing MACPEX instrument sub-folders.
    pattern : str, optional
        Glob pattern for matching files (default: ``"*.ict"``).
    wv_source : {"DLH", "HWV", "JLH"}, optional
        Water-vapor instrument to use for Si computation.

        - ``"DLH"`` *(default)* — Diode Laser Hygrometer (``DLH-H2O_H2O_ppmv``).
          Highest overall data coverage (≈70 %) and best quality for Si.
        - ``"HWV"`` — Harvard Water Vapor (``HWV_H2O``).
          Good quality; use when DLH data are unavailable.
        - ``"JLH"`` — JPL Laser Hygrometer (``JLH_H2O(v)_ppmv``).
          Limited coverage (≈13 %); primarily for cross-validation.

    time_tolerance : str, optional
        Tolerance passed to ``merge_asof`` when aligning instruments
        (default: ``"1s"``).

    Returns
    -------
    pd.DataFrame
        Combined data with columns including ``Timestamp``, ``T_K``, ``T_C``,
        ``P_hPa``, ``Si``, and the raw instrument columns.  Also includes
        ``Si_HWV`` and ``Si_JLH`` if those instrument files were loaded
        alongside DLH.

    Raises
    ------
    FileNotFoundError
        If no ``.ict`` files are found under *data_dir*.
    ValueError
        If *wv_source* is not one of the accepted values, or if no files could
        be parsed.
    KeyError
        If the selected water-vapor column cannot be located after merging.
    """
    if wv_source not in WV_SOURCE_COLS:
        raise ValueError(
            f"wv_source must be one of {list(WV_SOURCE_COLS)}, got '{wv_source}'"
        )

    data_dir = Path(data_dir)
    files = list(data_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir}"
        )
    print(f"  Found {len(files)} .ict files in {data_dir}")

    # ------------------------------------------------------------------
    # 1. Parse and merge all instruments
    # ------------------------------------------------------------------
    df = _load_and_merge(files, time_tolerance=time_tolerance)

    # ------------------------------------------------------------------
    # 2. Scale MMS-MetData temperature and pressure from raw integers
    # ------------------------------------------------------------------
    t_raw_col = _find_mms_col(df, "T")
    p_raw_col = _find_mms_col(df, "P")

    if t_raw_col is not None:
        raw_t_med = df[t_raw_col].median()
        df["T_K"] = df[t_raw_col] * MMS_T_SCALE
        print(
            f"  Scaled T: {t_raw_col} × {MMS_T_SCALE}  "
            f"(raw median = {raw_t_med:.0f} → "
            f"{raw_t_med * MMS_T_SCALE:.1f} K)"
        )
    else:
        print("  WARNING: MMS-MetData temperature column not found.")
        df["T_K"] = np.nan

    if p_raw_col is not None:
        raw_p_med = df[p_raw_col].median()
        df["P_hPa"] = df[p_raw_col] * MMS_P_SCALE
        print(
            f"  Scaled P: {p_raw_col} × {MMS_P_SCALE}  "
            f"(raw median = {raw_p_med:.0f} → "
            f"{raw_p_med * MMS_P_SCALE:.1f} hPa)"
        )
    else:
        print("  WARNING: MMS-MetData pressure column not found.")
        df["P_hPa"] = np.nan

    # Re-mask any scaled values that are still at a missing-flag level
    # (edge case: a flag like -9999 × 0.01 = -99.99 is not a valid T_K)
    df["T_K"] = df["T_K"].replace(
        [f * MMS_T_SCALE for f in MACPEX_MISSING_FLAGS], np.nan
    )
    df["P_hPa"] = df["P_hPa"].replace(
        [f * MMS_P_SCALE for f in MACPEX_MISSING_FLAGS], np.nan
    )

    # ------------------------------------------------------------------
    # 3. Physical bounds on T and P
    # ------------------------------------------------------------------
    n_bad_t = ((df["T_K"] < T_BOUNDS_K[0]) | (df["T_K"] > T_BOUNDS_K[1])).sum()
    if n_bad_t:
        print(
            f"  Masking {n_bad_t:,} T values outside "
            f"{T_BOUNDS_K[0]}–{T_BOUNDS_K[1]} K"
        )
    df.loc[
        (df["T_K"] < T_BOUNDS_K[0]) | (df["T_K"] > T_BOUNDS_K[1]), "T_K"
    ] = np.nan

    n_bad_p = ((df["P_hPa"] < P_BOUNDS_HPA[0]) | (df["P_hPa"] > P_BOUNDS_HPA[1])).sum()
    if n_bad_p:
        print(
            f"  Masking {n_bad_p:,} P values outside "
            f"{P_BOUNDS_HPA[0]}–{P_BOUNDS_HPA[1]} hPa"
        )
    df.loc[
        (df["P_hPa"] < P_BOUNDS_HPA[0]) | (df["P_hPa"] > P_BOUNDS_HPA[1]), "P_hPa"
    ] = np.nan

    # ------------------------------------------------------------------
    # 4. Locate water-vapor column(s) and mask invalid WV values
    # ------------------------------------------------------------------
    wv_col = _find_col(df, WV_SOURCE_COLS[wv_source])
    if wv_col is None:
        available = [c for c in df.columns if "h2o" in c.lower() or "wv" in c.lower()]
        raise KeyError(
            f"Could not find water-vapor column for source '{wv_source}'. "
            f"Tried: {WV_SOURCE_COLS[wv_source]}. "
            f"H2O-like columns present: {available}"
        )

    def _sanitize_wv(col: str) -> None:
        """Mask non-physical water vapor values: ≤ 0 or unrealistically large."""
        invalid = (df[col] <= WV_MIN_PPMV) | ~np.isfinite(df[col])
        n_bad = invalid.sum()
        if n_bad:
            print(f"  Masking {n_bad:,} invalid {col} values (≤ 0 or non-finite)")
        df.loc[invalid, col] = np.nan

    _sanitize_wv(wv_col)

    # Also sanitize alternate WV sources if present (for cross-validation columns)
    alt_wv_cols: Dict[str, Optional[str]] = {}
    for src, cands in WV_SOURCE_COLS.items():
        if src == wv_source:
            continue
        c = _find_col(df, cands)
        if c is not None:
            _sanitize_wv(c)
            alt_wv_cols[src] = c

    # ------------------------------------------------------------------
    # 5. Compute Si from the selected water-vapor source
    # ------------------------------------------------------------------
    valid_mask = df[wv_col].notna() & df["T_K"].notna() & df["P_hPa"].notna()
    n_valid = valid_mask.sum()
    print(
        f"  {n_valid:,} rows with valid {wv_col}, T, P "
        f"for Si calculation (source: {wv_source})"
    )

    df["Si"] = np.nan
    if n_valid > 0:
        df.loc[valid_mask, "Si"] = si_from_ppmv(
            df.loc[valid_mask, wv_col],
            df.loc[valid_mask, "T_K"],
            df.loc[valid_mask, "P_hPa"],
        )

    # ------------------------------------------------------------------
    # 6. Clip Si to physically plausible range for cirrus / UT
    # ------------------------------------------------------------------
    n_clipped = ((df["Si"] < SI_BOUNDS[0]) | (df["Si"] > SI_BOUNDS[1])).sum()
    if n_clipped:
        print(
            f"  Masking {n_clipped:,} Si values outside "
            f"[{SI_BOUNDS[0]}, {SI_BOUNDS[1]}] (instrument artefacts)"
        )
    df.loc[
        (df["Si"] < SI_BOUNDS[0]) | (df["Si"] > SI_BOUNDS[1]), "Si"
    ] = np.nan

    # ------------------------------------------------------------------
    # 7. Optional Si for alternate water-vapor sources
    # ------------------------------------------------------------------
    for src, alt_col in alt_wv_cols.items():
        if alt_col is None:
            continue
        si_col = f"Si_{src}"
        alt_valid = df[alt_col].notna() & df["T_K"].notna() & df["P_hPa"].notna()
        df[si_col] = np.nan
        if alt_valid.sum() > 0:
            df.loc[alt_valid, si_col] = si_from_ppmv(
                df.loc[alt_valid, alt_col],
                df.loc[alt_valid, "T_K"],
                df.loc[alt_valid, "P_hPa"],
            )
            df.loc[
                (df[si_col] < SI_BOUNDS[0]) | (df[si_col] > SI_BOUNDS[1]), si_col
            ] = np.nan
            print(
                f"  Also computed {si_col} using {alt_col} "
                f"({alt_valid.sum():,} valid rows)"
            )

    # ------------------------------------------------------------------
    # 8. Derived/convenience columns
    # ------------------------------------------------------------------
    df["T_C"] = df["T_K"] - 273.15

    # Standard Timestamp (UTC-aware)
    if "datetime_utc" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")

    df["Campaign"] = "MACPEX"

    return df


def extract_macpex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return standardized columns suitable for combined multi-campaign analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Data loaded by :func:`load_macpex`.

    Returns
    -------
    pd.DataFrame
        Columns: Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign, source_file.
    """
    # Position columns may be prefixed with the MMS instrument name
    lat_col = _find_col(
        df, ["MMS-Met_LAT", "MMS-Met_Lat",
             "MMS-MetData_LAT", "MMS-MetData_Lat", "LAT", "Lat", "latitude"]
    ) or next((c for c in df.columns if "lat" in c.lower()), None)

    lon_col = _find_col(
        df, ["MMS-Met_LON", "MMS-Met_Lon",
             "MMS-MetData_LON", "MMS-MetData_Lon", "LON", "Lon", "longitude"]
    ) or next((c for c in df.columns if "lon" in c.lower()), None)

    alt_col = _find_col(
        df, ["MMS-Met_ALT", "MMS-Met_Alt",
             "MMS-MetData_ALT", "MMS-MetData_Alt", "ALT", "Alt", "ALTITUDE"]
    ) or next((c for c in df.columns if "alt" in c.lower()), None)

    # Consolidate source_file columns (one per instrument after merging)
    src_cols = [c for c in df.columns if c.startswith("source_file")]
    if src_cols:
        source = df[src_cols[0]].astype(str)
        for sc in src_cols[1:]:
            source = source + "|" + df[sc].astype(str)
    else:
        source = pd.Series("", index=df.index)

    return pd.DataFrame({
        "Timestamp": df.get("Timestamp", pd.NaT),
        "Tair_C":    df.get("T_C", np.nan),
        "Si":        df.get("Si", np.nan),
        "Lat":       df[lat_col] if lat_col else np.nan,
        "Lon":       df[lon_col] if lon_col else np.nan,
        "Alt_m":     df[alt_col] if alt_col else np.nan,
        "Campaign":  df.get("Campaign", "MACPEX"),
        "source_file": source,
    })
