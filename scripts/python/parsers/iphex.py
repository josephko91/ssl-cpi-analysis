"""
IPHEX (Integrated Precipitation and Hydrology Experiment) campaign data parser.

Campaign: IPHEX
Data Format: NASA ICARTT-style text files

Required variables
------------------
- FrostPoint: chilled mirror hygrometer measurement (C)
- Air_Temp: ambient air temperature (C)
- STATIC_PR: static pressure (hPa)

Si derivation
-------------
Uses the notebook-validated formula:
    e  = 6.112 * exp(22.46 * Tf / (272.62 + Tf))
    ei = 6.112 * exp(22.46 * Ta / (272.62 + Ta))
    Si = e / ei - 1
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import clean_column_name, extract_takeoff_date, COMMON_NA_VALUES


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


def _find_header_line(lines: list[str], n_header: int) -> str:
    candidates = []
    for idx in range(max(0, n_header - 5), n_header):
        line = lines[idx].strip()
        score = 0
        if "FrostPoint" in line:
            score += 3
        if "Air_Temp" in line:
            score += 3
        if "STATIC_PR" in line:
            score += 3
        if line.count(",") >= 5:
            score += 1
        if len(line.split()) >= 6:
            score += 1
        candidates.append((score, idx, line))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]


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
    filepath = Path(filepath)

    with open(filepath) as f:
        lines = f.readlines()

    n_header = int(re.split(r"[\s,]+", lines[0].strip())[0])
    header_line = _find_header_line(lines, n_header)

    delimiter = "," if "," in header_line else r"\s+"
    if delimiter == ",":
        columns = [clean_column_name(c) for c in header_line.split(",") if c.strip()]
    else:
        columns = [clean_column_name(c) for c in header_line.split() if c.strip()]

    takeoff_date = extract_takeoff_date(lines[:n_header])

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        skiprows=n_header,
        names=columns,
        na_values=COMMON_NA_VALUES,
        engine="python",
        on_bad_lines="skip",
    )

    if df.empty:
        return df

    for required in ("FrostPoint", "Air_Temp", "STATIC_PR"):
        if required not in df.columns:
            raise KeyError(f"Missing required IPHEX column '{required}' in {filepath.name}")

    df["FrostPoint"] = _coerce_and_mask(df["FrostPoint"])
    df["Air_Temp"] = _coerce_and_mask(df["Air_Temp"])
    df["STATIC_PR"] = _coerce_and_mask(df["STATIC_PR"])

    med_t = np.nanmedian(df["Air_Temp"].to_numpy(dtype=float))
    if np.isfinite(med_t) and med_t > 150:
        df["Air_Temp"] = df["Air_Temp"] - 273.15
    med_fp = np.nanmedian(df["FrostPoint"].to_numpy(dtype=float))
    if np.isfinite(med_fp) and med_fp > 150:
        df["FrostPoint"] = df["FrostPoint"] - 273.15

    med_p = np.nanmedian(df["STATIC_PR"].to_numpy(dtype=float))
    if np.isfinite(med_p) and 2000 < med_p < 120000:
        df["STATIC_PR"] = df["STATIC_PR"] / 100.0

    df.loc[(df["Air_Temp"] < -95) | (df["Air_Temp"] > 60), "Air_Temp"] = np.nan
    df.loc[(df["FrostPoint"] < -120) | (df["FrostPoint"] > 40), "FrostPoint"] = np.nan
    df.loc[(df["STATIC_PR"] < 50) | (df["STATIC_PR"] > 1100), "STATIC_PR"] = np.nan

    df.loc[df["FrostPoint"] > (df["Air_Temp"] + 20), "FrostPoint"] = np.nan

    df["Si"] = _compute_si_from_frostpoint(df["FrostPoint"], df["Air_Temp"])
    df.loc[(df["Si"] < -1.0) | (df["Si"] > 5.0), "Si"] = np.nan

    time_col = None
    for candidate in ("Time", "Time_Start", "Time_UTC", "UTC"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df["Timestamp"] = pd.to_datetime(
            df[time_col].apply(
                lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
            ),
            utc=True,
        )
    else:
        df["Timestamp"] = pd.NaT

    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower()), None)

    if lat_col:
        df[lat_col] = _coerce_and_mask(df[lat_col])
        df.loc[(df[lat_col] < -90) | (df[lat_col] > 90), lat_col] = np.nan
    if lon_col:
        df[lon_col] = _coerce_and_mask(df[lon_col])
        df.loc[(df[lon_col] < -180) | (df[lon_col] > 180), lon_col] = np.nan
    if alt_col:
        df[alt_col] = _coerce_and_mask(df[alt_col])
        df.loc[(df[alt_col] < -500) | (df[alt_col] > 25000), alt_col] = np.nan

    df["source_file"] = filepath.name
    df["Campaign"] = "IPHEX"

    _print_distribution_check(df["Si"])

    return df


def load_iphex(data_dir: Union[str, Path], pattern: str = "*.ict") -> pd.DataFrame:
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file()]

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
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower()), None)

    return pd.DataFrame(
        {
            "Timestamp": df.get("Timestamp", pd.NaT),
            "Tair_C": df.get("Air_Temp", np.nan),
            "Si": df.get("Si", np.nan),
            "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
            "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
            "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
            "Campaign": df.get("Campaign", "IPHEX"),
            "source_file": df.get("source_file", ""),
        }
    )
