"""
ICE-L (Ice in Clouds Experiment - Layer clouds) campaign data parser.

Campaign: ICE-L
Data Format: RAF Nimbus NetCDF files (.nc)

Notebook-validated variable mapping
----------------------------------
- Temperature: ATX
- Pressure: PSXC
- Relative humidity: RHUM (pre-calculated)

Si derivation
-------------
Si is derived directly from RHUM using:
    Si = RHUM / 100 - 1
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from .utils import si_from_rh


ICE_L_FILE_RE = re.compile(
    r"RF\d+\.(\d{8})\.(\d{6})_(\d{6})\.PNI\.nc$",
    re.IGNORECASE,
)


def _to_float_1d(values) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    return pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float)


def _match_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr
    if len(arr) > target_len:
        return arr[:target_len]
    out = np.full(target_len, np.nan, dtype=float)
    out[: len(arr)] = arr
    return out


def _pick_var(ds: xr.Dataset, candidates: list[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.variables:
            return name
    return None


def _extract_flight_date_from_filename(filepath: Path) -> Optional[datetime]:
    match = ICE_L_FILE_RE.search(filepath.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)


def _extract_time(ds: xr.Dataset, filepath: Path) -> pd.DatetimeIndex:
    time_name = _pick_var(ds, ["Time", "time", "TIME"])
    if time_name is None:
        return pd.to_datetime(pd.Series([], dtype="datetime64[ns]"), utc=True)

    raw = np.asarray(ds[time_name].values).reshape(-1)

    # Prefer direct datetime decoding when available.
    try:
        times = pd.to_datetime(raw, utc=True, errors="coerce")
        if pd.notna(times).sum() > 0:
            return pd.DatetimeIndex(times)
    except Exception:
        pass

    # Fallback: numeric seconds from flight day inferred from filename.
    raw_num = pd.to_numeric(raw, errors="coerce")
    flight_date = _extract_flight_date_from_filename(filepath)
    if flight_date is not None and np.isfinite(raw_num).any():
        base = pd.Timestamp(flight_date)
        return pd.to_datetime(base + pd.to_timedelta(raw_num, unit="s"), utc=True, errors="coerce")

    return pd.to_datetime(pd.Series(raw), utc=True, errors="coerce")


def load_ice_l_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a single ICE-L netCDF file and derive Si from RHUM."""
    filepath = Path(filepath)

    ds = xr.open_dataset(filepath, decode_times=True, decode_timedelta=True)
    try:
        temp_var = _pick_var(ds, ["ATX"])
        pres_var = _pick_var(ds, ["PSXC"])
        rh_var = _pick_var(ds, ["RHUM"])

        if temp_var is None or pres_var is None or rh_var is None:
            missing = [
                name
                for name, var in (("ATX", temp_var), ("PSXC", pres_var), ("RHUM", rh_var))
                if var is None
            ]
            raise KeyError(f"Missing required ICE-L variable(s) in {filepath.name}: {missing}")

        times = _extract_time(ds, filepath)
        n = len(times)

        tair_c = _match_length(_to_float_1d(ds[temp_var].values), n)
        psxc_hpa = _match_length(_to_float_1d(ds[pres_var].values), n)
        rhum = _match_length(_to_float_1d(ds[rh_var].values), n)

        # Unit normalization: ATX is expected in Celsius for ICE-L.
        med_t = np.nanmedian(tair_c)
        if np.isfinite(med_t) and med_t > 150:
            tair_c = tair_c - 273.15

        lat_name = _pick_var(ds, ["LAT", "LATC", "GGLAT"])
        lon_name = _pick_var(ds, ["LON", "LONC", "GGLON"])
        alt_name = _pick_var(ds, ["PALTF", "GGALT", "ALT", "PALT"])

        lat = _match_length(_to_float_1d(ds[lat_name].values), n) if lat_name else np.full(n, np.nan)
        lon = _match_length(_to_float_1d(ds[lon_name].values), n) if lon_name else np.full(n, np.nan)
        alt_m = _match_length(_to_float_1d(ds[alt_name].values), n) if alt_name else np.full(n, np.nan)

        # Physical plausibility checks.
        tair_c[(tair_c < -95) | (tair_c > 60)] = np.nan
        psxc_hpa[(psxc_hpa < 50) | (psxc_hpa > 1100)] = np.nan
        rhum[(rhum < -20) | (rhum > 200)] = np.nan
        lat[(lat < -90) | (lat > 90)] = np.nan
        lon[(lon < -180) | (lon > 180)] = np.nan
        alt_m[(alt_m < -500) | (alt_m > 25000)] = np.nan

        si = si_from_rh(rhum)
        si[(si < -1) | (si > 5)] = np.nan

        df = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(times, utc=True, errors="coerce"),
                "ATX_C": tair_c,
                "PSXC_hPa": psxc_hpa,
                "RHUM": rhum,
                "Si": si,
                "Lat": lat,
                "Lon": lon,
                "Alt_m": alt_m,
                "source_file": filepath.name,
                "Campaign": "ICE-L",
            }
        )

        if "Timestamp" in df.columns:
            df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)

        return df
    finally:
        ds.close()


def load_ice_l(data_dir: Union[str, Path], pattern: str = "*.PNI.nc") -> pd.DataFrame:
    """Load all ICE-L files and return a combined DataFrame."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")

    frames = []
    for filepath in files:
        try:
            frames.append(load_ice_l_file(filepath))
        except Exception as exc:
            print(f"Warning: Could not load {filepath.name}: {exc}")

    if not frames:
        raise ValueError(f"No valid ICE-L files parsed from {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Campaign"] = "ICE-L"
    return combined


def extract_ice_l_standard(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standardized columns for combined campaign output."""
    return pd.DataFrame(
        {
            "Timestamp": df.get("Timestamp", pd.NaT),
            "Tair_C": df.get("ATX_C", np.nan),
            "Si": df.get("Si", np.nan),
            "Lat": df.get("Lat", np.nan),
            "Lon": df.get("Lon", np.nan),
            "Alt_m": df.get("Alt_m", np.nan),
            "Campaign": df.get("Campaign", "ICE-L"),
            "source_file": df.get("source_file", ""),
        }
    )
