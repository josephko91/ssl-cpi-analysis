"""
AIRS-II (Alliance Icing Research Study II) campaign data parser.

Campaign: AIRS-II
Data Source: https://data.eol.ucar.edu/project/AIRS-II
Data Format: NetCDF files with LRT (Low-Rate) flight-level data
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Union


def load_airs_ii_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single AIRS-II netCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the netCDF file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with Timestamp, RHUM, ATX_C, Si, and position data.
    """
    filepath = Path(filepath)
    
    # Use CFDatetimeCoder for proper time decoding
    ds = xr.open_dataset(filepath, decode_times=True, decode_timedelta=True)
    
    try:
        if "RHUM" not in ds.variables or "ATX" not in ds.variables:
            raise KeyError("RHUM or ATX missing in dataset")
        
        # Extract time coordinate
        if "Time" in ds:
            times = pd.to_datetime(ds["Time"].values)
        elif "time" in ds.coords:
            times = pd.to_datetime(ds.coords["time"].values)
        else:
            raise KeyError("No time coordinate found")
        
        rh = np.asarray(ds["RHUM"].values).ravel().astype(float)
        atx = np.asarray(ds["ATX"].values).ravel().astype(float)  # Temperature in Celsius
        
        # Filter out unrealistic RH values (> 200%)
        valid_mask = rh <= 200
        rh = rh[valid_mask]
        atx = atx[valid_mask]
        times = np.asarray(times).ravel()[valid_mask]
        
        # Extract position data if available
        lat = np.asarray(ds.get("LAT", ds.get("LATC", [np.nan] * len(times)))).ravel()
        lon = np.asarray(ds.get("LON", ds.get("LONC", [np.nan] * len(times)))).ravel()
        alt = np.asarray(ds.get("ALT", ds.get("GGALT", [np.nan] * len(times)))).ravel()
        
        # Apply same mask to position data
        if len(lat) == len(valid_mask):
            lat = lat[valid_mask]
            lon = lon[valid_mask]
            alt = alt[valid_mask]
        
        n = min(len(times), len(rh), len(atx))
        
        df = pd.DataFrame({
            "Timestamp": pd.to_datetime(times[:n], utc=True),
            "RHUM": rh[:n],
            "ATX_C": atx[:n],
            "Lat": lat[:n] if len(lat) >= n else np.nan,
            "Lon": lon[:n] if len(lon) >= n else np.nan,
            "Alt_m": alt[:n] if len(alt) >= n else np.nan,
        })
        
        df = df.sort_values("Timestamp").reset_index(drop=True)
        
        # Si is RHUM/100 - 1 (since RHUM is already w.r.t. ice in some cases)
        # Note: Check metadata for actual RH type
        df["Si"] = df["RHUM"] / 100.0 - 1.0
        
        df["source_file"] = filepath.name
        
        return df
        
    finally:
        ds.close()


def load_airs_ii(
    data_dir: Union[str, Path],
    pattern: str = "*.nc"
) -> pd.DataFrame:
    """
    Load all AIRS-II netCDF files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing AIRS-II netCDF files.
    pattern : str, optional
        Glob pattern for matching files (default: "*.nc").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_airs_ii_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "AIRS-II"
    
    return combined


def extract_airs_ii_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from AIRS-II data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_airs_ii.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("ATX_C", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get("Lat", np.nan),
        "Lon": df.get("Lon", np.nan),
        "Alt_m": df.get("Alt_m", np.nan),
        "Campaign": df.get("Campaign", "AIRS-II"),
        "source_file": df["source_file"],
    })
