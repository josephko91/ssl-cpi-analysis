"""
MC3E (Midlatitude Continental Convective Clouds Experiment) campaign data parser.

Campaign: MC3E
Data Source: https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-gpmcmmc3e-1
Data Format: NASA ICARTT-style text files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    clean_column_name,
    extract_takeoff_date,
    si_from_frost_point,
    COMMON_NA_VALUES,
)


def load_mc3e_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single MC3E data file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the MC3E data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with computed Si.
    """
    filepath = Path(filepath)
    
    with open(filepath) as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    columns = [clean_column_name(c) for c in lines[n_header - 2].split()]
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name
    
    # Parse UTC timestamp
    if "Time" in df.columns:
        df["Timestamp"] = df["Time"].apply(
            lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Calculate Si from frost point (preferred method for MC3E)
    if "FrostPoint" in df.columns and "Air_Temp" in df.columns:
        df["Si"] = si_from_frost_point(df["FrostPoint"], df["Air_Temp"])
    
    return df


def load_mc3e(
    data_dir: Union[str, Path],
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all MC3E files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing MC3E data files.
    pattern : str, optional
        Glob pattern for matching files (default: "*").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    files = [f for f in data_dir.glob(pattern) if f.is_file()]
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_mc3e_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "MC3E"
    
    return combined


def extract_mc3e_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from MC3E data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_mc3e.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower()), None)
    
    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("Air_Temp", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "MC3E"),
        "source_file": df["source_file"],
    })
