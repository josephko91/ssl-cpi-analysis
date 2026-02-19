"""
CRYSTAL-FACE NASA (WB-57) campaign data parser.

Campaign: CRYSTAL-FACE NASA WB-57 aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/crystalf/WB57
Data Format: NASA ICARTT-style text files with JPL Laser Hygrometer (JLH) data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Union

from .utils import (
    parse_columns_with_units,
    extract_takeoff_date,
    si_from_rh,
)


def load_crystal_face_nasa_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single CRYSTAL-FACE NASA JLH file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with environmental measurements and computed Si.
    """
    filepath = Path(filepath)
    
    with open(filepath) as f:
        lines = f.readlines()
    
    # First line contains number of header lines
    n_header = int(lines[0].split()[0])
    
    # Parse column names from last header line
    columns = parse_columns_with_units(lines[n_header - 1])
    
    # Extract takeoff date from header
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    # Read data
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=columns,
    )
    
    # Find UTC seconds column
    ut_col = next((c for c in df.columns if c.lower().startswith("ut")), None)
    if ut_col is None:
        raise ValueError(f"No UT seconds column found in {filepath.name}")
    
    # Create timestamp
    df["Timestamp"] = df[ut_col].apply(
        lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Calculate Si from RH (relative humidity w.r.t. ice)
    # JLH files typically have RH or %RH column
    rh_col = next((c for c in df.columns if "rh" in c.lower()), None)
    if rh_col:
        df["Si"] = si_from_rh(df[rh_col])
    
    # Convert temperature from Kelvin to Celsius if present
    temp_col = next((c for c in df.columns if c.lower() in ["t_k", "t"]), None)
    if temp_col:
        df["T_C"] = df[temp_col] - 273.15
    
    df["source_file"] = filepath.name
    
    return df


def load_crystal_face_nasa(
    data_dir: Union[str, Path], 
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all CRYSTAL-FACE NASA files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing JLH data files.
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
            dfs.append(load_crystal_face_nasa_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "CRYSTAL-FACE-NASA"
    
    return combined


def extract_crystal_face_nasa_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from CRYSTAL-FACE NASA data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_crystal_face_nasa.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns (may vary by file format)
    lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
    alt_col = next((c for c in df.columns if "alt" in c.lower() or "z" in c.lower()), None)
    
    result = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("T_C", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "CRYSTAL-FACE-NASA"),
        "source_file": df["source_file"],
    })
    
    return result
