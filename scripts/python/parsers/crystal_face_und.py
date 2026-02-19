"""
CRYSTAL-FACE UND (Citation aircraft) campaign data parser.

Campaign: CRYSTAL-FACE UND Citation aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/crystalf/Citation
Data Format: ND* files (MIS.CIT humidity data, MET.CIT meteorology data)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    clean_column_name,
    extract_takeoff_date,
    si_from_rh,
    COMMON_NA_VALUES,
)


def _read_mis_cit_file(filepath: Path) -> pd.DataFrame:
    """Read a MIS.CIT humidity file."""
    with open(filepath, "r") as f:
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
    
    return df


def _read_met_cit_file(filepath: Path) -> pd.DataFrame:
    """Read a MET.CIT meteorology file."""
    with open(filepath, "r") as f:
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
    
    return df


def load_crystal_face_und_file(filepath_mis: Union[str, Path]) -> pd.DataFrame:
    """
    Load CRYSTAL-FACE UND data from MIS.CIT file and merge with MET.CIT.
    
    Parameters
    ----------
    filepath_mis : str or Path
        Path to the MIS.CIT humidity file.
        
    Returns
    -------
    pd.DataFrame
        Merged humidity and meteorology data with computed Si.
    """
    filepath_mis = Path(filepath_mis)
    
    # Read MIS.CIT (humidity data)
    df_mis = _read_mis_cit_file(filepath_mis)
    
    # Find corresponding MET.CIT file
    met_path = Path(str(filepath_mis).replace("MIS.CIT", "MET.CIT"))
    
    if not met_path.exists():
        df_mis["Tair"] = np.nan
        if "RH" in df_mis.columns:
            df_mis["Si"] = si_from_rh(df_mis["RH"])
        return df_mis
    
    # Read MET.CIT (meteorology data)
    df_met = _read_met_cit_file(met_path)
    
    # Find air temperature column
    air_temp_col = next((c for c in df_met.columns if c.startswith("Air_Temp")), None)
    if air_temp_col is None:
        df_mis["Tair"] = np.nan
    else:
        # Merge on Timestamp
        df_mis = pd.merge(
            df_mis,
            df_met[["Timestamp", air_temp_col]],
            on="Timestamp",
            how="left",
        )
        df_mis["Tair"] = df_mis[air_temp_col]
    
    # Calculate Si from RH
    if "RH" in df_mis.columns:
        df_mis["Si"] = si_from_rh(df_mis["RH"])
    
    return df_mis


def load_crystal_face_und(
    data_dir: Union[str, Path],
    pattern: str = "*MIS.CIT"
) -> pd.DataFrame:
    """
    Load all CRYSTAL-FACE UND files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing *MIS.CIT files.
    pattern : str, optional
        Glob pattern for MIS.CIT files (default: "*MIS.CIT").
        
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
            dfs.append(load_crystal_face_und_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "CRYSTAL-FACE-UND"
    
    return combined


def extract_crystal_face_und_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from CRYSTAL-FACE UND data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_crystal_face_und.
        
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
        "Tair_C": df.get("Tair", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "CRYSTAL-FACE-UND"),
        "source_file": df["source_file"],
    })
