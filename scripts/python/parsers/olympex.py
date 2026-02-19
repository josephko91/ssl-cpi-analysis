"""
OLYMPEX (Olympic Mountains Experiment) campaign data parser.

Campaign: OLYMPEX Citation aircraft
Data Source: https://www.earthdata.nasa.gov/data/catalog/ghrc-daac-gpmcmolyx-1
Data Format: UND Citation data files with predefined column structure

Note: OLYMPEX water vapor measurements may have instrument issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Union

from .utils import (
    extract_takeoff_date,
    si_from_frost_point,
    COMMON_NA_VALUES,
)


# Predefined OLYMPEX column names
OLYMPEX_COLUMNS = [
    "Time", "Air_Temp", "MachNo_N", "IAS", "TAS", "Press_Alt", "Pot_Temp_T1",
    "STATIC_PR", "DEWPT", "REL_HUM", "MixingRatio", "DewPoint", "FrostPoint",
    "RH", "IceMSOFreq", "TSG_Date", "POS_Roll", "POS_Pitch", "POS_Head",
    "POSZ_Acc", "POS_Lat", "POS_Lon", "POS_Alt", "POS_Spd", "POS_Trk",
    "Alpha", "Beta", "VERT_VEL", "Wind_Z", "Wind_M", "Wind_D", "TURB",
    "King_LWC_ad", "Nev_TWC", "Nev_LWCcor", "Nev_IWC", "CSI_M_Ratio",
    "CSI_CWC", "CDP_Conc", "CDP_LWC", "CDP_MenD", "CDP_VolDia", "CDP_EffRad",
    "2-DC_Conc", "2-DC_MenD", "2-DC_VolDia", "2-DC_EffRad", "Nt2DSHGT105",
    "Nt2DSH_all", "Nt2DSVGT105", "Nt2DSV_all", "Nt_HVPS3H", "Nt_HVPS3BV",
]


def load_olympex_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single OLYMPEX data file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the OLYMPEX data file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with computed Si.
    """
    filepath = Path(filepath)
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    n_header = int(lines[0].split()[0])
    takeoff_date = extract_takeoff_date(lines[:n_header])
    
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        skiprows=n_header,
        names=OLYMPEX_COLUMNS,
        na_values=COMMON_NA_VALUES,
    )
    
    df["source_file"] = filepath.name
    
    # Parse UTC timestamp
    if "Time" in df.columns:
        df["Timestamp"] = df["Time"].apply(
            lambda x: takeoff_date + timedelta(seconds=float(x)) if pd.notnull(x) else pd.NaT
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
    
    # Calculate Si from frost point
    if "FrostPoint" in df.columns and "Air_Temp" in df.columns:
        df["Si"] = si_from_frost_point(df["FrostPoint"], df["Air_Temp"])
    
    return df


def load_olympex(
    data_dir: Union[str, Path],
    pattern: str = "*"
) -> pd.DataFrame:
    """
    Load all OLYMPEX files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing OLYMPEX data files.
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
            dfs.append(load_olympex_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "OLYMPEX"
    
    return combined


def extract_olympex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from OLYMPEX data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_olympex.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    return pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Tair_C": df.get("Air_Temp", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get("POS_Lat", np.nan),
        "Lon": df.get("POS_Lon", np.nan),
        "Alt_m": df.get("POS_Alt", np.nan),
        "Campaign": df.get("Campaign", "OLYMPEX"),
        "source_file": df["source_file"],
    })
