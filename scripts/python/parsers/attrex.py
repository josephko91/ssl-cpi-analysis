"""
ATTREX (Airborne Tropical TRopopause EXperiment) campaign data parser.

Campaign: ATTREX Global Hawk aircraft
Data Source: https://espoarchive.nasa.gov/archive/browse/attrex/id4/GHawk
Data Format: NASA ICARTT (.ict) files

Water vapor instruments: NOAA-H2O, DLH (Diode Laser Hygrometer)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict

from .utils import si_from_ppmv


# Time column names for different ATTREX instruments
ATTREX_TIME_COLS: Dict[str, str] = {
    "DLH-H2O": "Time_Start",
    "NOAA-H2O": "Time_Start",
    "MMS": "Time_Start",
}

# Missing value flags
ATTREX_MISSING_FLAGS = [-9999, -9999.99, -7777, -7777.77, -8888, -8888.88]


def _replace_invalid_values(value: str) -> float:
    """Replace invalid values during CSV parsing."""
    try:
        val = float(value)
        if val in ATTREX_MISSING_FLAGS:
            return np.nan
        return val
    except (ValueError, TypeError):
        return np.nan


def _parse_ict_file(filepath: Path, time_col: str = "Time_Start") -> pd.DataFrame:
    """
    Parse NASA ICARTT format (.ict) file.
    
    Parameters
    ----------
    filepath : Path
        Path to the .ict file.
    time_col : str
        Name of the time column to use.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with datetime_utc column.
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        n_header_lines = int(first_line.split(',')[0])
        
        f.seek(0)
        header_lines = [f.readline().strip() for _ in range(n_header_lines)]
    
    # Find flight date from header (YYYY, MM, DD format)
    flight_date = None
    for line in header_lines:
        parts = [p.strip() for p in line.split(',')]
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
    
    # Extract column names from last header line
    col_line = header_lines[-1]
    columns = [col.strip() for col in col_line.split(',') if col.strip()]
    
    # Read data
    df = pd.read_csv(
        filepath,
        skiprows=n_header_lines,
        names=columns,
        converters={col: _replace_invalid_values for col in columns},
        skipinitialspace=True,
        on_bad_lines='skip',
    )
    
    df.columns = df.columns.str.strip()
    
    # Create datetime_utc from time column
    if time_col in df.columns:
        df['datetime_utc'] = flight_date + pd.to_timedelta(df[time_col], unit='s')
    else:
        # Try to find any time-like column
        time_candidates = [c for c in df.columns if 'time' in c.lower()]
        if time_candidates:
            df['datetime_utc'] = flight_date + pd.to_timedelta(df[time_candidates[0]], unit='s')
    
    # Handle missing data flags
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(ATTREX_MISSING_FLAGS, np.nan)
    
    return df


def load_attrex_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single ATTREX ICT file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .ict file.
        
    Returns
    -------
    pd.DataFrame
        Parsed data with computed Si.
    """
    filepath = Path(filepath)
    
    # Determine instrument from parent directory or filename
    instrument = filepath.parent.name
    time_col = ATTREX_TIME_COLS.get(instrument, "Time_Start")
    
    df = _parse_ict_file(filepath, time_col)
    
    # Rename datetime column
    if 'datetime_utc' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['datetime_utc'], utc=True)
    
    # Calculate Si for DLH-H2O
    h2o_col = next((c for c in df.columns if 'H2O' in c and 'ppm' in c.lower()), None)
    temp_col = next((c for c in df.columns if c == 'T' or c.startswith('T_')), None)
    pres_col = next((c for c in df.columns if c == 'P' or c.startswith('P_')), None)
    
    if h2o_col and temp_col and pres_col:
        df["Si"] = si_from_ppmv(df[h2o_col], df[temp_col], df[pres_col])
    
    # Temperature in Celsius
    if temp_col and df[temp_col].median() > 200:  # Likely in Kelvin
        df["T_C"] = df[temp_col] - 273.15
    elif temp_col:
        df["T_C"] = df[temp_col]
    
    df["source_file"] = filepath.name
    
    return df


def load_attrex(
    data_dir: Union[str, Path],
    pattern: str = "*.ict"
) -> pd.DataFrame:
    """
    Load all ATTREX ICT files from a directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing ATTREX .ict files.
    pattern : str, optional
        Glob pattern for matching files (default: "*.ict").
        
    Returns
    -------
    pd.DataFrame
        Combined data from all files.
    """
    data_dir = Path(data_dir)
    
    # Search recursively for .ict files
    files = list(data_dir.rglob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {data_dir}")
    
    dfs = []
    for f in sorted(files):
        try:
            dfs.append(load_attrex_file(f))
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined["Campaign"] = "ATTREX"
    
    return combined


def extract_attrex_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract standardized columns from ATTREX data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data loaded by load_attrex.
        
    Returns
    -------
    pd.DataFrame
        Standardized data with Timestamp, Tair_C, Si, Lat, Lon, Alt_m, Campaign.
    """
    # Find position columns
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    alt_col = next((c for c in df.columns if 'alt' in c.lower()), None)
    
    return pd.DataFrame({
        "Timestamp": df.get("Timestamp", pd.NaT),
        "Tair_C": df.get("T_C", np.nan),
        "Si": df.get("Si", np.nan),
        "Lat": df.get(lat_col, np.nan) if lat_col else np.nan,
        "Lon": df.get(lon_col, np.nan) if lon_col else np.nan,
        "Alt_m": df.get(alt_col, np.nan) if alt_col else np.nan,
        "Campaign": df.get("Campaign", "ATTREX"),
        "source_file": df["source_file"],
    })
