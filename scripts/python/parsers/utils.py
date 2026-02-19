"""
Shared utility functions for campaign data parsing.

Provides thermodynamic calculations, column name cleaning, and
common parsing helpers used across multiple campaign parsers.
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Optional


# =============================================================================
# Thermodynamic Functions
# =============================================================================

def es_ice(T_C: np.ndarray) -> np.ndarray:
    """
    Calculate saturation vapor pressure over ice using Murphy & Koop (2005).
    
    Parameters
    ----------
    T_C : array-like
        Temperature in degrees Celsius.
        
    Returns
    -------
    np.ndarray
        Saturation vapor pressure over ice in hPa.
    """
    T_K = np.asarray(T_C) + 273.15
    return np.exp(9.550426 - 5723.265 / T_K + 3.53068 * np.log(T_K) - 0.007283 * T_K)


def si_from_frost_point(frost_point_C: np.ndarray, temperature_C: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation (Si) from frost point and ambient temperature.
    
    Parameters
    ----------
    frost_point_C : array-like
        Frost point temperature in degrees Celsius.
    temperature_C : array-like
        Ambient air temperature in degrees Celsius.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si), where values > 0 indicate supersaturation.
    """
    return es_ice(frost_point_C) / es_ice(temperature_C) - 1.0


def si_from_ppmv(wv_ppmv: np.ndarray, temp_K: np.ndarray, pressure_hPa: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation from water vapor mixing ratio (ppmv).
    
    Parameters
    ----------
    wv_ppmv : array-like
        Water vapor mixing ratio in parts per million by volume.
    temp_K : array-like
        Temperature in Kelvin.
    pressure_hPa : array-like
        Pressure in hPa.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si).
    """
    wv_ppmv = np.asarray(wv_ppmv, dtype=float)
    temp_K = np.asarray(temp_K, dtype=float)
    pressure_hPa = np.asarray(pressure_hPa, dtype=float)

    # Mask physically invalid inputs so they don't produce extreme Si
    invalid = (
        ~np.isfinite(wv_ppmv) | (wv_ppmv <= 0)
        | ~np.isfinite(temp_K) | (temp_K < 150) | (temp_K > 350)
        | ~np.isfinite(pressure_hPa) | (pressure_hPa <= 0)
    )

    # Calculate saturation vapor pressure over ice in hPa
    e_s = 6.112 * np.exp((22.46 * (temp_K - 273.15)) / (temp_K - 0.55))

    # Convert vapor mixing ratio (ppmv) to actual vapor pressure (e) in hPa
    e = (wv_ppmv / 1e6) * pressure_hPa

    result = (e / e_s) - 1.0

    # Replace invalid entries with NaN
    if np.ndim(result) == 0:
        return np.nan if invalid else float(result)
    result[invalid] = np.nan
    return result


def si_from_rh(rh_percent: np.ndarray) -> np.ndarray:
    """
    Compute ice supersaturation from relative humidity with respect to ice.
    
    Parameters
    ----------
    rh_percent : array-like
        Relative humidity with respect to ice in percent.
        
    Returns
    -------
    np.ndarray
        Ice supersaturation (Si).
    """
    return np.asarray(rh_percent) / 100.0 - 1.0


# =============================================================================
# Column Name Utilities
# =============================================================================

def clean_column_name(name: str) -> str:
    """
    Clean a column name for safe DataFrame usage.
    
    Strips whitespace and replaces non-word characters with underscores.
    
    Parameters
    ----------
    name : str
        Original column name.
        
    Returns
    -------
    str
        Cleaned column name.
    """
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    return name.strip("_")


def parse_columns_with_units(header_line: str) -> List[str]:
    """
    Parse column names that include units in parentheses.
    
    Handles formats like: "UT (s) T (K) RH (%)" -> ["UT_s", "T_K", "RH"]
    
    Parameters
    ----------
    header_line : str
        Header line containing column names with optional units.
        
    Returns
    -------
    list of str
        List of cleaned column names.
    """
    tokens = header_line.split()
    cols = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and re.match(r"^\(.*\)$", tokens[i + 1]):
            cols.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
        else:
            cols.append(tokens[i])
            i += 1
    return [clean_column_name(c) for c in cols]


# =============================================================================
# Date Extraction Utilities
# =============================================================================

def extract_takeoff_date(lines: List[str]) -> datetime:
    """
    Extract the first date (YYYY MM DD) from header lines.
    
    Common format in NASA ICARTT and ESPO archive files.
    
    Parameters
    ----------
    lines : list of str
        Header lines to search.
        
    Returns
    -------
    datetime
        Extracted date.
        
    Raises
    ------
    ValueError
        If no valid date is found in the header.
    """
    for line in lines:
        match = re.match(r"\s*(\d{4})\s+(\d{2})\s+(\d{2})", line)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)
    raise ValueError("Takeoff date not found in header.")


# =============================================================================
# Common NA Values
# =============================================================================

# Standard missing value flags used across campaigns
COMMON_NA_VALUES = [
    999999.9999,
    999.9999999,
    9999.999999,
    99999.99999,
    99999999999,
    999999,
    9.9999E+30,
    9.999E+30,
    -9999,
    -9999.99,
    -7777,
    -7777.77,
    -8888,
    -8888.88,
]
