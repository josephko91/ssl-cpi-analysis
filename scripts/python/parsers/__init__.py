"""
Campaign data parsers for environmental and positional data.

This package provides modular parsers for reading in situ atmospheric
measurements from various field campaigns.
"""

from .utils import (
    es_ice,
    si_from_frost_point,
    si_from_ppmv,
    si_from_rh,
    clean_column_name,
    parse_columns_with_units,
    extract_takeoff_date,
)

from .arm import load_arm
from .crystal_face_nasa import load_crystal_face_nasa
from .crystal_face_und import load_crystal_face_und
from .mc3e import load_mc3e
from .midcix import load_midcix
from .olympex import load_olympex
from .airs_ii import load_airs_ii
from .attrex import load_attrex

# Campaign registry for easy iteration
CAMPAIGN_LOADERS = {
    "ARM": load_arm,
    "CRYSTAL-FACE-NASA": load_crystal_face_nasa,
    "CRYSTAL-FACE-UND": load_crystal_face_und,
    "MC3E": load_mc3e,
    "MIDCIX": load_midcix,
    "OLYMPEX": load_olympex,
    "AIRS-II": load_airs_ii,
    "ATTREX": load_attrex,
}

__all__ = [
    # Utility functions
    "es_ice",
    "si_from_frost_point",
    "si_from_ppmv",
    "si_from_rh",
    "clean_column_name",
    "parse_columns_with_units",
    "extract_takeoff_date",
    # Campaign loaders
    "load_arm",
    "load_crystal_face_nasa",
    "load_crystal_face_und",
    "load_mc3e",
    "load_midcix",
    "load_olympex",
    "load_airs_ii",
    "load_attrex",
    "CAMPAIGN_LOADERS",
]
