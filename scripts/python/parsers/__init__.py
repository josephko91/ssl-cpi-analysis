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
from .escape import load_escape
from .ice_l import load_ice_l
from .iphex import load_iphex
from .isdac import load_isdac
from .macpex import load_macpex, extract_macpex_standard
from .posidon import load_posidon

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
    "IPHEX": load_iphex,
    "ISDAC": load_isdac,
    "MACPEX": load_macpex,
    "POSIDON": load_posidon,
    "ESCAPE": load_escape,
    "ICE-L": load_ice_l,
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
    "load_iphex",
    "load_isdac",
    "load_macpex",
    "extract_macpex_standard",
    "load_posidon",
    "load_escape",
    "load_ice_l",
    "CAMPAIGN_LOADERS",
]
