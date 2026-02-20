#!/usr/bin/env python3
"""
Main script for processing field campaign environmental and positional data.

This script orchestrates the loading and processing of data from multiple
atmospheric field campaigns, extracting standardized environmental (Si, temperature)
and positional (lat, lon, altitude) measurements.

Usage:
    python main.py --campaigns ARM MC3E --output combined_env_data.parquet
    python main.py --all --output combined_env_data.parquet
    python main.py --config config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parsers import CAMPAIGN_LOADERS
from parsers.arm import extract_arm_standard
from parsers.crystal_face_nasa import extract_crystal_face_nasa_standard
from parsers.crystal_face_und import extract_crystal_face_und_standard
from parsers.mc3e import extract_mc3e_standard
from parsers.midcix import extract_midcix_standard
from parsers.olympex import extract_olympex_standard
from parsers.airs_ii import extract_airs_ii_standard
from parsers.attrex import extract_attrex_standard
from parsers.iphex import extract_iphex_standard
from parsers.isdac import extract_isdac_standard
from parsers.posidon import extract_posidon_standard
from parsers.escape import extract_escape_standard
from parsers.ice_l import extract_ice_l_standard
from parsers.macpex import extract_macpex_standard


# =============================================================================
# Campaign Configuration
# =============================================================================

# Default data directories (update these paths for your system)
DEFAULT_CAMPAIGN_CONFIG: Dict[str, Dict] = {
    "ARM": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/ARM",
        "pattern": "*.t4archive.gz",
        "loader": "load_arm",
        "extractor": extract_arm_standard,
    },
    "CRYSTAL-FACE-NASA": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/CRYSTAL-FACE-NASA",
        "pattern": "*",
        "loader": "load_crystal_face_nasa",
        "extractor": extract_crystal_face_nasa_standard,
    },
    "CRYSTAL-FACE-UND": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/CRYSTAL-FACE-UND",
        "pattern": "*MIS.CIT",
        "loader": "load_crystal_face_und",
        "extractor": extract_crystal_face_und_standard,
    },
    "MC3E": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/MC3E",
        "pattern": "*",
        "loader": "load_mc3e",
        "extractor": extract_mc3e_standard,
    },
    "MIDCIX": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/MidCix",
        "pattern": "*",
        "loader": "load_midcix",
        "extractor": extract_midcix_standard,
    },
    "OLYMPEX": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/OLYMPEX",
        "pattern": "*",
        "loader": "load_olympex",
        "extractor": extract_olympex_standard,
    },
    "AIRS-II": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/AIRS-II",
        "pattern": "*.nc",
        "loader": "load_airs_ii",
        "extractor": extract_airs_ii_standard,
    },
    "ATTREX": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/ATTREX",
        "pattern": "*.ict",
        "loader": "load_attrex",
        "extractor": extract_attrex_standard,
    },
    "IPHEX": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/IPHEX",
        "pattern": "*.iphex",
        "loader": "load_iphex",
        "extractor": extract_iphex_standard,
    },
    "ISDAC": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/ISDAC/strapp-convair_bulk/CommaDelimited",
        "pattern": "*.txt",
        "loader": "load_isdac",
        "extractor": extract_isdac_standard,
    },
    "POSIDON": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/POSIDON",
        "pattern": "*.ict",
        "loader": "load_posidon",
        "extractor": extract_posidon_standard,
    },
    "ESCAPE": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/ESCAPE/learjet-state-measurements/jk4731582465",
        "pattern": "ESCAPE-Page0_Learjet_*.ict",
        "loader": "load_escape",
        "extractor": extract_escape_standard,
    },
    "ICE-L": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/ICE-L",
        "pattern": "*.PNI.nc",
        "loader": "load_ice_l",
        "extractor": extract_ice_l_standard,
    },
    "MACPEX": {
        "path": "/home/jko/ssl-cpi-analysis/data/env/MACPEX",
        "pattern": "*.ict",
        "loader": "load_macpex",
        "extractor": extract_macpex_standard,
    },
}


# =============================================================================
# Processing Functions
# =============================================================================

def load_campaign_config(config_path: Path) -> Dict:
    """Load campaign configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_campaign(
    campaign_name: str,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process a single campaign and return standardized DataFrame.
    
    Parameters
    ----------
    campaign_name : str
        Name of the campaign (e.g., "ARM", "MC3E").
    config : dict, optional
        Campaign configuration. If None, uses DEFAULT_CAMPAIGN_CONFIG.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    pd.DataFrame
        Standardized environmental and positional data.
    """
    if config is None:
        config = DEFAULT_CAMPAIGN_CONFIG.get(campaign_name)
    
    if config is None:
        raise ValueError(f"Unknown campaign: {campaign_name}")
    
    data_dir = Path(config["path"])
    pattern = config.get("pattern", "*")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    if verbose:
        print(f"Processing {campaign_name}...")
        print(f"  Data path: {data_dir}")
        print(f"  Pattern: {pattern}")
    
    # Get loader function
    loader = CAMPAIGN_LOADERS.get(campaign_name)
    if loader is None:
        raise ValueError(f"No loader found for campaign: {campaign_name}")
    
    # Load raw data
    df_raw = loader(data_dir, pattern)
    
    if verbose:
        print(f"  Loaded {len(df_raw):,} records")
    
    # Extract standardized columns
    extractor = config.get("extractor")
    if extractor:
        df_std = extractor(df_raw)
    else:
        # Basic extraction if no custom extractor
        df_std = pd.DataFrame({
            "Timestamp": df_raw.get("Timestamp", pd.NaT),
            "Tair_C": df_raw.get("Air_Temp", df_raw.get("T_C", pd.NA)),
            "Si": df_raw.get("Si", pd.NA),
            "Lat": pd.NA,
            "Lon": pd.NA,
            "Alt_m": pd.NA,
            "Campaign": campaign_name,
            "source_file": df_raw.get("source_file", ""),
        })
    
    # Ensure Campaign column
    df_std["Campaign"] = campaign_name
    
    if verbose:
        n_valid = df_std[["Timestamp", "Tair_C", "Si"]].notna().all(axis=1).sum()
        print(f"  Valid records (with Timestamp, Tair_C, Si): {n_valid:,}")
    
    return df_std


def process_all_campaigns(
    campaigns: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process multiple campaigns and return combined DataFrame.
    
    Parameters
    ----------
    campaigns : list of str, optional
        List of campaign names to process. If None, processes all campaigns.
    config : dict, optional
        Campaign configurations. If None, uses DEFAULT_CAMPAIGN_CONFIG.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    pd.DataFrame
        Combined standardized data from all campaigns.
    """
    if config is None:
        config = DEFAULT_CAMPAIGN_CONFIG
    
    if campaigns is None:
        campaigns = list(config.keys())
    
    all_dfs = []
    
    for campaign in campaigns:
        try:
            campaign_config = config.get(campaign, DEFAULT_CAMPAIGN_CONFIG.get(campaign))
            df = process_campaign(campaign, campaign_config, verbose)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {campaign}: {e}")
            continue
    
    if not all_dfs:
        raise RuntimeError("No campaigns were successfully processed")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if verbose:
        print(f"\nCombined dataset: {len(combined):,} total records")
        print(f"Campaigns: {combined['Campaign'].unique().tolist()}")
    
    return combined


def save_output(
    df: pd.DataFrame,
    output_path: Path,
    verbose: bool = True,
) -> None:
    """Save DataFrame to file (supports parquet and CSV)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    if verbose:
        print(f"Saved to: {output_path}")


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for the combined dataset."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    # Per-campaign summary
    summary = df.groupby("Campaign").agg(
        n_records=("Timestamp", "count"),
        earliest=("Timestamp", "min"),
        latest=("Timestamp", "max"),
        tair_mean=("Tair_C", "mean"),
        tair_std=("Tair_C", "std"),
        si_mean=("Si", "mean"),
        si_std=("Si", "std"),
    )
    
    print("\nPer-campaign statistics:")
    print(summary.to_string())
    
    # Missing data summary
    print("\nMissing data per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process field campaign environmental data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific campaigns
    python main.py --campaigns ARM MC3E --output data/combined.parquet
    
    # Process all campaigns
    python main.py --all --output data/combined.parquet
    
    # Use custom config file
    python main.py --config campaigns.yaml --output data/combined.parquet
    
    # Dry run (no output file)
    python main.py --campaigns ARM --dry-run
        """,
    )
    
    parser.add_argument(
        "--campaigns",
        nargs="+",
        choices=list(DEFAULT_CAMPAIGN_CONFIG.keys()),
        help="Campaign(s) to process",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available campaigns",
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("combined_env_data.parquet"),
        help="Output file path (parquet or csv)",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data without saving",
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet
    
    # Determine campaigns to process
    if args.all:
        campaigns = None  # Will process all
    elif args.campaigns:
        campaigns = args.campaigns
    else:
        print("Error: Specify --campaigns or --all")
        sys.exit(1)
    
    # Load custom config if provided
    config = None
    if args.config:
        config = load_campaign_config(args.config)
    
    # Process campaigns
    df = process_all_campaigns(campaigns, config, verbose)
    
    # Print summary
    if verbose:
        print_summary(df)
    
    # Save output
    if not args.dry_run:
        save_output(df, args.output, verbose)
    
    return df


if __name__ == "__main__":
    main()
