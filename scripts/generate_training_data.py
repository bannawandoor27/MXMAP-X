#!/usr/bin/env python3
"""
Generate training data for the XGBoost model.

Reads the extracted corpus CSV, maps fields to the training schema,
augments with physics-informed synthetic samples, and outputs a clean
CSV ready for train_model.py.

Usage:
    python scripts/generate_training_data.py [--corpus data/corpus_dataset.csv]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "mxene_type",
    "terminations",
    "electrolyte",
    "electrolyte_concentration",
    "thickness_um",
    "deposition_method",
    "annealing_temp_c",
    "annealing_time_min",
    "interlayer_spacing_nm",
    "specific_surface_area_m2g",
    "pore_volume_cm3g",
    "optical_transmittance",
    "sheet_resistance_ohm_sq",
]

TARGET_COLS = [
    "areal_capacitance_mf_cm2",
    "esr_ohm",
    "rate_capability_percent",
    "cycle_life_cycles",
]

ALL_COLS = FEATURE_COLS + TARGET_COLS

# ── Default distributions from MXene literature ──────────────────────────────

MXENE_TYPES = ["Ti3C2Tx", "Mo2CTx", "V2CTx", "Ti2CTx", "Nb2CTx", "Ti3CNTx", "Ta4C3Tx"]
TERMINATIONS = ["O", "OH", "F", "mixed"]
ELECTROLYTES = ["H2SO4", "KOH", "NaOH", "ionic_liquid", "EMIMBF4", "PVA_H2SO4", "PVA_KOH", "organic"]
DEPOSITION_METHODS = ["vacuum_filtration", "spray_coating", "spin_coating", "drop_casting", "inkjet"]


def map_corpus_to_training(df: pd.DataFrame) -> pd.DataFrame:
    """Map extracted corpus columns to training schema."""
    out = pd.DataFrame()

    # Direct mappings
    out["mxene_type"] = df.get("mxene_type", pd.Series(dtype=str))
    out["electrolyte"] = df.get("electrolyte", pd.Series(dtype=str))
    out["areal_capacitance_mf_cm2"] = df.get("areal_capacitance_mf_cm2")
    out["specific_surface_area_m2g"] = df.get("ssa_m2_g")
    out["annealing_temp_c"] = df.get("annealing_temperature_c")

    # Map specific capacitance (F/g) → rough areal estimate if areal is missing
    if "specific_capacitance_f_g" in df.columns:
        mask = out["areal_capacitance_mf_cm2"].isna()
        # Rough conversion: areal ≈ specific * thickness(μm) * density(~3.5 g/cm³) * 0.1
        # → roughly specific * 1.5 mF/cm² per F/g (for ~4 μm films)
        out.loc[mask, "areal_capacitance_mf_cm2"] = df.loc[mask, "specific_capacitance_f_g"] * 1.5

    # Map cycling stability (%) → cycle_life estimate
    if "cycling_stability" in df.columns and "num_cycles" in df.columns:
        # cycling_stability = retention % after num_cycles
        # Extrapolate to 80% retention: cycles_80 ≈ num_cycles * ln(0.8) / ln(retention/100)
        retention = df["cycling_stability"].clip(50, 99.9) / 100.0
        out["cycle_life_cycles"] = (
            df["num_cycles"] * np.log(0.8) / np.log(retention)
        ).clip(1000, 100000)
    else:
        out["cycle_life_cycles"] = np.nan

    # Map scan rate / current density (metadata, not direct targets)
    out["scan_rate_mv_s"] = df.get("scan_rate_mv_s")
    out["current_density_a_g"] = df.get("current_density_a_g")

    # Fill synthesis_method → deposition_method
    out["deposition_method"] = df.get("synthesis_method", "vacuum_filtration")

    # Defaults for missing features
    out["terminations"] = "mixed"
    out["electrolyte_concentration"] = np.nan
    out["thickness_um"] = np.nan
    out["annealing_time_min"] = np.nan
    out["interlayer_spacing_nm"] = np.nan
    out["pore_volume_cm3g"] = np.nan
    out["optical_transmittance"] = np.nan
    out["sheet_resistance_ohm_sq"] = np.nan

    # ESR and rate capability are rarely extracted — leave NaN for imputation
    out["esr_ohm"] = np.nan
    out["rate_capability_percent"] = np.nan

    return out


def combine_and_clean(
    corpus_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Clean corpus data and impute missing values from the corpus itself."""
    if corpus_df is None or len(corpus_df) == 0:
        print("No corpus data provided.")
        return pd.DataFrame(columns=ALL_COLS)

    # Filter to rows that have at least (mxene_type + one target)
    has_target = corpus_df[TARGET_COLS].notna().any(axis=1)
    has_mxene = corpus_df["mxene_type"].notna()
    real_rows = corpus_df[has_target & has_mxene].copy()
    print(f"  Usable corpus rows: {len(real_rows)}/{len(corpus_df)}")

    # Impute missing targets from medians of the real corpus data
    for col in TARGET_COLS:
        median = real_rows[col].median()
        # Fallback if median is nan because field was extracted exactly 0 times globally
        if pd.isna(median):
            if col == "areal_capacitance_mf_cm2": median = 200.0
            elif col == "esr_ohm": median = 2.0
            elif col == "rate_capability_percent": median = 80.0
            elif col == "cycle_life_cycles": median = 5000.0
        real_rows[col] = real_rows[col].fillna(median)

    # Impute missing features
    for col in FEATURE_COLS:
        if col in real_rows.columns and real_rows[col].dtype in ("float64", "Float64"):
            median = real_rows[col].median()
            if pd.isna(median):
                median = 0.0 # dummy if no papers extracted this
            real_rows[col] = real_rows[col].fillna(median)

    combined = real_rows[ALL_COLS].copy()

    # Final cleanup: drop rows with missing required targets (mitigated by imputation above)
    combined = combined.dropna(subset=["areal_capacitance_mf_cm2", "esr_ohm",
                                        "rate_capability_percent", "cycle_life_cycles"])

    # Ensure correct types
    combined["cycle_life_cycles"] = combined["cycle_life_cycles"].astype(int)

    return combined


def main():
    parser = argparse.ArgumentParser(description="Generate training data for XGBoost model")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus_dataset_geminicli.csv"),
                        help="Path to extracted corpus CSV")
    parser.add_argument("--output", type=Path, default=Path("data/original_training_data.csv"))
    args = parser.parse_args()

    # 1. Load corpus data
    corpus_mapped = None
    if args.corpus.exists():
        print(f"\nLoading corpus data from {args.corpus}...")
        raw = pd.read_csv(args.corpus)
        print(f"  Raw corpus: {len(raw)} rows")
        corpus_mapped = map_corpus_to_training(raw)
    else:
        print(f"\n  ⚠ Corpus CSV not found at {args.corpus}")
        sys.exit(1)

    # 2. Clean and impute using purely corpus distributions
    print("\nCleaning data...")
    combined = combine_and_clean(corpus_mapped)

    # 4. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"TRAINING DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples : {len(combined)}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Targets       : {len(TARGET_COLS)}")
    print(f"\n  Target ranges:")
    for col in TARGET_COLS:
        vals = combined[col].dropna()
        print(f"    {col:35s}  mean={vals.mean():9.1f}  [{vals.min():.1f}, {vals.max():.1f}]")
    print(f"\n  MXene types: {combined['mxene_type'].value_counts().to_dict()}")
    print(f"\n  Output: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
