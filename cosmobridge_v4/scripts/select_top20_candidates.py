"""Select top-20 candidates for DFT surface calculation (I5 step 1).

Takes v3 screening results (v4 preserves γ₁/P rankings since Fusion path
dominates those) and extracts top-20 by desirability for DFT pipeline.

Output: cosmobridge_v4/screening/top20_candidates.csv
"""

import sys
import json
import csv
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    # Load v3 all-candidates file (has full rankings)
    all_cand_path = Path("results/virtual_screening/all_candidates.csv")
    df = pd.read_csv(all_cand_path)
    print(f"Loaded {len(df)} candidates")
    print(f"Columns: {df.columns.tolist()}")

    # Find desirability column
    desir_col = None
    for c in ['desirability', 'D_combined', 'D', 'score']:
        if c in df.columns:
            desir_col = c
            break
    if desir_col is None:
        print("ERROR: No desirability column found")
        sys.exit(1)
    print(f"Using column: {desir_col}")

    # Sort and take top 20 unique (cation, anion) pairs
    df_sorted = df.sort_values(desir_col, ascending=False)

    # Collapse to unique IL identifiers
    # Use either 'name' or combine cation+anion
    id_col = None
    for c in ['name', 'il_name', 'short_name', 'cation_anion', 'cation+anion']:
        if c in df.columns:
            id_col = c
            break

    if id_col is None:
        # Use SMILES as ID
        id_col = 'smiles' if 'smiles' in df.columns else 'full_smiles'

    top20_unique = df_sorted.drop_duplicates(subset=[id_col]).head(20)

    # Output directory
    out_dir = Path("cosmobridge_v4/screening")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "top20_candidates.csv"

    # Simple output: rank, id, smiles, desirability
    cols_to_keep = [id_col, desir_col]
    if 'smiles' in df.columns and 'smiles' != id_col:
        cols_to_keep.append('smiles')
    if 'full_smiles' in df.columns:
        cols_to_keep.append('full_smiles')
    for prop in ['gamma1', 'G_mix', 'P']:
        if prop in df.columns:
            cols_to_keep.append(prop)

    top20_unique[cols_to_keep].to_csv(out_path, index=False)
    print(f"\nSaved top 20 to: {out_path}")
    print(f"\nTop 20 candidates:")
    for i, (_, row) in enumerate(top20_unique.iterrows(), 1):
        print(f"  {i:2d}. {row[id_col]}")
        if desir_col in row:
            print(f"       D={row[desir_col]:.4f}")


if __name__ == "__main__":
    main()
