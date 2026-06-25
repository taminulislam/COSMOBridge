"""Compute surface descriptors for ILThermo compounds.

Same descriptors as step4b_surface_descriptors.py but for the 115+ ILThermo ILs.
Uses SMILES as the key since il_short_name is a long string in ILThermo data.

Input:  data/augmented/ilthermo_data.csv
Output: data/pipeline/surface_descriptors_ilthermo.csv
"""

import csv
import sys
import numpy as np
from pathlib import Path

# Reuse all functions from the original script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4b_surface_descriptors import compute_descriptors


def main():
    import pandas as pd

    base_dir = Path(__file__).resolve().parent.parent.parent

    # Load existing descriptors to avoid recomputing
    existing_path = base_dir / "data" / "pipeline" / "surface_descriptors.csv"
    existing_smiles = set()
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        # We need to map il_short_name back to smiles from the original data
        raw_csv = base_dir / "data" / "processed" / "il_data_raw.csv"
        if raw_csv.exists():
            raw_df = pd.read_csv(raw_csv)
            name_to_smiles = dict(zip(raw_df["il_short_name"], raw_df["smiles"]))
            for name in existing["il_short_name"]:
                if name in name_to_smiles:
                    existing_smiles.add(name_to_smiles[name])
        print(f"Existing descriptors: {len(existing)} ILs ({len(existing_smiles)} SMILES)")

    # Load ILThermo unique SMILES
    ilthermo_csv = base_dir / "data" / "augmented" / "ilthermo_data.csv"
    if not ilthermo_csv.exists():
        print(f"ERROR: {ilthermo_csv} not found.")
        return

    df = pd.read_csv(ilthermo_csv)
    unique_ils = df.drop_duplicates(subset=["smiles"])[["smiles", "il_short_name"]].sort_values("smiles")
    # Filter out already-computed
    unique_ils = unique_ils[~unique_ils["smiles"].isin(existing_smiles)]
    print(f"ILThermo unique ILs to compute: {len(unique_ils)}\n")

    results = []
    failed = 0
    for i, (_, row) in enumerate(unique_ils.iterrows()):
        smiles = row["smiles"]
        il_name = row["il_short_name"][:30]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(unique_ils)}] {il_name}...")
        desc = compute_descriptors(smiles, il_name)
        if desc is not None:
            # Use smiles as key for ILThermo (il_short_name is not unique/clean)
            desc["smiles"] = smiles
            results.append(desc)
        else:
            failed += 1

    # Save
    output_path = base_dir / "data" / "pipeline" / "surface_descriptors_ilthermo.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"\nSaved {len(results)} descriptors to {output_path}")
    print(f"Failed: {failed}/{len(unique_ils)}")


if __name__ == "__main__":
    main()
