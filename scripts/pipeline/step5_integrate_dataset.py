"""Step 5: Integrate ILThermo data + generated COSMO images with existing dataset.

Fetches actual thermodynamic property data from ILThermo for compounds
that have generated COSMO images, merges with the existing 28-IL dataset,
and re-preprocesses for model training.

Input:  data/pipeline/ilthermo_compounds.csv
        data/pipeline/ilthermo_datasets.csv
        data/pipeline/cosmo_images/
        data/processed/ (existing dataset)
Output: data/augmented/ (merged dataset with new ILs + images)
"""

import csv
import json
import time
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path


BASE_URL = "https://ilthermo.boulder.nist.gov"
HEADERS = {"User-Agent": "ILResearch/1.0 (Academic; SIU)"}

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]


def fetch_json(url):
    """Fetch JSON from ILThermo."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("iso-8859-1"))
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return None


def fetch_dataset_data(setid):
    """Fetch and parse a single ILThermo dataset."""
    data = fetch_json(f"{BASE_URL}/ILT2/ilset?set={setid}")
    if not data:
        return None

    # Parse column headers
    columns = []
    for col in data.get("dhead", []):
        col_name = col[0] if isinstance(col, list) else str(col)
        columns.append(col_name)

    # Parse data rows
    rows = []
    for row in data.get("data", []):
        values = []
        for cell in row:
            if isinstance(cell, list) and len(cell) > 0:
                try:
                    values.append(float(cell[0]))
                except (ValueError, TypeError):
                    values.append(None)
            else:
                values.append(None)
        rows.append(values)

    if not rows or not columns:
        return None

    df = pd.DataFrame(rows, columns=columns[:len(rows[0])])

    # Extract metadata
    components = data.get("components", [])
    comp_names = [c.get("name", "") for c in components]
    ref = data.get("ref", {}).get("full", "")

    return {
        "df": df,
        "columns": columns,
        "components": comp_names,
        "reference": ref,
        "property": data.get("title", ""),
    }


def identify_property_columns(columns, property_title):
    """Map ILThermo column names to our target properties."""
    mapping = {}
    col_lower = {c: c.lower() for c in columns}

    for col, cl in col_lower.items():
        if "temperature" in cl:
            mapping["temperature"] = col
        elif "mole fraction" in cl:
            mapping["mole_fraction"] = col
        elif "pressure" in cl and "excess" not in cl and "vapor" not in cl:
            mapping["pressure"] = col
        elif "activity" in cl or "coefficient" in cl:
            mapping["activity_coefficient"] = col
        elif "excess enthalpy" in cl or "excess molar enthalpy" in cl:
            mapping["H_E"] = col
        elif "excess gibbs" in cl:
            mapping["G_E"] = col
        elif "vapor" in cl and "pressure" in cl:
            mapping["P"] = col
        elif "enthalpy" in cl and "vaporization" in cl:
            mapping["H_vap"] = col

    return mapping


def process_activity_dataset(dataset_info, il_compound):
    """Process an activity coefficient dataset into rows for our format."""
    df = dataset_info["df"]
    col_map = identify_property_columns(list(df.columns), dataset_info["property"])

    if "temperature" not in col_map or "activity_coefficient" not in col_map:
        return []

    rows = []
    for _, row in df.iterrows():
        temp = row.get(col_map["temperature"])
        x1 = row.get(col_map.get("mole_fraction", ""), 0.5)
        gamma = row.get(col_map["activity_coefficient"])

        if temp is None or gamma is None:
            continue
        if not (250 < temp < 600):  # reasonable T range
            continue

        rows.append({
            "temperature": temp,
            "x1": x1 if x1 is not None else 0.5,
            "gamma1": gamma,
            "source": "ilthermo",
            "reference": dataset_info["reference"][:100],
        })

    return rows


def process_excess_enthalpy_dataset(dataset_info, il_compound):
    """Process an excess enthalpy dataset."""
    df = dataset_info["df"]
    col_map = identify_property_columns(list(df.columns), dataset_info["property"])

    if "temperature" not in col_map or "H_E" not in col_map:
        return []

    rows = []
    for _, row in df.iterrows():
        temp = row.get(col_map["temperature"])
        he = row.get(col_map["H_E"])
        x1 = row.get(col_map.get("mole_fraction", ""), 0.5)

        if temp is None or he is None:
            continue
        if not (250 < temp < 600):
            continue

        rows.append({
            "temperature": temp,
            "x1": x1 if x1 is not None else 0.5,
            "H_E": he,
            "source": "ilthermo",
            "reference": dataset_info["reference"][:100],
        })

    return rows


def main():
    pipeline_dir = Path("data/pipeline")
    output_dir = Path("data/augmented")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing dataset
    existing_raw = Path("data/processed/il_data_raw.csv")
    if existing_raw.exists():
        df_existing = pd.read_csv(existing_raw)
        print(f"Existing dataset: {len(df_existing)} rows, "
              f"{df_existing['il_short_name'].nunique()} ILs")
    else:
        df_existing = pd.DataFrame()
        print("No existing dataset found")

    # Load ILThermo compounds
    compounds_csv = pipeline_dir / "ilthermo_compounds.csv"
    if not compounds_csv.exists():
        print("ERROR: Run step 1 first")
        return

    compounds = {}
    with open(compounds_csv) as f:
        for row in csv.DictReader(f):
            if row.get("smiles"):
                compounds[row["compound_id"]] = row

    # Load datasets info
    datasets_csv = pipeline_dir / "ilthermo_datasets.csv"
    datasets_info = []
    if datasets_csv.exists():
        with open(datasets_csv) as f:
            datasets_info = list(csv.DictReader(f))

    # Check which compounds have COSMO images
    images_dir = pipeline_dir / "cosmo_images"
    compounds_with_images = set()
    if images_dir.exists():
        for img_file in images_dir.glob("*_cosmo.png"):
            cid = img_file.stem.replace("_cosmo", "")
            compounds_with_images.add(cid)

    print(f"\nILThermo compounds: {len(compounds)}")
    print(f"Compounds with COSMO images: {len(compounds_with_images)}")
    print(f"Dataset entries to fetch: {len(datasets_info)}")

    # Fetch actual thermodynamic data for compounds with images
    # (limit to manageable number)
    target_compounds = {cid: compounds[cid] for cid in compounds_with_images
                        if cid in compounds}

    if not target_compounds:
        # If no images yet, process top compounds by dataset count
        sorted_compounds = sorted(compounds.items(),
                                  key=lambda x: -int(x[1].get("n_datasets", 0)))
        target_compounds = dict(sorted_compounds[:50])
        print(f"No images yet — targeting top {len(target_compounds)} compounds by data availability")

    # Fetch property data for target compounds
    print(f"\nFetching thermodynamic data for {len(target_compounds)} compounds...")

    all_new_rows = []
    fetched = 0

    # Group datasets by compound
    compound_datasets = {}
    for ds in datasets_info:
        for key in ["cmp1_id", "cmp2_id"]:
            cid = ds.get(key, "")
            if cid in target_compounds:
                compound_datasets.setdefault(cid, []).append(ds)

    for cid, comp_info in target_compounds.items():
        comp_ds = compound_datasets.get(cid, [])
        if not comp_ds:
            continue

        # Fetch up to 5 datasets per compound
        for ds in comp_ds[:5]:
            setid = ds["setid"]
            prop = ds["property"]

            dataset_data = fetch_dataset_data(setid)
            if not dataset_data:
                continue

            # Process based on property type
            if "activity" in prop.lower():
                rows = process_activity_dataset(dataset_data, comp_info)
            elif "excess enthalpy" in prop.lower():
                rows = process_excess_enthalpy_dataset(dataset_data, comp_info)
            else:
                continue

            for row in rows:
                row["il_short_name"] = comp_info["name"][:30]
                row["smiles"] = comp_info["smiles"]
                row["compound_id"] = cid
                # Image paths
                cosmo_img = images_dir / f"{cid}_cosmo.png"
                ep_img = images_dir / f"{cid}_ep.png"
                row["cosmo_image_path"] = str(cosmo_img) if cosmo_img.exists() else ""
                row["ep_image_path"] = str(ep_img) if ep_img.exists() else ""

            all_new_rows.extend(rows)
            fetched += 1

            time.sleep(0.5)  # Rate limiting

        if fetched % 20 == 0 and fetched > 0:
            print(f"  Fetched {fetched} datasets, {len(all_new_rows)} data points so far")

    print(f"\nFetched {len(all_new_rows)} new data points from ILThermo")

    # Convert to DataFrame
    if all_new_rows:
        df_new = pd.DataFrame(all_new_rows)

        # Fill missing targets with NaN
        for col in TARGET_COLUMNS:
            if col not in df_new.columns:
                df_new[col] = np.nan

        # Save new data
        new_data_path = output_dir / "ilthermo_data.csv"
        df_new.to_csv(new_data_path, index=False)
        print(f"Saved new ILThermo data to {new_data_path}")

        # Merge with existing (for compounds that have all required targets)
        # For now, save separately — full integration requires matching
        # the exact format of the existing preprocessing pipeline

        # Summary stats
        print(f"\n{'='*60}")
        print(f"Integration Summary")
        print(f"{'='*60}")
        print(f"  Existing dataset: {len(df_existing)} rows, "
              f"{df_existing['il_short_name'].nunique() if len(df_existing) > 0 else 0} ILs")
        print(f"  New ILThermo data: {len(df_new)} rows")
        n_new_ils = df_new["il_short_name"].nunique() if len(df_new) > 0 else 0
        print(f"  New unique ILs: {n_new_ils}")
        print(f"\n  Available properties in new data:")
        for col in TARGET_COLUMNS:
            if col in df_new.columns:
                n_valid = df_new[col].notna().sum()
                if n_valid > 0:
                    print(f"    {col}: {n_valid} data points")
    else:
        print("No new data fetched.")

    # Save pipeline metadata
    meta = {
        "existing_rows": len(df_existing),
        "existing_ils": int(df_existing["il_short_name"].nunique()) if len(df_existing) > 0 else 0,
        "new_rows": len(all_new_rows),
        "new_ils_with_images": len(compounds_with_images),
        "total_ilthermo_compounds": len(compounds),
    }
    with open(output_dir / "pipeline_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
