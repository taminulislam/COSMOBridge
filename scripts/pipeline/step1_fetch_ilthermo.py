"""Step 1: Fetch ionic liquid compound data from ILThermo (NIST).

Queries ILThermo for binary mixture datasets with activity coefficients
and excess enthalpy, extracts compound names/SMILES, and resolves missing
SMILES via PubChem.

Output: data/pipeline/ilthermo_compounds.csv
"""

import json
import time
import csv
import re
import urllib.request
import urllib.parse
from pathlib import Path

BASE_URL = "https://ilthermo.boulder.nist.gov"
HEADERS = {"User-Agent": "ILResearch/1.0 (Academic; SIU)"}


def fetch_json(url):
    """Fetch JSON from ILThermo endpoint with retry."""
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("iso-8859-1"))
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED: {url}: {e}")
                return None


def get_property_keys():
    """Fetch current property name -> 4-char key mapping."""
    data = fetch_json(f"{BASE_URL}/ILT2/ilprpls")
    if not data:
        return {}
    keys = {}
    # ILThermo returns {plist: [{cls, name: [...], key: [...]}, ...]}
    for cls_entry in data.get("plist", []):
        names = cls_entry.get("name", [])
        key_list = cls_entry.get("key", [])
        for n, k in zip(names, key_list):
            keys[n.strip()] = k
    # Fallback: old format {res: [[key, name], ...]}
    for row in data.get("res", []):
        if len(row) >= 2:
            keys[row[1].strip()] = row[0]
    return keys


def search_datasets(prp_key, ncmp=2):
    """Search ILThermo for datasets of a given property type."""
    url = f"{BASE_URL}/ILT2/ilsearch?cmp=&ncmp={ncmp}&prp={prp_key}&year=&auth=&keyw="
    data = fetch_json(url)
    if not data:
        return []
    return data.get("res", [])


def fetch_dataset(setid):
    """Fetch full data for a dataset by setid."""
    url = f"{BASE_URL}/ILT2/ilset?set={setid}"
    data = fetch_json(url)
    return data


def resolve_smiles_pubchem(name):
    """Try to resolve compound name to SMILES via PubChem."""
    try:
        encoded = urllib.parse.quote(name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES,IsomericSMILES/JSON"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                p = props[0]
                # PubChem may return different SMILES field names
                return (p.get("CanonicalSMILES") or p.get("IsomericSMILES")
                        or p.get("ConnectivitySMILES") or p.get("SMILES"))
    except Exception:
        pass
    return None


def is_ionic_liquid_name(name):
    """Heuristic to identify if a compound name is likely an ionic liquid."""
    il_indicators = [
        "imidazolium", "pyridinium", "pyrrolidinium", "ammonium", "phosphonium",
        "sulfonium", "guanidinium", "cholinium", "choline",
        "bis(trifluoromethylsulfonyl)", "tetrafluoroborate", "hexafluorophosphate",
        "chloride", "bromide", "acetate", "lactate", "sulfate",
        "1-ethyl-3-methyl", "1-butyl-3-methyl", "1-hexyl-3-methyl",
        "1-octyl-3-methyl", "triethyl", "tributyl",
        "EMIM", "BMIM", "HMIM", "OMIM",
    ]
    name_lower = name.lower()
    return any(ind.lower() in name_lower for ind in il_indicators)


def main():
    output_dir = Path("data/pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Fetching property keys from ILThermo...")
    prop_keys = get_property_keys()
    print(f"  Found {len(prop_keys)} property types")

    # Target properties for our dataset
    target_props = {
        "Activity": None,
        "Excess enthalpy": None,
        "Excess Gibbs energy": None,
        "Vapor or sublimation pressure": None,
        "Enthalpy of vaporization or sublimation": None,
    }

    # Match property keys (case-insensitive partial match)
    for target_name in list(target_props.keys()):
        for prop_name, key in prop_keys.items():
            if target_name.lower() in prop_name.lower():
                target_props[target_name] = key
                print(f"  {target_name} -> key={key}")
                break

    # Collect all unique compounds from matching datasets
    all_compounds = {}  # compound_id -> {name, datasets, is_il}
    all_datasets_info = []

    for prop_name, prop_key in target_props.items():
        if not prop_key:
            print(f"\n  SKIP: No key found for '{prop_name}'")
            continue

        print(f"\nSearching for '{prop_name}' (key={prop_key}, binary mixtures)...")
        datasets = search_datasets(prop_key, ncmp=2)
        print(f"  Found {len(datasets)} datasets")

        for row in datasets:
            # row format varies: [setid, ref, prp, phases, cmp1_id, cmp2_id, (cmp3_id), np, nm1, nm2, (nm3)]
            if len(row) < 8:
                continue
            setid = row[0]
            ref = row[1]
            prp = row[2] if len(row) > 2 else ""
            cmp1_id = row[4] if len(row) > 4 else ""
            cmp2_id = row[5] if len(row) > 5 else ""
            np_str = row[7] if len(row) > 7 else "0"
            nm1 = row[8] if len(row) > 8 else ""
            nm2 = row[9] if len(row) > 9 else ""

            for cmp_id, name in [(cmp1_id, nm1), (cmp2_id, nm2)]:
                if cmp_id and name and cmp_id not in all_compounds:
                    all_compounds[cmp_id] = {
                        "name": name,
                        "is_il": is_ionic_liquid_name(name),
                        "datasets": [],
                    }
                if cmp_id in all_compounds:
                    all_compounds[cmp_id]["datasets"].append({
                        "setid": setid,
                        "property": prop_name,
                        "ref": ref,
                        "n_points": np_str,
                    })

            all_datasets_info.append({
                "setid": setid,
                "property": prop_name,
                "ref": ref,
                "cmp1_id": cmp1_id or "",
                "cmp1_name": nm1 or "",
                "cmp2_id": cmp2_id or "",
                "cmp2_name": nm2 or "",
                "n_points": np_str or "0",
            })

        time.sleep(1)  # Rate limiting

    # Filter to ionic liquids only
    il_compounds = {k: v for k, v in all_compounds.items() if v["is_il"]}
    print(f"\nTotal unique compounds: {len(all_compounds)}")
    print(f"Identified as ionic liquids: {len(il_compounds)}")

    # Resolve SMILES for IL compounds
    print(f"\nResolving SMILES via PubChem (up to {min(len(il_compounds), 200)} ILs)...")
    resolved = 0
    for i, (cmp_id, info) in enumerate(il_compounds.items()):
        if i >= 200:  # Limit to avoid excessive PubChem queries
            break
        smiles = resolve_smiles_pubchem(info["name"])
        info["smiles"] = smiles
        if smiles:
            resolved += 1
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{min(len(il_compounds), 200)}, resolved={resolved}")
        time.sleep(0.5)  # PubChem rate limiting

    print(f"  Resolved {resolved}/{min(len(il_compounds), 200)} SMILES")

    # Save compounds CSV
    compounds_path = output_dir / "ilthermo_compounds.csv"
    with open(compounds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["compound_id", "name", "smiles", "is_il", "n_datasets"])
        for cmp_id, info in sorted(il_compounds.items(), key=lambda x: -len(x[1]["datasets"])):
            writer.writerow([
                cmp_id,
                info["name"],
                info.get("smiles", ""),
                info["is_il"],
                len(info["datasets"]),
            ])
    print(f"\nSaved {len(il_compounds)} IL compounds to {compounds_path}")

    # Save datasets CSV
    datasets_path = output_dir / "ilthermo_datasets.csv"
    with open(datasets_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["setid", "property", "ref", "cmp1_id", "cmp1_name",
                         "cmp2_id", "cmp2_name", "n_points"])
        for d in all_datasets_info:
            writer.writerow([d["setid"], d["property"], d["ref"], d["cmp1_id"],
                            d["cmp1_name"], d["cmp2_id"], d["cmp2_name"], d["n_points"]])
    print(f"Saved {len(all_datasets_info)} dataset entries to {datasets_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ILThermo Fetch Summary")
    print(f"{'='*60}")
    print(f"  Total datasets found: {len(all_datasets_info)}")
    print(f"  Total unique compounds: {len(all_compounds)}")
    print(f"  Ionic liquids identified: {len(il_compounds)}")
    print(f"  ILs with resolved SMILES: {resolved}")
    print(f"\nTop 20 ILs by dataset count:")
    for cmp_id, info in sorted(il_compounds.items(), key=lambda x: -len(x[1]["datasets"]))[:20]:
        smiles_str = info.get("smiles", "N/A") or "N/A"
        print(f"  {info['name'][:50]:50s} datasets={len(info['datasets']):3d}  SMILES={'✓' if smiles_str != 'N/A' else '✗'}")


if __name__ == "__main__":
    main()
