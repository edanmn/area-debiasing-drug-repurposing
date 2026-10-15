"""Fetch ATC Level-1 classification for every one of the 9,878 LINCS compounds via the ChEMBL API,
using the same request pattern already working in this codebase (ICHI2027_mgan/code/fetch_labels.py,
survey_atc.py): paged JSON GET with retries.

Strategy (more efficient than survey_atc.py's reverse lookup, which loops over ATC codes and is
only tractable for a handful of known classes):
  1. Fetch the whole ChEMBL atc_class table once (~5,600 rows, 6 paged requests) to build a
     level5 -> (level1_code, level1_description) map.
  2. Compute each compound's FULL InChIKey locally with RDKit from its SMILES (compounds.parquet
     only stores the 14-char skeleton prefix ik14).
  3. Batch-query the ChEMBL molecule endpoint by molecule_structures__standard_inchi_key__in
     (50 keys/request) to pull back each resolved molecule's atc_classifications (list of level5
     codes), matched back to our compound via the returned standard_inchi_key.
  4. Map level5 -> level1 and write one row per (ik14, atc_level1_code) pair to data/atc_level1.csv.

Checkpoints to results/logs so a partial run can be resumed/audited.
"""
import json, sys, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

sys.path.insert(0, ".")
from common import D, load_compounds

API = "https://www.ebi.ac.uk/chembl/api/data/"
LOG = open("../logs/02_fetch_atc_level1.log", "a")


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def get_json(url, tries=5):
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception as e:
            log(f"  retry ({e})")
            time.sleep(3 * (a + 1))
    return None


def fetch_full_atc_table():
    out, offset = [], 0
    while True:
        u = API + f"atc_class.json?limit=1000&offset={offset}"
        d = get_json(u)
        if d is None:
            break
        out += d["atc"]
        if not d["page_meta"]["next"]:
            break
        offset += 1000
    return pd.DataFrame(out)


def full_inchikeys(cmp):
    keys = {}
    n_fail = 0
    for ik14, smi in zip(cmp.ik14, cmp.smiles):
        if not isinstance(smi, str) or not smi:
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                n_fail += 1
                continue
            keys[ik14] = inchi.MolToInchiKey(m)
        except Exception:
            n_fail += 1
    log(f"RDKit InChIKey computed for {len(keys)}/{len(cmp)} compounds ({n_fail} failed to parse)")
    return keys


def batched_molecule_lookup(full_keys, batch=50):
    """full_keys: dict ik14 -> full inchikey. Returns dict full_inchikey -> list of level5 ATC codes."""
    all_keys = sorted(set(full_keys.values()))
    result = {}
    for i in range(0, len(all_keys), batch):
        chunk = all_keys[i:i + batch]
        u = (API + "molecule.json?molecule_structures__standard_inchi_key__in=" +
             urllib.parse.quote(",".join(chunk)) +
             "&only=molecule_chembl_id,molecule_structures,atc_classifications&limit=1000")
        d = get_json(u)
        if d is None:
            continue
        for mol in d.get("molecules", []):
            ms = mol.get("molecule_structures") or {}
            k = ms.get("standard_inchi_key")
            if k:
                result[k] = mol.get("atc_classifications") or []
        if (i // batch) % 20 == 0:
            log(f"  molecule lookup: {i + len(chunk)}/{len(all_keys)} inchikeys queried, "
                f"{len(result)} resolved so far")
    return result


def main():
    log("=== fetch ATC level-1 for all 9,878 compounds ===")
    log("Step 1: fetching full ChEMBL atc_class table ...")
    atc_tbl = fetch_full_atc_table()
    log(f"  atc_class table: {len(atc_tbl)} rows")
    atc_tbl.to_csv("../data/_atc_class_full.csv", index=False)
    level5_to_1 = atc_tbl.set_index("level5")[["level1", "level1_description"]].to_dict("index")

    cmp = load_compounds()
    log(f"Step 2: computing full InChIKeys for {len(cmp)} compounds ...")
    full_keys = full_inchikeys(cmp)

    log(f"Step 3: batch ChEMBL molecule lookup for {len(set(full_keys.values()))} unique full InChIKeys ...")
    key_to_atc = batched_molecule_lookup(full_keys)

    rows = []
    n_resolved = 0
    n_with_atc = 0
    for ik14, fk in full_keys.items():
        codes = key_to_atc.get(fk, [])
        if fk in key_to_atc:
            n_resolved += 1
        if codes:
            n_with_atc += 1
            seen_l1 = set()
            for c in codes:
                info = level5_to_1.get(c)
                if info is None:
                    continue
                l1 = info["level1"]
                if l1 in seen_l1:
                    continue
                seen_l1.add(l1)
                rows.append({"ik14": ik14, "atc_level1_code": l1,
                             "atc_level1_name": info["level1_description"]})
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv("../data/atc_level1.csv", index=False)
    log(f"\nResolved molecule record in ChEMBL (by full InChIKey) for {n_resolved}/{len(cmp)} compounds")
    log(f"Of those, {n_with_atc} have >=1 ATC classification")
    log(f"Wrote {len(out)} (ik14, atc_level1) rows to data/atc_level1.csv "
        f"covering {out.ik14.nunique()} distinct compounds")
    log(out.atc_level1_name.value_counts().to_string())


if __name__ == "__main__":
    main()
