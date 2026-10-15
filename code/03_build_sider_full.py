"""Extend the PubChem-CID-to-SIDER join from the existing 1,725-compound Phase-2 subset
(ICLR2027_repurpose/data/lincs_sider_map.csv, name-based, 534 of 1,725 resolved) to the FULL
9,878-compound library in compounds.parquet, per DATA_PLAN.md.

Method: compute each compound's full InChIKey locally with RDKit (same routine as
02_fetch_atc_level1.py), resolve PubChem CID via the PubChem PUG-REST InChIKey->CID lookup
(batched, with each response's own CID+InChIKey pairs used to match results back correctly --
PubChem's "cids" endpoint does NOT reliably preserve input order under batching, confirmed by a
smoke test, so we use "property/InChIKey/JSON" instead, which echoes the matched InChIKey per row),
then join the resulting PubChem CID against the existing raw meddra_all_se.tsv.gz
(STITCH flat compound id = 'CID1' + 8-digit zero-padded PubChem CID) -- no new download needed.

Writes data/sider_full.parquet: one row per (ik14, se_name) pair, for every compound that resolves
to a SIDER-mapped PubChem CID. Reports actual final coverage out of 9,878.
"""
import sys, time, urllib.parse, urllib.request, json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

sys.path.insert(0, ".")
from common import D, load_compounds

LOG = open("../logs/03_build_sider_full.log", "a")


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def full_inchikeys(cmp):
    keys = {}
    for ik14, smi in zip(cmp.ik14, cmp.smiles):
        if not isinstance(smi, str) or not smi:
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            keys[ik14] = inchi.MolToInchiKey(m)
        except Exception:
            continue
    return keys


def pubchem_cid_lookup(full_keys, batch=60):
    all_keys = sorted(set(full_keys.values()))
    key_to_cid = {}
    for i in range(0, len(all_keys), batch):
        chunk = all_keys[i:i + batch]
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/" +
               urllib.parse.quote(",".join(chunk)) + "/property/InChIKey/JSON")
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    d = json.load(r)
                for row in d.get("PropertyTable", {}).get("Properties", []):
                    key_to_cid[row["InChIKey"]] = row["CID"]
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break  # none of this batch resolved
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.25)  # PubChem rate-limit courtesy
        if (i // batch) % 20 == 0:
            log(f"  pubchem lookup: {i + len(chunk)}/{len(all_keys)} keys queried, "
                f"{len(key_to_cid)} resolved so far")
    return key_to_cid


def main():
    log("=== extend SIDER coverage to full 9,878-compound library ===")
    cmp = load_compounds()
    full_keys = full_inchikeys(cmp)
    log(f"RDKit full InChIKey computed for {len(full_keys)}/{len(cmp)} compounds")

    log("Resolving PubChem CIDs via InChIKey batch lookup ...")
    key_to_cid = pubchem_cid_lookup(full_keys)
    log(f"Resolved PubChem CID for {len(key_to_cid)}/{len(set(full_keys.values()))} unique InChIKeys")

    ik14_to_cid = {ik: key_to_cid.get(fk) for ik, fk in full_keys.items() if key_to_cid.get(fk)}
    log(f"-> {len(ik14_to_cid)}/{len(cmp)} of the 9,878 compounds have a resolved PubChem CID")

    log("Loading raw meddra_all_se.tsv.gz (SIDER side-effect table, flat STITCH CIDs) ...")
    meddra = pd.read_csv(D + "meddra_all_se.tsv.gz", sep="\t", header=None,
                         names=["cid_flat", "cid_stereo", "umls_label", "meddra_type",
                                "umls_meddra", "se_name"])
    meddra_pt = meddra[meddra.meddra_type == "PT"][["cid_flat", "se_name"]].drop_duplicates()
    log(f"meddra PT rows: {len(meddra_pt)}, distinct flat CIDs: {meddra_pt.cid_flat.nunique()}")

    rows = []
    n_cid_in_sider = 0
    for ik14, cid in ik14_to_cid.items():
        flat = f"CID1{int(cid):08d}"
        sub = meddra_pt[meddra_pt.cid_flat == flat]
        if len(sub):
            n_cid_in_sider += 1
            for se in sub.se_name.unique():
                rows.append({"ik14": ik14, "pubchem_cid": cid, "se_name": se})

    out = pd.DataFrame(rows)
    out.to_parquet("../data/sider_full.parquet", index=False)
    n_total = len(cmp)
    log(f"\nFinal coverage: {n_cid_in_sider}/{n_total} ({100*n_cid_in_sider/n_total:.1f}%) of the "
        f"9,878 compounds resolve to a SIDER-mapped PubChem CID")
    log(f"Wrote data/sider_full.parquet: {len(out)} (ik14, se_name) rows, "
        f"{out.ik14.nunique() if len(out) else 0} distinct compounds, "
        f"{out.se_name.nunique() if len(out) else 0} distinct MedDRA PTs")

    # Cross-check against the previous partial mapping for sanity (coverage should only improve).
    prev = pd.read_csv(D + "lincs_sider_map.csv")
    log(f"\n[sanity] previous partial map (name-based, Phase-2 subset only): {len(prev)} compounds, "
        f"of which resolved to a cid: {prev.cid.notna().sum()}")


if __name__ == "__main__":
    main()
