"""Extend the 9-class indication-label set with additional ChEMBL/ATC-defined indications so the
negative-control (RUV-style) mechanism has enough related-but-distinct diseases to draw on per
therapeutic area, per the task instructions ("as many additional ChEMBL indications as you can
reasonably pull in ... check what build_disease_labels.py already generalizes to").

Uses the exact same ATC-only labeling rule as ICHI2027_mgan/code/build_disease_labels.py's
antineoplastic/antibacterial/nsaid/lipid/betablocker entries (positive = ATC member; no indication
mesh/efo term search needed for these since the ATC class IS the definition): for each new code,
fetch its ATC level-5 codes, then the molecules under each, and flag LINCS membership by InChIKey14.

One negative-control partner (same ATC level-1 letter, distinct level-2/3 mechanism) is added for
each of the 9 original classes:
  epilepsy/depression/antipsychotic (all 'N', nervous system) get FOUR more N-area partners:
    parkinson (N04), anxiolytic (N05B), antidementia (N06D), opioid_analgesic (N02A)
  betablocker/lipid (both 'C', cardiovascular) get: raas_antihypertensive (C09)
  antidiabetic ('A', alimentary/metabolism) gets: antacid (A02)
  nsaid ('M', musculoskeletal) gets: osteoporosis (M05B)
  antibacterial ('J', anti-infective) gets: antimycotic (J02)
  antineoplastic ('L', antineoplastic/immune) gets: immunosuppressant (L04)
"""
import json, sys, time, urllib.parse, urllib.request
import pandas as pd

sys.path.insert(0, ".")
from common import load_compounds

API = "https://www.ebi.ac.uk/chembl/api/data/"
LOG = open("../logs/05_build_extra_indications.log", "a")

EXTRA = {
    "parkinson": "N04", "anxiolytic": "N05B", "antidementia": "N06D", "opioid_analgesic": "N02A",
    "raas_antihypertensive": "C09", "antacid": "A02", "osteoporosis": "M05B",
    "antimycotic": "J02", "immunosuppressant": "L04",
}

# area(indication) -> ATC level-1 letter; mechanism(indication) -> the ATC code used to define it
# (so "same area, different mechanism" negative controls can be selected programmatically later).
AREA_OF = {
    "epilepsy": "N", "depression": "N", "antipsychotic": "N", "parkinson": "N", "anxiolytic": "N",
    "antidementia": "N", "opioid_analgesic": "N",
    "betablocker": "C", "lipid": "C", "raas_antihypertensive": "C",
    "antidiabetic": "A", "antacid": "A",
    "nsaid": "M", "osteoporosis": "M",
    "antibacterial": "J", "antimycotic": "J",
    "antineoplastic": "L", "immunosuppressant": "L",
}
MECH_OF = {  # ATC code defining the indication's own mechanism/class
    "epilepsy": "N03", "depression": "N06A", "antipsychotic": "N05A", "parkinson": "N04",
    "anxiolytic": "N05B", "antidementia": "N06D", "opioid_analgesic": "N02A",
    "betablocker": "C07", "lipid": "C10", "raas_antihypertensive": "C09",
    "antidiabetic": "A10", "antacid": "A02", "nsaid": "M01A", "osteoporosis": "M05B",
    "antibacterial": "J01", "antimycotic": "J02", "antineoplastic": "L01", "immunosuppressant": "L04",
}


def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


def paged(ep, params):
    out, offset = [], 0
    while True:
        u = API + ep + ".json?" + urllib.parse.urlencode(dict(params, limit=1000, offset=offset))
        for a in range(4):
            try:
                with urllib.request.urlopen(u, timeout=90) as r:
                    d = json.load(r)
                break
            except Exception:
                time.sleep(3 * (a + 1))
        else:
            return out
        key = [k for k in d if k != "page_meta"][0]
        out += d[key]
        if not d["page_meta"]["next"]:
            return out
        offset += 1000


def build_one(name, atc_code):
    level = "level2" if len(atc_code) == 3 else ("level3" if len(atc_code) == 4 else "level1")
    atcs = paged("atc_class", {level: atc_code})
    aik = set()
    for a in atcs:
        for m in paged("molecule", {"atc_classifications__level5": a["level5"], "only": "molecule_structures"}):
            s = (m.get("molecule_structures") or {}).get("standard_inchi_key")
            if s:
                aik.add(s[:14])
    cmp = load_compounds()[["ik14", "name"]]
    out = cmp.copy()
    out["atc"] = out.ik14.isin(aik)
    out["positive"] = out.atc
    out["clinical"] = out.atc
    out["max_phase"] = pd.NA
    out.to_csv(f"../data/labels_{name}.csv", index=False)
    log(f"{name} ({atc_code}): {len(atcs)} ATC codes, {len(aik)} molecules, "
        f"{int(out.positive.sum())} positives in LINCS of {len(out)}")
    return int(out.positive.sum())


def main():
    log("=== building additional ChEMBL/ATC indications for negative-control pool ===")
    counts = {}
    for name, code in EXTRA.items():
        counts[name] = build_one(name, code)
    pd.DataFrame({"area_of": AREA_OF, "mechanism_of": MECH_OF}).to_csv("../data/indication_area_map.csv")
    log(f"\nTotal indications now available: {len(counts) + 9} "
        f"(9 original + {len(counts)} new negative-control partners)")
    log(str(counts))


if __name__ == "__main__":
    main()
