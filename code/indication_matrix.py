"""Build the compound x indication binary matrix Y used by the MF method and all baselines
(step 3/4 of the task), over the 9 original ChEMBL indication classes plus the 9 additional
ATC-defined negative-control-partner indications built by 05_build_extra_indications.py.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import D, load_compounds, load_disease_labels, load_ecfp, NINE_DISEASES

EXTRA_DISEASES = ["parkinson", "anxiolytic", "antidementia", "opioid_analgesic",
                   "raas_antihypertensive", "antacid", "osteoporosis", "antimycotic",
                   "immunosuppressant"]
ALL_DISEASES = NINE_DISEASES + EXTRA_DISEASES

AREA_OF = {
    "epilepsy": "N", "depression": "N", "antipsychotic": "N", "parkinson": "N", "anxiolytic": "N",
    "antidementia": "N", "opioid_analgesic": "N",
    "betablocker": "C", "lipid": "C", "raas_antihypertensive": "C",
    "antidiabetic": "A", "antacid": "A",
    "nsaid": "M", "osteoporosis": "M",
    "antibacterial": "J", "antimycotic": "J",
    "antineoplastic": "L", "immunosuppressant": "L",
}
MECH_OF = {
    "epilepsy": "N03", "depression": "N06A", "antipsychotic": "N05A", "parkinson": "N04",
    "anxiolytic": "N05B", "antidementia": "N06D", "opioid_analgesic": "N02A",
    "betablocker": "C07", "lipid": "C10", "raas_antihypertensive": "C09",
    "antidiabetic": "A10", "antacid": "A02", "nsaid": "M01A", "osteoporosis": "M05B",
    "antibacterial": "J01", "antimycotic": "J02", "antineoplastic": "L01", "immunosuppressant": "L04",
}


def negative_controls(target, diseases=ALL_DISEASES):
    """Other indications in the same ATC-level1 area as `target` but a DIFFERENT level2/3
    mechanism class (the RUV-style negative-control set, literature_review.md item 1)."""
    area = AREA_OF[target]
    mech = MECH_OF[target]
    return [d for d in diseases if d != target and AREA_OF.get(d) == area and MECH_OF.get(d) != mech]


def build_matrix(diseases=ALL_DISEASES, pool="drugs"):
    """pool='drugs' restricts to the candidate pool actually worth ranking (ChEMBL max_phase>=1 for
    ANY of the 18 indications, OR already positive for one) -- see logs/model_diagnostics.log for
    why evaluating on the full 9,878-compound library (94% all-zero rows) gives no usable signal
    for a pure cross-indication co-occurrence factorization."""
    cmp = load_compounds()
    Y_full = np.zeros((len(cmp), len(diseases)), dtype=np.float32)
    for j, d in enumerate(diseases):
        lab = load_disease_labels(d)
        Y_full[:, j] = cmp.ik14.map(lab["positive"]).fillna(False).astype(bool).values
    keep = cmp.has_structure.values.copy()
    if pool == "drugs":
        ph = pd.read_csv(D + "chembl_max_phase_all.csv").set_index("ik14").chembl_max_phase
        any_phase = cmp.ik14.map(ph).fillna(0).values >= 1
        any_pos = Y_full.sum(1) > 0
        keep = keep & (any_phase | any_pos)
    cmp = cmp[keep].reset_index(drop=True)
    Y = Y_full[keep]
    ecfp = load_ecfp()[keep]
    return cmp, Y, diseases, ecfp


if __name__ == "__main__":
    cmp, Y, diseases, ecfp = build_matrix()
    print(f"Indication matrix (drug pool): {Y.shape[0]} compounds x {Y.shape[1]} indications")
    for j, d in enumerate(diseases):
        print(f"  {d:24s} area={AREA_OF[d]:1s} mech={MECH_OF[d]:5s} positives={int(Y[:, j].sum())}  "
              f"neg_controls={negative_controls(d)}")
