"""Shared data loading for idea2 (area-debiasing). Mirrors ICHI2027_mgan/code/evallib.py's
loading convention but is self-contained in this project's data/ (symlinked to the originals,
see data/ directory listing). All indices below are over the full 9,878-compound LINCS library
unless pool='drugs' is requested.
"""
import os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data") + os.sep

# The nine ChEMBL-derived indication classes used throughout idea1/idea2 (per DATA_PLAN.md).
NINE_DISEASES = ["epilepsy", "depression", "antipsychotic", "antineoplastic", "antibacterial",
                  "antidiabetic", "betablocker", "lipid", "nsaid"]


def load_compounds():
    """Base compound table: ik14, name, smiles, scaffold, has_structure, ... (9,878 rows)."""
    return pd.read_parquet(D + "compounds.parquet")


def load_disease_labels(disease):
    """Returns a DataFrame indexed by ik14 with columns: positive (bool), clinical (bool), max_phase.
    disease='epilepsy' reads straight from compounds.parquet (approved/clinical/max_phase columns);
    all others read data/labels_<disease>.csv, built by ICHI2027_mgan/code/build_disease_labels.py."""
    cmp = load_compounds()
    if disease == "epilepsy":
        out = cmp[["ik14"]].copy()
        out["positive"] = cmp["approved"].fillna(False).astype(bool).values
        out["clinical"] = cmp["clinical"].fillna(False).astype(bool).values
        out["max_phase"] = cmp["max_phase"].values
    else:
        lab = pd.read_csv(D + f"labels_{disease}.csv")
        out = lab[["ik14", "positive", "clinical", "max_phase"]].copy()
        out["positive"] = out["positive"].fillna(False).astype(bool)
        out["clinical"] = out["clinical"].fillna(False).astype(bool)
    return out.set_index("ik14")


def load_pool(pool="all"):
    """Boolean mask over the 9,878-row compounds.parquet table selecting compounds with a resolved
    structure (pool='all') or additionally restricted to the drug-like candidate pool used for
    ranking (pool='drugs': ChEMBL max_phase>=1 for ANY indication, or approved for epilepsy)."""
    cmp = load_compounds()
    keep = cmp.has_structure.values.copy()
    if pool == "drugs":
        ph = pd.read_csv(D + "chembl_max_phase_all.csv").set_index("ik14").chembl_max_phase
        approved_any = cmp.approved.fillna(False).values.astype(bool)
        keep = keep & ((cmp.ik14.map(ph).fillna(0).values >= 1) | approved_any)
    return keep


def load_ecfp():
    return np.load(D + "cmp_ecfp.npy").astype(np.float32)


def tanimoto_max(A, B):
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return (inter / np.maximum(union, 1)).max(1)


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for the mean of `values` (a 1D array of per-fold/per-seed statistics)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return (float(values.mean()), float(lo), float(hi))
