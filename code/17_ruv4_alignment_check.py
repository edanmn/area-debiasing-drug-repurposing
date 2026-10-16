"""Extends the alignment check (13_naive_vs_proposed_alignment.py) to RUV-4-style joint
estimation (11_ruv4_joint.py), which a reviewer pointed out was mentioned only in passing
("did not beat naive subtraction either") without the same direct alignment evidence given for
two-stage RUV-2. This measures it rather than assumes the same mechanism explains it.

RUV-4's own penalty term, made concrete the same way as RUV-2's (13_naive_vs_proposed_alignment.py
line ~95): A @ clf_k0.coef_[0][n_base:], the contribution the joint-estimated area factor's
coefficients would have made to the score had they not been projected out. Scored as a standalone
classifier of the same three bias-metric flags (same-ATC-area, hard-negative, proconvulsant), same
candidate population, same CVS2/5-seed protocol as the RUV-2 check.

Writes: results/ruv4_alignment_check.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from common import D
from indication_matrix import AREA_OF, ALL_DISEASES, build_matrix
from models import fit_shared_mf, chem_pca, fit_per_indication_logreg
import importlib
run_eval = importlib.import_module("06_run_evaluation")
ruv4 = importlib.import_module("11_ruv4_joint")

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32


def safe_auroc(flag, score):
    flag = np.asarray(flag)
    if flag.sum() == 0 or flag.sum() == len(flag):
        return np.nan
    return roc_auc_score(flag, score)


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)

    atc = pd.read_csv(D + "atc_level1.csv")
    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        from indication_matrix import negative_controls
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]
        area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
        same_area_flag = cmp.ik14.isin(area_iks).values

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            if len(nc_idx) == 0:
                continue

            clf_k0, n_base, A, rounds_run = ruv4.ruv4_joint_fit(Zrep, Y, train_mask, nc_idx, k0)
            if clf_k0 is None:
                continue
            area_penalty_ruv4 = A @ clf_k0.coef_[0][n_base:]

            cand = test_rows[y_full[test_rows] == 0]
            if len(cand) == 0:
                continue

            rows.append(dict(
                target=target, seed=seed, n_cand=len(cand), rounds_run=rounds_run,
                auroc_ruv4_samearea=safe_auroc(same_area_flag[cand], area_penalty_ruv4[cand]),
                auroc_ruv4_hardneg=safe_auroc(hard_col[cand], area_penalty_ruv4[cand]),
            ))
            if target == "epilepsy":
                rows[-1]["auroc_ruv4_proconv"] = safe_auroc(proconv_mask_all[cand], area_penalty_ruv4[cand])

    out = pd.DataFrame(rows)
    out.to_csv("../results/ruv4_alignment_check.csv", index=False)

    print("RUV-4-style joint estimator's own penalty term, AUROC as a standalone classifier of each flag:")
    summary = out.groupby("target")[[c for c in out.columns if c.startswith("auroc_")]].mean()
    print(summary.round(3))
    print("\n0.5 = no information about the flag (same convention as Table tab:alignment).")
    print("Wrote results/ruv4_alignment_check.csv")


if __name__ == "__main__":
    main()
