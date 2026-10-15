"""Scaffold-disjoint (Murcko scaffold group) version of CVS2, specifically because the shared
representation includes an ECFP4-based PCA component (models.chem_pca) -- literature_review.md
section 5 item 3 requires any ECFP4-based component to use a Tanimoto/scaffold-disjoint split, not
a random one, "exactly as the PACT abstract itself already does." The main 06_run_evaluation.py
CVS2 uses a random 20% compound holdout; this script reruns the SAME CVS2 protocol but with
scaffold-grouped holdout (entire Murcko-scaffold groups assigned to train/test, never split across
the boundary), so a chemically near-duplicate compound of a training drug can never leak into the
test set for that fold. Reports whether this materially changes the headline AUPRC numbers.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import bootstrap_ci
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from splits import scaffold_groups
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, nrbdmf_signed_matrix)

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE, R_CHEM, LAMBDA_NAIVE = 16, 32, 2.0


def scaffold_holdout_mask(groups, n_groups_total, seed, frac=0.2):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test_groups = max(1, int(round(frac * len(uniq))))
    test_groups = set(uniq[:n_test_groups])
    return np.array([g in test_groups for g in groups])


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    groups = scaffold_groups(cmp)
    print(f"Scaffold groups in drug pool: {len(np.unique(groups))} distinct groups over {N} compounds")
    hard_hits = np.zeros((N, K), dtype=bool)  # not needed for this check's AUPRC-only comparison

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        for seed in SEEDS:
            row_test = scaffold_holdout_mask(groups, N, seed)
            train_mask = np.ones((N, K), dtype=bool)
            train_mask[row_test, :] = False
            test_rows = np.where(row_test)[0]
            if y_full[test_rows].sum() == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            s_plain = score_project(clfs[k0], Zrep, Zrep.shape[1])
            area_rate = raw_area_rate(Y, nc_idx)
            ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
            s_naive = s_plain - LAMBDA_NAIVE * ar
            if len(nc_idx) > 0:
                A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
                train_idx_k0 = np.where(train_mask[:, k0])[0]
                clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
                s_debiased = score_project(clf_db, Zrep, n_base)
            else:
                s_debiased = s_plain.copy()
            for name, s in [("plain", s_plain), ("naive_subtract", s_naive),
                             ("proposed_negctrl", s_debiased)]:
                yt = y_full[test_rows]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                ap = average_precision_score(yt, s[test_rows])
                rows.append(dict(target=target, method=name, seed=seed, auprc=ap, split="scaffold_cvs2"))

    df = pd.DataFrame(rows)
    df.to_csv("../results/scaffold_disjoint_cvs2_auprc.csv", index=False)
    summary = []
    for (target, method), g in df.groupby(["target", "method"]):
        m, lo, hi = bootstrap_ci(g.auprc.values)
        summary.append(dict(target=target, method=method, mean=m, ci_lo=lo, ci_hi=hi, n=len(g)))
    summ = pd.DataFrame(summary)
    summ.to_csv("../results/scaffold_disjoint_cvs2_summary.csv", index=False)
    print(summ.to_string(index=False))

    rand = pd.read_csv("../results/eval_summary.csv")
    rand = rand[(rand.metric == "auprc") & (rand.cvs == "CVS2") & (rand.method != "nrbdmf")]
    print("\nFor comparison, random-row CVS2 AUPRC from the main run (results/eval_summary.csv):")
    print(rand[["target", "method", "mean", "ci_lo", "ci_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
