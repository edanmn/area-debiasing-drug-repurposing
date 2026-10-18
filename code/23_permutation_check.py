"""Permutation test for Table 3 (sec:alignment), prompted by a second external review
(assessment_updated.txt, Issue 6): max(AUROC, 1-AUROC) is upward-biased under the null when the
sign is picked on the same evaluation set the statistic is reported on, since |AUROC-0.5| has
positive expectation even when the true AUROC is exactly 0.5. This matters most for small,
imbalanced flag-positive counts, exactly the epilepsy hard-negative (36 positives) and
proconvulsant (11 positives) cells.

For each (target, flag) cell already in Table 3, shuffle the flag within the candidate set 5000
times, holding the real fitted penalty term fixed, and recompute the same sign-free statistic each
time. The reported p-value is the fraction of permutations whose statistic is >= the observed one:
this is the correctly-sized test for "does this penalty term carry real information about this
flag," since it uses the same sign-selection procedure under the null that was used to compute the
observed statistic.

Result: depression and antipsychotic (all cells) and epilepsy same-ATC-area are each clearly one
side or the other (p<0.01 or p>0.3 consistently across seeds). Epilepsy hard-negative and
proconvulsant are NOT distinguishable from this null (p=0.335, p=0.162 pooled across 5 seeds,
individual seeds ranging from p<0.01 to p>0.5) -- the numeric AUROC values in Table 3 for these two
cells (0.565, 0.676) are not safe to read as "the correction's own term carries real information"
without this caveat.

Writes: results/permutation_check.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, project_area_factor_oos)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

R_FREE, R_CHEM = 16, 32
N_PERM = 5000
SEEDS = range(5)


def sign_free_auroc(flag, score):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


def perm_pvalue(flag, score, n_perm=N_PERM, seed=0):
    obs = sign_free_auroc(flag, score)
    rng = np.random.default_rng(seed)
    flag = np.asarray(flag)
    count = sum(sign_free_auroc(rng.permutation(flag), score) >= obs for _ in range(n_perm))
    return obs, count / n_perm


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)
    proconv = pd.read_csv("../data/proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    rows = []
    for target in ["depression", "antipsychotic", "epilepsy"]:
        k0 = col_idx[target]
        area = AREA_OF[target]
        area_diseases = [d for d in cols if AREA_OF.get(d) == area]
        area_iks = set()
        for d in area_diseases:
            area_iks |= set(cmp.ik14[Y[:, col_idx[d]] == 1])
        same_area_flag = cmp.ik14.isin(area_iks).values
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]

        flags = {"samearea": same_area_flag, "hardneg": hard_col}
        if target == "epilepsy":
            flags["proconv"] = proconv_mask_all

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            if clfs[k0] is None:
                continue
            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            coef_A = clf_db.coef_[0][n_base:]
            area_penalty = A_oos @ coef_A
            cand = test_rows[y_full[test_rows] == 0]

            for flag_name, flag in flags.items():
                obs, pv = perm_pvalue(flag[cand], area_penalty[cand], seed=seed)
                rows.append(dict(target=target, flag=flag_name, seed=seed,
                                  n_cand=len(cand), n_pos=int(flag[cand].sum()),
                                  observed_auroc=obs, perm_pvalue=pv))

    out = pd.DataFrame(rows)
    out.to_csv("../results/permutation_check.csv", index=False)
    print(out.groupby(["target", "flag"])[["observed_auroc", "perm_pvalue"]].mean().round(3))
    print("\nWrote results/permutation_check.csv")


if __name__ == "__main__":
    main()
