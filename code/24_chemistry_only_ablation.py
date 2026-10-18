"""Chemistry-only CVS2 ablation (a third round of review, pasted inline, Issue 1): under CVS2, a held-out
compound's free-MF embedding never receives a gradient and stays at its random initialization
(documented in paper/main.tex's Method section and RESULTS.md section 5i). That random vector
carries no biological or chemical information about the compound and could in principle distort
the comparison between corrections differently, since naive subtraction and the proposed method
apply to the shared representation through different mathematical operations (a raw rate subtraction
vs. a projected-out SVD factor fit jointly with it).

This script removes the free-MF embedding entirely and reruns CVS2 (the only split where the
cold-start issue applies -- CVS1 holds out individual cells, not whole compound rows, and CVS3 is
already a stated degenerate limitation) with ONLY the 32-dimensional chemistry PCA representation,
for plain / naive_subtract / proposed_negctrl. If the main findings (naive wins every real-data
bias-diagnostic metric; the proposed method's own penalty term reads near chance except for
epilepsy's same-ATC-area flag) survive with no random cold-start vector in the representation at
all, the main paper's conclusions do not depend on it.

Writes: results/chemistry_only_ablation.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import (chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, project_area_factor_oos)
from sklearn.metrics import average_precision_score, roc_auc_score
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_CHEM = 32
LAMBDA_NAIVE = 2.0


def sign_free_auroc(flag, score):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)  # the representation, no MF embedding appended

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    atc = pd.read_csv(D + "atc_level1.csv")

    rows = []
    align_rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_names = negative_controls(target, diseases=cols)
        nc_idx = [col_idx[d] for d in nc_names]
        hard_col = hard_hits[:, k0]
        y_full = Y[:, k0].astype(int)
        # same-ATC-area flag: matches run_eval.area_enrichment / 13_naive_vs_proposed_alignment.py
        # (external WHO ATC level-1 code), not this project's own indication-matrix co-occurrence.
        area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
        same_area_flag = cmp.ik14.isin(area_iks).values

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Zrep = Xchem  # chemistry only -- no fit_shared_mf, no Ufree concatenation

            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None:
                continue
            s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

            area_rate = raw_area_rate(Y, nc_idx)
            ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
            s_naive = s_plain - LAMBDA_NAIVE * ar

            if len(nc_idx) == 0:
                s_debiased = s_plain.copy()
                area_penalty = np.zeros(N)
            else:
                A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
                train_idx_k0 = np.where(train_mask[:, k0])[0]
                clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
                if clf_db is None:
                    s_debiased = s_plain.copy()
                    area_penalty = np.zeros(N)
                else:
                    s_debiased = score_project(clf_db, Zrep, n_base)
                    A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
                    coef_A = clf_db.coef_[0][n_base:]
                    area_penalty = A_oos @ coef_A

            for name, s in [("plain", s_plain), ("naive_subtract", s_naive),
                             ("proposed_negctrl", s_debiased)]:
                yt = y_full[test_rows]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                ap = average_precision_score(yt, s[test_rows])
                top_rate, base_rate = run_eval.area_enrichment(cmp, s, y_full, target, test_rows)
                hn_pct = run_eval.hard_neg_rank_pct(s, test_rows, hard_col)
                row = dict(target=target, method=name, seed=seed, auprc=ap,
                           area_rate_top50=top_rate, area_base_rate=base_rate,
                           hard_neg_median_pct=hn_pct)
                if target == "epilepsy":
                    row["proconv_median_pct"] = run_eval.hard_neg_rank_pct(s, test_rows, proconv_mask_all)
                rows.append(row)

            if len(nc_idx) > 0:
                cand = test_rows[y_full[test_rows] == 0]
                flags = {"samearea": same_area_flag, "hardneg": hard_col}
                if target == "epilepsy":
                    flags["proconv"] = proconv_mask_all
                for flag_name, flag in flags.items():
                    if flag[cand].sum() == 0 or flag[cand].sum() == len(cand):
                        continue
                    align_rows.append(dict(target=target, flag=flag_name, seed=seed,
                                            observed_auroc=sign_free_auroc(flag[cand], area_penalty[cand])))

    out = pd.DataFrame(rows)
    out.to_csv("../results/chemistry_only_ablation.csv", index=False)
    align = pd.DataFrame(align_rows)
    align.to_csv("../results/chemistry_only_alignment.csv", index=False)

    print("=== bias-diagnostic metrics, chemistry-only CVS2 ===")
    print(out.groupby(["target", "method"])[["auprc", "area_rate_top50", "hard_neg_median_pct"]]
          .mean().round(3))
    print("\n=== standalone alignment AUROC, chemistry-only CVS2 ===")
    print(align.groupby(["target", "flag"]).observed_auroc.mean().round(3))
    print("\nWrote results/chemistry_only_ablation.csv, results/chemistry_only_alignment.csv")


if __name__ == "__main__":
    main()
