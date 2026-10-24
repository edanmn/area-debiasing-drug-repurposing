"""W2b (reviewer feedback round 3): isolates whether proposed's / direct-subtraction's apparent
benefit on the real-data bias metrics depends on genuine negative-control information, or would
appear from adding and then removing ANY similarly-shaped nuisance factor.

Design choice (stated explicitly, as the task asked): rather than a random projection of the shared
representation Zrep (which would still correlate with real chemistry/indication signal through
Zrep itself, undermining the point of a null control), this draws a factor R of the SAME shape as
the real negative-control area factor A (N x r_area=2) as i.i.d. Gaussian noise, independent of
every compound's chemistry, indication membership, and negative-control structure, with each
column's standard deviation matched to that seed/target's REAL fitted A (so the magnitude the
downstream logistic fit sees is comparable, not just the rank). Because R does not depend on Zrep
or train/test split in any systematic way, no out-of-sample projection step is needed or applicable:
R is simply drawn once per (target, seed) for every row, so "train" and "test" rows get the same
kind of value a real negative-control factor's out-of-sample estimate would, a draw carrying no
real information by construction.

Applies both application mechanisms to this random factor, same as the real analyses:
  refit_drop       -- fit_with_extra(Zrep, R, y, train_idx) then score_project (mirrors "proposed")
  direct_subtract  -- calibrate a logistic coefficient on R[train_idx], subtract R @ coef from the
                       plain logit (mirrors code/31_direct_subtraction_real_data.py)

Writes: results/random_factor_control.csv, results/random_factor_control_summary.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, sigmoid)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    auprc_rows, enrich_rows, hardneg_rows, unique_compound_rows = [], [], [], []

    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        hard_col = hard_hits[:, k0]
        y_full = Y[:, k0].astype(int)
        test_iks_union = set()

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            test_iks_union.update(cmp.ik14.values[test_rows])
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None or len(nc_idx) == 0:
                continue

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            col_std = A.std(0)
            rng = np.random.default_rng(10_000 + seed * 37 + k0)
            R = rng.normal(0.0, 1.0, size=A.shape) * col_std[None, :]

            train_idx_k0 = np.where(train_mask[:, k0])[0]
            ytr = y_full[train_idx_k0]
            if ytr.sum() < 2 or ytr.sum() == len(ytr):
                continue

            # refit_drop: mirrors "proposed"
            clf_db, n_base = fit_with_extra(Zrep, R, y_full, train_idx_k0)
            s_refit_drop = score_project(clf_db, Zrep, n_base) if clf_db is not None else None

            # direct_subtract: mirrors code/31
            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                R[train_idx_k0], ytr)
            logit_plain = clf_plain.decision_function(Zrep)
            s_direct = sigmoid(logit_plain - R @ calib.coef_[0])

            methods = [("random_direct_subtract", s_direct)]
            if s_refit_drop is not None:
                methods.append(("random_refit_drop", s_refit_drop))

            for name, s in methods:
                yt = y_full[test_rows]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                ap = average_precision_score(yt, s[test_rows])
                auprc_rows.append(dict(target=target, cvs="CVS2", method=name, seed=seed, auprc=ap,
                                       n_test=len(test_rows), n_pos=int(yt.sum())))
                top_rate, base_rate = run_eval.area_enrichment(cmp, s, y_full, target, test_rows)
                enrich_rows.append(dict(target=target, cvs="CVS2", method=name, seed=seed,
                                        top50_rate=top_rate, base_rate=base_rate))
                hn_pct = run_eval.hard_neg_rank_pct(s, test_rows, hard_col)
                row = dict(target=target, cvs="CVS2", method=name, seed=seed,
                           hard_neg_median_pct=hn_pct, n_hard_neg=int(hard_col[test_rows].sum()),
                           proconv_median_pct=np.nan, n_proconv=0)
                if target == "epilepsy":
                    row["proconv_median_pct"] = run_eval.hard_neg_rank_pct(s, test_rows, proconv_mask_all)
                    row["n_proconv"] = int(proconv_mask_all[test_rows].sum())
                hardneg_rows.append(row)

        unique_compound_rows.append(dict(target=target, n_unique_test_compounds=len(test_iks_union),
                                          n_total_pool=N))

    pd.DataFrame(unique_compound_rows).to_csv("../results/unique_compound_counts_random_factor.csv", index=False)
    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/random_factor_control.csv", index=False)
    enrich_df.to_csv("../results/random_factor_control_enrich.csv", index=False)
    hardneg_df.to_csv("../results/random_factor_control_hardneg.csv", index=False)

    summary = []
    for (target, method), g in auprc_df.groupby(["target", "method"]):
        m, lo, hi = bootstrap_ci(g.auprc.values)
        summary.append(dict(target=target, method=method, metric="auprc", mean=m, ci_lo=lo, ci_hi=hi))
    for (target, method), g in enrich_df.groupby(["target", "method"]):
        m, lo, hi = bootstrap_ci(g.top50_rate.values)
        summary.append(dict(target=target, method=method, metric="top50_area_rate", mean=m, ci_lo=lo, ci_hi=hi))
    for (target, method), g in hardneg_df.groupby(["target", "method"]):
        vals = g.hard_neg_median_pct.dropna().values
        if len(vals):
            m, lo, hi = bootstrap_ci(vals)
            summary.append(dict(target=target, method=method, metric="hard_neg_median_pct", mean=m, ci_lo=lo, ci_hi=hi))
        if target == "epilepsy" and g.proconv_median_pct.notna().any():
            vals2 = g.proconv_median_pct.dropna().values
            m2, lo2, hi2 = bootstrap_ci(vals2)
            summary.append(dict(target=target, method=method, metric="proconv_median_pct", mean=m2, ci_lo=lo2, ci_hi=hi2))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("../results/random_factor_control_summary.csv", index=False)
    print(summary_df.round(3).to_string(index=False))
    print("\nWrote results/random_factor_control.csv, random_factor_control_summary.csv")


if __name__ == "__main__":
    main()
