"""W2a (reviewer feedback round 3): the synthetic ground-truth experiment (code/20_synthetic_groundtruth.py)
already showed "RUV-factor direct subtraction" (calibrate a coefficient on the already-estimated
area factor, subtract it straight from the plain score) outperforms "proposed" (refit jointly with
the factor, then drop it) at recovering the TRUE confound. That comparison has never been run on
real data, where there is no ground truth, only the same bias-diagnostic metrics as Table main-bias.
This runs direct subtraction as a fifth scoring method, same CVS2 protocol, same 5 targets/seeds as
code/06_run_evaluation.py, directly comparable to plain/naive/proposed.

Design note (logged honestly, not silently matching an inconsistency): the synthetic script fits
its calibration logistic regression on the full area factor A_oos (r_area=2 columns) but then only
subtracts the first column's coefficient (coef_direct = calib.coef_[0][0], not the full coefficient
vector), leaving the second component's fitted coefficient unused. That looks like an oversight
rather than a deliberate choice -- there is no reason given in that script for discarding one
dimension of a jointly-fit calibration. This script uses the full fitted coefficient vector instead
(subtracts A_oos @ calib.coef_[0], not A_oos[:, 0] * calib.coef_[0][0]), which is the more defensible
reading of "subtract the calibrated factor directly." Numbers here are therefore not expected to be
bit-identical to a real-data version of the synthetic script's exact (likely buggy) mechanism.

Writes: results/direct_subtraction_real_data.csv, results/direct_subtraction_real_data_summary.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, project_area_factor_oos, sigmoid)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
LAMBDA_NAIVE = 2.0


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
            s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)

            ytr = y_full[train_idx_k0]
            if ytr.sum() < 2 or ytr.sum() == len(ytr):
                continue
            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                A_oos[train_idx_k0], ytr)
            logit_plain = clf_plain.decision_function(Zrep)
            s_direct = sigmoid(logit_plain - A_oos @ calib.coef_[0])

            for name, s in [("direct_subtract", s_direct)]:
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

    pd.DataFrame(unique_compound_rows).to_csv("../results/unique_compound_counts_direct_subtract.csv", index=False)
    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/direct_subtraction_real_data.csv", index=False)
    enrich_df.to_csv("../results/direct_subtraction_real_data_enrich.csv", index=False)
    hardneg_df.to_csv("../results/direct_subtraction_real_data_hardneg.csv", index=False)

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
    summary_df.to_csv("../results/direct_subtraction_real_data_summary.csv", index=False)
    print(summary_df.round(3).to_string(index=False))
    print("\nWrote results/direct_subtraction_real_data.csv, direct_subtraction_real_data_summary.csv")


if __name__ == "__main__":
    main()
