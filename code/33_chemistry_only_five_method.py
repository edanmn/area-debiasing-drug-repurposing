"""Reviewer roadmap Gap 1 (chemistry-only as primary representation): code/24_chemistry_only_ablation.py
already reran plain/naive_subtract/proposed_negctrl under Zrep=Xchem only (no free-MF embedding,
which carries no information for a held-out CVS2 compound -- see that script's docstring). This
extends the same chemistry-only representation to the two newer methods from the prior reviewer
round, direct_subtract (code/31) and random_factor_control's both mechanisms (code/32), so all five
methods are comparable under the representation the reviewer wants as primary.

Reuses models.py fitting functions directly (fit_per_indication_logreg, negative_control_area_factor,
fit_with_extra, score_project, project_area_factor_oos, sigmoid) rather than reimplementing, so this
cannot silently diverge from 06/24/31/32's own protocol. Same CVS2-only scope as 24 (the only split
where the chemistry-only question applies), same 5 targets, same 5 seeds.

Writes: results/chemistry_only_five_method.csv, results/chemistry_only_five_method_enrich.csv,
        results/chemistry_only_five_method_hardneg.csv, results/chemistry_only_five_method_summary.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, project_area_factor_oos, sigmoid)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_CHEM = 32
LAMBDA_NAIVE = 2.0


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)  # chemistry only, no fit_shared_mf/Ufree

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    auprc_rows, enrich_rows, hardneg_rows = [], [], []

    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        hard_col = hard_hits[:, k0]
        y_full = Y[:, k0].astype(int)

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Zrep = Xchem  # the one change from 06/31/32: no free-MF concatenation

            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None or len(nc_idx) == 0:
                continue
            s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

            area_rate = raw_area_rate(Y, nc_idx)
            ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
            s_naive = s_plain - LAMBDA_NAIVE * ar

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            ytr = y_full[train_idx_k0]
            if ytr.sum() < 2 or ytr.sum() == len(ytr):
                continue

            # proposed: refit jointly with A, drop A's coefficients at scoring
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            s_proposed = score_project(clf_db, Zrep, n_base) if clf_db is not None else s_plain.copy()

            # direct_subtract: calibrate on A_oos (train-fold only), subtract from plain logit
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            calib_real = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                A_oos[train_idx_k0], ytr)
            logit_plain = clf_plain.decision_function(Zrep)
            s_direct = sigmoid(logit_plain - A_oos @ calib_real.coef_[0])

            # matched random-factor control, same shape/magnitude as A, both mechanisms
            col_std = A.std(0)
            rng = np.random.default_rng(10_000 + seed * 37 + k0)
            R = rng.normal(0.0, 1.0, size=A.shape) * col_std[None, :]
            clf_rand, n_base_r = fit_with_extra(Zrep, R, y_full, train_idx_k0)
            s_random_refit = score_project(clf_rand, Zrep, n_base_r) if clf_rand is not None else None
            calib_rand = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                R[train_idx_k0], ytr)
            s_random_direct = sigmoid(logit_plain - R @ calib_rand.coef_[0])

            methods = [("plain", s_plain), ("naive_subtract", s_naive),
                       ("proposed_negctrl", s_proposed), ("direct_subtract", s_direct),
                       ("random_direct_subtract", s_random_direct)]
            if s_random_refit is not None:
                methods.append(("random_refit_drop", s_random_refit))

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

    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/chemistry_only_five_method.csv", index=False)
    enrich_df.to_csv("../results/chemistry_only_five_method_enrich.csv", index=False)
    hardneg_df.to_csv("../results/chemistry_only_five_method_hardneg.csv", index=False)

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
    summary_df.to_csv("../results/chemistry_only_five_method_summary.csv", index=False)
    print(summary_df.round(3).to_string(index=False))
    print("\nWrote results/chemistry_only_five_method*.csv")


if __name__ == "__main__":
    main()
