"""Direct intervention test for the alignment check (Critical #1 in assessment.txt): does a
bias-diagnostic metric actually respond when a correction's real effect is dialed up, rather than
inferring "the metric can't see this mechanism" from a standalone AUROC near chance?

Standalone AUROC(penalty, flag) asks whether the penalty term alone discriminates the flag. It
does not directly show whether ADDING that penalty into the score moves the metric that is
actually reported (Table "main-bias": top-50 same-ATC-area rate, hard-negative/proconvulsant rank
percentile). This script measures that directly: for both corrections, sweep how much of the
penalty is actually included in the score, alpha in {0, 0.25, 0.5, 0.75, 1.0}, and recompute the
real bias metrics at each dose, on the same CVS2 split Table "main-bias" uses.

  naive:    s(alpha) = s_plain - alpha * LAMBDA_NAIVE * normalized_area_rate   (alpha=1 is Table 2's
            reported "naive subtract" row; alpha=0 is plain)
  proposed: s(alpha) = sigmoid(Zrep @ coef_base + alpha * A_oos @ coef_A + intercept)   (alpha=0 is
            Table 2's reported "proposed" row, A fully projected out; alpha=1 is the undropped fit,
            i.e. NOT debiased at all)

A_oos (models.project_area_factor_oos) is used for test rows, since 06_run_evaluation.py's
score_project path for the proposed method never needed a test-row A value (it always drops A's
coefficients entirely, alpha=0 in this script's terms) -- this script is the first place a
non-zero A is actually applied to test-row scores, so it is the first place the test-row
out-of-sample estimate (added in code/models.py this round) is load-bearing for anything beyond a
standalone AUROC diagnostic.

If the proposed method's bias metrics move smoothly with alpha the same way naive's do, the metric
DOES respond to this mechanism (Table 3's near-chance standalone AUROC, where it holds, was not
evidence of blindness). If they stay flat while naive's visibly move, that is the direct,
intervention-based version of the alignment claim the standalone-AUROC proxy only gestured at.

Writes: results/intervention_doseresponse.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, raw_area_rate, project_area_factor_oos, sigmoid, score_project)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
LAMBDA_NAIVE = 2.0
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


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
    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None or len(nc_idx) == 0:
                continue
            s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

            area_rate = raw_area_rate(Y, nc_idx)
            ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            if clf_db is None:
                continue
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            coef_base = clf_db.coef_[0][:n_base]
            coef_A = clf_db.coef_[0][n_base:]
            base_logit = Zrep @ coef_base + clf_db.intercept_[0]
            area_logit = A_oos @ coef_A

            for alpha in ALPHAS:
                s_naive_dose = s_plain - alpha * LAMBDA_NAIVE * ar
                s_proposed_dose = sigmoid(base_logit + alpha * area_logit)

                for method, s in [("naive", s_naive_dose), ("proposed", s_proposed_dose)]:
                    area_rate_top50, base_rate = run_eval.area_enrichment(cmp, s, y_full, target, test_rows)
                    hardneg_pct = run_eval.hard_neg_rank_pct(s, test_rows, hard_col)
                    row = dict(target=target, seed=seed, method=method, alpha=alpha,
                               area_rate_top50=area_rate_top50, area_base_rate=base_rate,
                               hardneg_pct=hardneg_pct)
                    if target == "epilepsy":
                        row["proconv_pct"] = run_eval.hard_neg_rank_pct(s, test_rows, proconv_mask_all)
                    rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../results/intervention_doseresponse.csv", index=False)
    print(out.groupby(["target", "method", "alpha"])[
        [c for c in ["area_rate_top50", "hardneg_pct", "proconv_pct"] if c in out.columns]
    ].mean().round(3))
    print("\nWrote results/intervention_doseresponse.csv")


if __name__ == "__main__":
    main()
