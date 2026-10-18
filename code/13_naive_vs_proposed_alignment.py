"""Tests the one candidate explanation flagged but not tested in Section 4.3 (overlap diagnostic)
and left open in the Limitations: that the bias-diagnostic metrics (top-50 same-ATC-area rate,
hard-negative/proconvulsant rank) are structurally closer to what naive subtraction directly
penalizes (raw negative-control indication membership) than to what the proposed method targets
(a residualized area factor A, estimated after chemistry and matrix-factorization signal are
already regressed out).

This is close to definitional rather than a deep empirical question, but it has not actually been
measured, so we measure it rather than assert it. For each target and the CVS2 split used in
Table "main-bias", using the EXACT SAME fitted objects 06_run_evaluation.py and
12_signal_overlap_diagnostic.py use (fit_shared_mf, chem_pca, fit_per_indication_logreg,
negative_control_area_factor, fit_with_extra), we compute two quantities per non-indicated test
compound:

  raw_rate       = raw_area_rate(Y, nc_idx)                         -- naive subtraction's penalty
  area_penalty   = A_oos @ clf_db.coef_[0][n_base:]                 -- proposed method's penalty,
                   i.e. the contribution the dropped area-factor coefficients WOULD have made to
                   the score, had they not been projected out at scoring time

Correction (logged honestly, not silently): an earlier version of this script used the raw `A`
returned by negative_control_area_factor directly on test rows. That function zeroes a test row's
residual before the SVD (it only has train-row labels to compute a residual from), which forces
that row's own factor value to be exactly zero by construction -- a zero row into an SVD forces a
zero row out, confirmed numerically (max|A[test_rows]| ~1e-16 vs up to 1.38 on train rows). The
"proposed method's penalty term" was therefore being measured as the AUROC of a value that is
mathematically zero for every single test compound, regardless of its chemistry or true
organ-system membership: a tautological "no information" result, not evidence about the method.
`models.project_area_factor_oos` fixes this: it fits a linear map from the shared representation Z
to the train rows' own (non-degenerate) factor values and uses it to give every row an actual
out-of-sample estimate. `area_penalty` below uses this corrected, out-of-sample `A_oos`.

We measure how well each penalty term separates compounds the bias metrics actually flag:
same-ATC-area membership (the area_enrichment criterion) and SIDER-hard-negative / proconvulsant
membership (the hard_neg_rank_pct criterion), via AUROC of each penalty term as a standalone
classifier of that flag.

Writes: results/naive_vs_proposed_alignment.csv (plus the original, zero-masked `area_penalty_raw`
columns, kept alongside the corrected ones for direct before/after comparison)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from common import D
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, raw_area_rate, project_area_factor_oos)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
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
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            if clfs[k0] is None or len(nc_idx) == 0:
                continue

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            if clf_db is None:
                continue
            coef_A = clf_db.coef_[0][n_base:]
            area_penalty_raw = A @ coef_A  # pre-fix: exactly ~0 on every test row, see module docstring
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            area_penalty = A_oos @ coef_A  # fixed: genuine out-of-sample estimate on test rows
            raw_rate = raw_area_rate(Y, nc_idx)

            cand = test_rows[y_full[test_rows] == 0]  # non-indicated candidates, matches area_enrichment
            if len(cand) == 0:
                continue

            rows.append(dict(
                target=target, seed=seed, n_cand=len(cand),
                auroc_rawrate_samearea=safe_auroc(same_area_flag[cand], raw_rate[cand]),
                auroc_areapenalty_samearea=safe_auroc(same_area_flag[cand], area_penalty[cand]),
                auroc_areapenalty_raw_samearea=safe_auroc(same_area_flag[cand], area_penalty_raw[cand]),
                auroc_rawrate_hardneg=safe_auroc(hard_col[cand], raw_rate[cand]),
                auroc_areapenalty_hardneg=safe_auroc(hard_col[cand], area_penalty[cand]),
                auroc_areapenalty_raw_hardneg=safe_auroc(hard_col[cand], area_penalty_raw[cand]),
            ))
            if target == "epilepsy":
                rows[-1]["auroc_rawrate_proconv"] = safe_auroc(proconv_mask_all[cand], raw_rate[cand])
                rows[-1]["auroc_areapenalty_proconv"] = safe_auroc(proconv_mask_all[cand], area_penalty[cand])
                rows[-1]["auroc_areapenalty_raw_proconv"] = safe_auroc(proconv_mask_all[cand], area_penalty_raw[cand])

    out = pd.DataFrame(rows)
    out.to_csv("../results/naive_vs_proposed_alignment.csv", index=False)

    print("Mean AUROC (5 seeds), each penalty term as a standalone classifier of the bias metric's own flag:")
    summary = out.groupby("target")[[c for c in out.columns if c.startswith("auroc_")]].mean()
    print(summary.round(3))
    print("\n0.5 = the penalty term carries no information about that flag. Higher = better separation.")
    print("Wrote results/naive_vs_proposed_alignment.csv")


if __name__ == "__main__":
    main()
