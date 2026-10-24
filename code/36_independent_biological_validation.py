"""Priority 3 (reviewer roadmap round 4): independent biological validation for epilepsy.

Step 0 (done by inspection, not code): the 43-drug proconvulsant panel (data/proconvulsants.csv,
built by code/04_build_proconvulsants.py from Chen/Albertson/Olson 2016, cross-validated against
Ruffmann/Bogliun/Beghi 2006) is used ONLY in code/06_run_evaluation.py as an evaluation-time
bias-diagnostic readout (proconv_mask_all fed to hard_neg_rank_pct). grep confirms it never appears
in models.py or indication_matrix.py -- it plays no role in negative_control_area_factor, fit_with_extra,
fit_shared_mf, chem_pca, or lambda calibration (LAMBDA_NAIVE=2.0 is a fixed constant, not tuned on
any held-out label). It is therefore already a clean, held-out evidence source for epilepsy; no new
curation was needed (Step 1 skipped as a result -- see fork report for the WebSearch attempt at an
independent second source, which did not find one distinct enough to be worth adding).

This script (Step 2) reports precision@10/@50 for recovering genuine antiepileptic positives, the
proconvulsant rate in the same top-10/top-50, and bemegride's rank specifically, for 6 scoring
variants on epilepsy/CVS2/5 seeds: plain, naive_subtract, proposed_negctrl (refit-drop),
direct_subtract, random_refit_drop, random_direct_subtract.

Step 3: depression/antipsychotic have no equivalently independent held-out list (their only
candidate hard-negative source IS SIDER, already used as the paper's hard-negative metric itself,
so using it again here would not be independent) -- skipped, as instructed, rather than forcing a
weaker analysis.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from common import D
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, project_area_factor_oos, sigmoid)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGET = "epilepsy"
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
LAMBDA_NAIVE = 2.0
BEMEGRIDE_IK = "ORRZGUBHBVWWOP"


def precision_at_k(scores, test_rows, y_true, k):
    order = test_rows[np.argsort(-scores[test_rows])]
    topk = order[:min(k, len(order))]
    return float(y_true[topk].mean()), topk


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask = cmp.ik14.isin(proconv_iks).values
    beme_rows = np.where(cmp.ik14.values == BEMEGRIDE_IK)[0]
    beme_row = int(beme_rows[0]) if len(beme_rows) else None
    print(f"Proconvulsant panel in pool: {proconv_mask.sum()}; bemegride found: {beme_row is not None}")

    k0 = col_idx[TARGET]
    nc_idx = [col_idx[d] for d in negative_controls(TARGET, diseases=cols)]
    y_full = Y[:, k0].astype(int)

    rows = []
    for seed in SEEDS:
        train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
        if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
            continue
        Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
        Zrep = np.hstack([Xchem, Ufree])
        clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
        clf_plain = clfs[k0]
        s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

        area_rate = raw_area_rate(Y, nc_idx)
        ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
        s_naive = s_plain - LAMBDA_NAIVE * ar

        A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
        train_idx_k0 = np.where(train_mask[:, k0])[0]
        clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
        s_proposed = score_project(clf_db, Zrep, n_base)

        A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
        ytr = y_full[train_idx_k0]
        logit_plain = clf_plain.decision_function(Zrep)
        s_direct = s_naive.copy()
        if ytr.sum() >= 2 and ytr.sum() < len(ytr):
            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                A_oos[train_idx_k0], ytr)
            s_direct = sigmoid(logit_plain - A_oos @ calib.coef_[0])

        rng = np.random.RandomState(1000 + seed)
        R = rng.normal(0, A.std(), size=A.shape)
        clf_rdb, n_base_r = fit_with_extra(Zrep, R, y_full, train_idx_k0)
        s_rand_refit = score_project(clf_rdb, Zrep, n_base_r)
        s_rand_direct = s_naive.copy()
        if ytr.sum() >= 2 and ytr.sum() < len(ytr):
            calib_r = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                R[train_idx_k0], ytr)
            s_rand_direct = sigmoid(logit_plain - R @ calib_r.coef_[0])

        for name, s in [("plain", s_plain), ("naive_subtract", s_naive), ("proposed_negctrl", s_proposed),
                        ("direct_subtract", s_direct), ("random_refit_drop", s_rand_refit),
                        ("random_direct_subtract", s_rand_direct)]:
            p10, top10 = precision_at_k(s, test_rows, y_full, 10)
            p50, top50 = precision_at_k(s, test_rows, y_full, 50)
            proconv10 = float(proconv_mask[top10].mean()) if len(top10) else np.nan
            proconv50 = float(proconv_mask[top50].mean()) if len(top50) else np.nan
            beme_rank, beme_pct = np.nan, np.nan
            if beme_row is not None and beme_row in test_rows:
                order = test_rows[np.argsort(-s[test_rows])]
                beme_rank = int(np.where(order == beme_row)[0][0]) + 1
                beme_pct = beme_rank / len(order) * 100
            rows.append(dict(target=TARGET, seed=seed, method=name, precision_at_10=p10, precision_at_50=p50,
                             proconv_rate_at_10=proconv10, proconv_rate_at_50=proconv50,
                             n_test=len(test_rows), n_pos=int(y_full[test_rows].sum()),
                             bemegride_rank=beme_rank, bemegride_pct=beme_pct))

    df = pd.DataFrame(rows)
    df.to_csv("../results/independent_bio_validation.csv", index=False)
    summary = df.groupby("method")[["precision_at_10", "precision_at_50", "proconv_rate_at_10",
                                     "proconv_rate_at_50", "bemegride_rank", "bemegride_pct"]].mean()
    print(summary.round(3).to_string())
    print("\nWrote results/independent_bio_validation.csv")


if __name__ == "__main__":
    main()
