"""Final iteration on the proposed method (per coordinator review of RESULTS.md section 5's
shrinkage follow-up). That follow-up ruled out "the area factor is just a noisy estimate" as the
explanation for why the implemented method (RUV-2-style: fit "plain" first, SVD its FIXED
negative-control residual once, refit the target once with [chemistry+MF, A]) does not beat naive
area-score subtraction on the bias-diagnostic metrics. RESULTS.md's own structural hypothesis after
that follow-up was: either the area factor doesn't carry enough confound signal beyond what
chemistry+MF already explains, or a genuinely different identification strategy is needed, not a
rank/shrinkage variant of the same SVD-residual mechanism.

This script tests the second possibility, once: Gerard & Stephens (RUV*, arXiv:1705.08393, already
cited in this project's literature review/bibliography as gerard2021ruvstar) distinguish RUV-2
(factor-analyze the RESIDUALS of an already-fit model) from RUV-4 (estimate the unwanted factor and
the effect of interest jointly/iteratively rather than in a strict two-stage pass). We implement a
RUV-4-STYLE joint alternating estimation:

  RUV-2 (already implemented, models.negative_control_area_factor + eval loop in 06):
    1. Fit "plain" per-indication models on Z = [chemistry, free-MF] ONCE (no area factor).
    2. SVD the FIXED train residual Y_nc - P_hat_nc (from that one plain fit) on the
       negative-control columns -> area factor A (never updated again).
    3. Refit k0 with [Z, A], project out A's coefficients at scoring time.

  RUV-4-style joint ALS (this script, ruv4_joint_fit below):
    1. Initialize A = 0.
    2. Repeat a few rounds:
       a. Refit EACH negative-control column's own logistic regression on [Z, A] using the
          CURRENT A (so the NC columns' own fits already account for the area factor, not just Z).
       b. Recompute each NC column's train residual using only the Z-part of that refit (A's
          contribution to that fit projected out -- the same "project out" convention used
          everywhere else in this project), restricted to train rows.
       c. SVD the stacked NC residual matrix -> updated A.
    3. After the rounds converge, refit the TARGET k0's own model on [Z, A] using the final A, and
       score with A's coefficients projected out, exactly as for every other method here.

The key mechanical difference from RUV-2: here the negative controls' own fits and the area factor
estimate update each other over several rounds (A influences what the NC columns' own logistic fits
attribute to Z vs. the shared factor, which changes the residual that re-estimates A), instead of a
single fixed-point-free two-stage pass. This is NOT a retuning of rank/shrinkage (already tried and
already ruled out in 10_shrinkage_negctrl.py) -- it is a different estimating equation for A.

Framing (set by the task, logged here so the result is read in the right spirit): this is the final
iteration on this method regardless of outcome. If RUV-4-style joint estimation ALSO fails to beat
naive subtraction on the bias metrics, that strengthens the existing structural diagnosis (the
failure is not specific to one identification strategy) and is reported exactly that plainly. No
further hyperparameter search follows from this result either way.

Writes:
  results/eval_auprc_ruv4.csv
  results/eval_area_enrichment_ruv4.csv
  results/eval_hard_negatives_ruv4.csv
  results/eval_summary_ruv4.csv          -- means + bootstrap 95% CIs across 5 reruns (seeds)
  results/negctrl_calibration_ruv4.csv / _summary_ruv4.csv  -- same leave-one-NC-out diagnostic
    used throughout this project (section 5 / 10_shrinkage_negctrl.py), applied to the joint
    RUV-4-style estimator instead of the two-stage RUV-2 one.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, fit_with_extra,
                     score_project, raw_area_rate)

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]              # main CVS1/2/3 re-evaluation, same convention as 06/10
CALIB_SEEDS = [0, 1, 2, 3, 4]        # leave-one-NC-out calibration check, same convention as 10
R_FREE, R_CHEM = 16, 32
R_AREA = 2                           # same rank as the original "proposed_negctrl" (section 4)
N_ITERS = 5                          # ALS rounds; a few, not tuned/swept (this is not a retune)
LAMBDA_NAIVE = 2.0


def ruv4_joint_fit(Zrep, Y, train_mask, nc_idx, k0_idx, r_area=R_AREA, n_iters=N_ITERS, C=1.0):
    """RUV-4-style joint alternating estimation of the area factor A and the target's own fit.
    Returns (clf_k0, n_base_k0, A, n_rounds_run). See module docstring for the procedure."""
    N = Zrep.shape[0]
    A = np.zeros((N, r_area))
    if len(nc_idx) == 0:
        train_idx_k0 = np.where(train_mask[:, k0_idx])[0]
        clf_k0, n_base = fit_with_extra(Zrep, A, Y[:, k0_idx], train_idx_k0, C=C)
        return clf_k0, n_base, A, 0
    mask_nc = train_mask[:, nc_idx]
    rounds_run = 0
    for it in range(n_iters):
        R = np.zeros((N, len(nc_idx)))
        any_fit = False
        for jj, j in enumerate(nc_idx):
            tr = np.where(train_mask[:, j])[0]
            clf_j, n_base_j = fit_with_extra(Zrep, A, Y[:, j], tr, C=C)
            if clf_j is None:
                continue
            any_fit = True
            p_hat_j = score_project(clf_j, Zrep, n_base_j)   # Z-only part of the joint fit
            R[:, jj] = Y[:, j] - p_hat_j
        if not any_fit:
            break
        R = R * mask_nc
        Uu, S, Vt = np.linalg.svd(R, full_matrices=False)
        r = min(r_area, Uu.shape[1])
        A_next = Uu[:, :r] * S[:r][None, :]
        if A_next.shape[1] < r_area:
            A_next = np.hstack([A_next, np.zeros((N, r_area - A_next.shape[1]))])
        delta = np.linalg.norm(A_next - A) / (np.linalg.norm(A) + 1e-9)
        A = A_next
        rounds_run += 1
    train_idx_k0 = np.where(train_mask[:, k0_idx])[0]
    clf_k0, n_base_k0 = fit_with_extra(Zrep, A, Y[:, k0_idx], train_idx_k0, C=C)
    return clf_k0, n_base_k0, A, rounds_run


def make_masks(N, K, k0_idx, cvs, seed):
    rng = np.random.default_rng(seed)
    train_mask = np.ones((N, K), dtype=bool)
    if cvs == "CVS1":
        cell_test = rng.random((N, K)) < 0.2
        train_mask[cell_test] = False
        test_rows = np.where(cell_test[:, k0_idx])[0]
    elif cvs == "CVS2":
        row_test = rng.random(N) < 0.2
        train_mask[row_test, :] = False
        test_rows = np.where(row_test)[0]
    elif cvs == "CVS3":
        train_mask[:, k0_idx] = False
        test_rows = np.arange(N)
    else:
        raise ValueError(cvs)
    return train_mask, test_rows


def sider_hit_matrix(cmp, cols):
    sider = pd.read_parquet(D + "sider_full.parquet")
    ik_to_terms = sider.groupby("ik14").se_name.apply(lambda s: set(x.lower() for x in s))
    hit = np.zeros((len(cmp), len(cols)), dtype=bool)
    for j, d in enumerate(cols):
        terms = HARD_NEG_TERMS.get(d)
        if not terms:
            continue
        for i, ik in enumerate(cmp.ik14):
            s = ik_to_terms.get(ik)
            if s and (s & terms):
                hit[i, j] = True
    return hit


def area_enrichment(cmp, scores, y_true, target, test_rows, k=50):
    atc = pd.read_csv(D + "atc_level1.csv")
    area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
    flag = cmp.ik14.isin(area_iks).values
    sub_rows = test_rows[y_true[test_rows] == 0]
    if len(sub_rows) == 0:
        return np.nan, np.nan
    order = sub_rows[np.argsort(-scores[sub_rows])]
    base_rate = flag[sub_rows].mean()
    topk = order[:min(k, len(order))]
    return flag[topk].mean(), base_rate


def hard_neg_rank_pct(scores, test_rows, hard_mask_col):
    rows = test_rows[hard_mask_col[test_rows]]
    if len(rows) == 0:
        return np.nan
    order = np.argsort(-scores[test_rows])
    ranks = np.empty(len(test_rows)); ranks[order] = np.arange(1, len(test_rows) + 1)
    pos_of_test = {r: i for i, r in enumerate(test_rows)}
    return float(np.median([ranks[pos_of_test[r]] / len(test_rows) * 100 for r in rows]))


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    print(f"Drug-pool indication matrix: {N} compounds x {K} indications")
    col_idx = {d: j for j, d in enumerate(cols)}
    hard_hits = sider_hit_matrix(cmp, cols)
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    print("\n=== Part 1: leave-one-negative-control-out calibration check (RUV-4-style) ===")
    calib_rows = []
    for target in TARGETS:
        nc_names = negative_controls(target, diseases=cols)
        for ncj in nc_names:
            calib_set = [d for d in nc_names if d != ncj]
            ncj_idx = col_idx[ncj]
            calib_idx = [col_idx[d] for d in calib_set]
            y_ncj = Y[:, ncj_idx].astype(int)
            if y_ncj.sum() < 5 or len(calib_idx) == 0:
                continue
            for seed in CALIB_SEEDS:
                rng = np.random.default_rng(seed)
                train_mask = np.ones((N, K), dtype=bool)
                row_test = rng.random(N) < 0.2
                train_mask[row_test, :] = False
                test_rows = np.where(row_test)[0]
                if y_ncj[test_rows].sum() == 0 or y_ncj[test_rows].sum() == len(test_rows):
                    continue
                Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
                Zrep = np.hstack([Xchem, Ufree])
                clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
                if clfs[ncj_idx] is None:
                    continue
                s_plain = score_project(clfs[ncj_idx], Zrep, Zrep.shape[1])
                clf_db, n_base, A, rounds_run = ruv4_joint_fit(Zrep, Y, train_mask, calib_idx, ncj_idx)
                s_db = score_project(clf_db, Zrep, n_base)
                ap_plain = average_precision_score(y_ncj[test_rows], s_plain[test_rows])
                ap_db = average_precision_score(y_ncj[test_rows], s_db[test_rows])
                calib_rows.append(dict(target=target, calibration_indication=ncj, seed=seed,
                                       auprc_plain=ap_plain, auprc_corrected=ap_db,
                                       delta=ap_db - ap_plain, rounds_run=rounds_run))
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv("../results/negctrl_calibration_ruv4.csv", index=False)
    calib_summ = calib_df.groupby(["target", "calibration_indication"]).delta.agg(
        ["mean", "std", "count"]).reset_index()
    calib_summ.to_csv("../results/negctrl_calibration_summary_ruv4.csv", index=False)
    print(calib_summ.to_string(index=False))

    print("\n=== Part 2: CVS1/CVS2/CVS3 re-evaluation (plain, naive_subtract, proposed_ruv4) ===")
    auprc_rows, enrich_rows, hardneg_rows = [], [], []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_names = negative_controls(target, diseases=cols)
        nc_idx = [col_idx[d] for d in nc_names]
        print(f"\n### target={target}  negative controls={nc_names}")
        hard_col = hard_hits[:, k0]
        y_full = Y[:, k0].astype(int)

        for cvs in ["CVS1", "CVS2", "CVS3"]:
            for seed in SEEDS:
                train_mask, test_rows = make_masks(N, K, k0, cvs, seed)
                if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                    continue
                Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
                Zrep = np.hstack([Xchem, Ufree])

                clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
                clf_plain = clfs[k0]
                cvs3_degenerate = clf_plain is None
                if cvs3_degenerate:
                    s_plain = P_hat[:, nc_idx].mean(1) if len(nc_idx) else np.full(N, y_full.mean())
                else:
                    s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

                area_rate = raw_area_rate(Y, nc_idx)
                ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
                s_naive = s_plain - LAMBDA_NAIVE * ar

                if len(nc_idx) > 0 and not cvs3_degenerate:
                    clf_ruv4, n_base, A, rounds_run = ruv4_joint_fit(Zrep, Y, train_mask, nc_idx, k0)
                    s_ruv4 = score_project(clf_ruv4, Zrep, n_base)
                else:
                    # Same structural degeneracy as the original proposed method (RESULTS.md
                    # section 4 / Limitations item 6): with zero target labels there is nothing for
                    # ANY target-specific estimator -- RUV-2 or RUV-4 -- to debias, so this
                    # provably collapses to the same cross-indication transfer score as "plain".
                    s_ruv4 = s_plain.copy()

                for name, s in [("plain", s_plain), ("naive_subtract", s_naive),
                                 ("proposed_ruv4", s_ruv4)]:
                    yt = y_full[test_rows]
                    if yt.sum() == 0 or yt.sum() == len(yt):
                        continue
                    ap = average_precision_score(yt, s[test_rows])
                    auprc_rows.append(dict(target=target, cvs=cvs, method=name, seed=seed, auprc=ap,
                                           n_test=len(test_rows), n_pos=int(yt.sum())))
                    top_rate, base_rate = area_enrichment(cmp, s, y_full, target, test_rows)
                    enrich_rows.append(dict(target=target, cvs=cvs, method=name, seed=seed,
                                            top50_rate=top_rate, base_rate=base_rate))
                    hn_pct = hard_neg_rank_pct(s, test_rows, hard_col)
                    row = dict(target=target, cvs=cvs, method=name, seed=seed,
                               hard_neg_median_pct=hn_pct, n_hard_neg=int(hard_col[test_rows].sum()),
                               proconv_median_pct=np.nan, n_proconv=0)
                    if target == "epilepsy":
                        row["proconv_median_pct"] = hard_neg_rank_pct(s, test_rows, proconv_mask_all)
                        row["n_proconv"] = int(proconv_mask_all[test_rows].sum())
                    hardneg_rows.append(row)
            print(f"  {cvs} done")

    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/eval_auprc_ruv4.csv", index=False)
    enrich_df.to_csv("../results/eval_area_enrichment_ruv4.csv", index=False)
    hardneg_df.to_csv("../results/eval_hard_negatives_ruv4.csv", index=False)

    summary = []
    for (target, cvs, method), g in auprc_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.auprc.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="auprc",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=len(g)))
    for (target, cvs, method), g in enrich_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.top50_rate.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="top50_area_rate",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=len(g)))
        m2, lo2, hi2 = bootstrap_ci(g.base_rate.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="area_base_rate",
                            mean=m2, ci_lo=lo2, ci_hi=hi2, n_seeds=len(g)))
    for (target, cvs, method), g in hardneg_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.hard_neg_median_pct.dropna().values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="hard_neg_median_pct",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=g.hard_neg_median_pct.notna().sum()))
        if target == "epilepsy" and g.proconv_median_pct.notna().any():
            m3, lo3, hi3 = bootstrap_ci(g.proconv_median_pct.dropna().values)
            summary.append(dict(target=target, cvs=cvs, method=method, metric="proconv_median_pct",
                                mean=m3, ci_lo=lo3, ci_hi=hi3, n_seeds=g.proconv_median_pct.notna().sum()))
    pd.DataFrame(summary).to_csv("../results/eval_summary_ruv4.csv", index=False)
    print("\nWrote results/eval_*_ruv4.csv, eval_summary_ruv4.csv, "
          "negctrl_calibration_ruv4.csv/_summary_ruv4.csv")


if __name__ == "__main__":
    main()
