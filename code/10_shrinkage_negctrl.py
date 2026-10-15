"""Follow-up per coordinator review: the raw RUV-2 area factor (Section-4 "proposed_negctrl") is
estimated from tiny negative-control classes (4-21 positives) and is probably mostly noise
(RESULTS.md Limitations item 1). Gerard & Stephens (RUV*/RUVB, arXiv:1705.08393) make exactly this
point: when negative controls are few/noisy, the estimated unwanted-variation factor should be
shrunk toward zero (empirical-Bayes / ridge-style) rather than used raw.

Simplified shrinkage implemented here: the area factor A_raw (top r_area left singular vectors of
the negative-control residual) is scaled by a shrinkage weight s in [0,1] BEFORE being used,
jointly with chemistry+free-MF, to fit the target's own logistic regression. At s=0 this reduces
exactly to the "plain" baseline (A contributes nothing, so the fit cannot use it to re-attribute
any of v_k0's variance); at s=1 it is the original unshrunk "proposed_negctrl". Because the score
reported for ranking ALWAYS drops A's own coefficients ("project out"), shrinkage acts only on how
much variance the FIT is allowed to attribute to A instead of to v_k0 during estimation -- a
simplified form of asymmetric ridge shrinkage on the RUV nuisance term (reducing a feature's scale
under a fixed L2 penalty increases its effective relative penalty, so it absorbs less of the
target's own variance).

(r_area, s) is selected PER TARGET using the leave-one-negative-control-out self-calibration
diagnostic already built in 07_negctrl_calibration.py: for each candidate (r_area, s), compute the
calibration delta (AUPRC corrected-vs-plain) for every ncj in NC(target) using an area factor
estimated from NC(target) \\ {ncj} at that (r_area, s); if the negative controls are genuinely
independent of one another, the best (r_area, s) is the one whose MEAN ABSOLUTE calibration delta
is smallest (closest to the "no spurious correction of mechanistically-unrelated indications"
ideal) -- this never looks at the real target's own labels, so it stays a legitimate,
leakage-free negative-control-only model-selection step.

The chosen (r_area, s) per target is then applied to a full CVS1/CVS2/CVS3 re-run (method name
"proposed_shrunk"), written to results/eval_*_shrunk.csv, for direct comparison against the
original (unshrunk) "proposed_negctrl" numbers already in results/eval_summary.csv.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, nrbdmf_signed_matrix)

TARGETS = ["depression", "antipsychotic", "epilepsy"]
CALIB_SEEDS = [0, 1, 2, 3, 4]        # for the (r_area, s) grid search (negative-control-only)
EVAL_SEEDS = [0, 1, 2, 3, 4]         # for the final CVS1/2/3 re-run
R_FREE, R_CHEM = 16, 32
R_AREA_GRID = [1, 2, 3]
SHRINK_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def select_best_shrinkage(Y, cols, col_idx, Zrep, target):
    """Grid search over (r_area, shrink), reusing the expensive shared-MF fit and per-indication
    plain fit across the whole grid for a given (ncj, seed) -- only the cheap SVD-of-residual and
    final logistic refit change per grid point, so this is ~15x faster than refitting from
    scratch for every (r_area, shrink) pair."""
    N, K = Y.shape
    nc_names = negative_controls(target, diseases=cols)
    deltas = {(r, s): [] for r in R_AREA_GRID for s in SHRINK_GRID}
    max_r = max(R_AREA_GRID)

    for ncj in nc_names:
        calib_idx_full = [col_idx[d] for d in nc_names if d != ncj]
        ncj_idx = col_idx[ncj]
        y_ncj = Y[:, ncj_idx].astype(int)
        if len(calib_idx_full) == 0:
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
            Zr = np.hstack([Zrep, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zr, Y, train_mask)
            if clfs[ncj_idx] is None:
                continue
            s_plain = score_project(clfs[ncj_idx], Zr, Zr.shape[1])
            ap_plain = average_precision_score(y_ncj[test_rows], s_plain[test_rows])
            train_idx = np.where(train_mask[:, ncj_idx])[0]
            # SVD once at the maximum rank, slice down for smaller r_area (same left-singular
            # vectors by construction, just truncated).
            A_raw_full = negative_control_area_factor(Y, P_hat, calib_idx_full, train_mask, r_area=max_r)
            for r_area in R_AREA_GRID:
                A_raw = A_raw_full[:, :r_area]
                for shrink in SHRINK_GRID:
                    A = shrink * A_raw
                    clf_db, n_base = fit_with_extra(Zr, A, y_ncj, train_idx)
                    s_debiased = score_project(clf_db, Zr, n_base)
                    ap_db = average_precision_score(y_ncj[test_rows], s_debiased[test_rows])
                    deltas[(r_area, shrink)].append(ap_db - ap_plain)

    rows = []
    for (r_area, shrink), vals in deltas.items():
        if not vals:
            continue
        vals = np.array(vals)
        se = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(dict(target=target, r_area=r_area, shrink=shrink,
                         mean_abs_delta=float(np.mean(np.abs(vals))),
                         mean_delta=float(np.mean(vals)), se_delta=se, n=len(vals)))
    grid = pd.DataFrame(rows)

    # IMPORTANT: shrink=0.0 makes A identically zero, so clf_db is mathematically IDENTICAL to
    # clf_plain and mean_abs_delta is EXACTLY 0 by construction for every r_area -- a trivial,
    # vacuous "pass" (it is just "apply no correction at all"), not evidence that some correction
    # is safe. Naively minimizing mean_abs_delta over the full grid would always select this
    # trivial point and never actually test the method. We therefore select among the NONTRIVIAL
    # shrink values only (candidates with calibration no worse than 1 standard error above the
    # best nontrivial calibration at that r_area), and report the trivial shrink=0 row separately
    # as a reference floor, not a candidate.
    nontrivial = grid[grid.shrink > 0].copy()
    floor = nontrivial.mean_abs_delta.min()
    floor_se = nontrivial.loc[nontrivial.mean_abs_delta.idxmin(), "se_delta"]
    tol = floor + (floor_se if not np.isnan(floor_se) else 0.0)
    acceptable = nontrivial[nontrivial.mean_abs_delta <= tol]
    # Among acceptably-calibrated nontrivial options, prefer the LARGEST shrink (closest to the
    # full unshrunk RUV2 correction) -- shrinkage should only back off from 1.0 as far as the
    # calibration data forces it to, not further.
    best = acceptable.sort_values(["shrink", "mean_abs_delta"], ascending=[False, True]).iloc[0]
    return best, grid


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
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    hard_hits = sider_hit_matrix(cmp, cols)
    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_mask_all = cmp.ik14.isin(set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())).values

    print("=== Step 1: grid search (r_area, shrink) per target via leave-one-NC-out calibration ===")
    best_cfg = {}
    all_grids = []
    for target in TARGETS:
        best, grid = select_best_shrinkage(Y, cols, col_idx, Xchem, target)
        best_cfg[target] = (int(best.r_area), float(best.shrink))
        all_grids.append(grid)
        print(f"{target}: best r_area={int(best.r_area)}, shrink={best.shrink:.2f}, "
              f"mean|delta|={best.mean_abs_delta:.4f} (n={int(best.n)} calibration cells)")
    pd.concat(all_grids, ignore_index=True).to_csv("../results/shrinkage_grid_search.csv", index=False)

    print("\n=== Step 2: apply chosen (r_area, shrink) to the real CVS1/CVS2/CVS3 evaluation ===")
    auprc_rows, enrich_rows, hardneg_rows = [], [], []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        r_area, shrink = best_cfg[target]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]
        for cvs in ["CVS1", "CVS2", "CVS3"]:
            for seed in EVAL_SEEDS:
                train_mask, test_rows = make_masks(N, K, k0, cvs, seed)
                if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                    continue
                Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
                Zrep = np.hstack([Xchem, Ufree])
                clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
                degenerate = clfs[k0] is None
                if degenerate:
                    s_shrunk = P_hat[:, nc_idx].mean(1) if len(nc_idx) else np.full(N, y_full.mean())
                else:
                    A_raw = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=r_area)
                    A = shrink * A_raw
                    train_idx_k0 = np.where(train_mask[:, k0])[0]
                    clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
                    s_shrunk = score_project(clf_db, Zrep, n_base)
                yt = y_full[test_rows]
                if yt.sum() == 0 or yt.sum() == len(yt):
                    continue
                ap = average_precision_score(yt, s_shrunk[test_rows])
                auprc_rows.append(dict(target=target, cvs=cvs, method="proposed_shrunk", seed=seed,
                                       auprc=ap, r_area=r_area, shrink=shrink))
                top_rate, base_rate = area_enrichment(cmp, s_shrunk, y_full, target, test_rows)
                enrich_rows.append(dict(target=target, cvs=cvs, method="proposed_shrunk", seed=seed,
                                        top50_rate=top_rate, base_rate=base_rate))
                hn = hard_neg_rank_pct(s_shrunk, test_rows, hard_col)
                row = dict(target=target, cvs=cvs, method="proposed_shrunk", seed=seed,
                           hard_neg_median_pct=hn, proconv_median_pct=np.nan)
                if target == "epilepsy":
                    row["proconv_median_pct"] = hard_neg_rank_pct(s_shrunk, test_rows, proconv_mask_all)
                hardneg_rows.append(row)
            print(f"  {target} {cvs} done")

    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/eval_auprc_shrunk.csv", index=False)
    enrich_df.to_csv("../results/eval_area_enrichment_shrunk.csv", index=False)
    hardneg_df.to_csv("../results/eval_hard_negatives_shrunk.csv", index=False)

    summary = []
    for (target, cvs), g in auprc_df.groupby(["target", "cvs"]):
        m, lo, hi = bootstrap_ci(g.auprc.values)
        summary.append(dict(target=target, cvs=cvs, method="proposed_shrunk", metric="auprc",
                            mean=m, ci_lo=lo, ci_hi=hi))
    for (target, cvs), g in enrich_df.groupby(["target", "cvs"]):
        m, lo, hi = bootstrap_ci(g.top50_rate.values)
        summary.append(dict(target=target, cvs=cvs, method="proposed_shrunk", metric="top50_area_rate",
                            mean=m, ci_lo=lo, ci_hi=hi))
    for (target, cvs), g in hardneg_df.groupby(["target", "cvs"]):
        m, lo, hi = bootstrap_ci(g.hard_neg_median_pct.dropna().values)
        summary.append(dict(target=target, cvs=cvs, method="proposed_shrunk", metric="hard_neg_median_pct",
                            mean=m, ci_lo=lo, ci_hi=hi))
        if target == "epilepsy" and g.proconv_median_pct.notna().any():
            m2, lo2, hi2 = bootstrap_ci(g.proconv_median_pct.dropna().values)
            summary.append(dict(target=target, cvs=cvs, method="proposed_shrunk", metric="proconv_median_pct",
                                mean=m2, ci_lo=lo2, ci_hi=hi2))
    pd.DataFrame(summary).to_csv("../results/eval_summary_shrunk.csv", index=False)
    print("\nWrote results/eval_*_shrunk.csv and results/shrinkage_grid_search.csv")


if __name__ == "__main__":
    main()
