"""Mechanistic diagnosis of why naive subtraction beats the negative-control correction, grounded
in the method's own cited identifiability condition rather than left as an unexplained empirical
gap. Gerard & Stephens (2021, already cited as gerard2021ruvstar; see their unification of the RUV
family, eq. 8-9 in the arXiv preprint 1705.08393) formalize the negative-control assumption as
beta_{i,control}=0 for every covariate i of the factor of interest: if a negative-control column's
residual is NOT actually free of the target disease's own signal, the area factor A estimated from
it will absorb some of that signal, and projecting A out (the "proposed" method's whole mechanism)
then removes real signal along with the confound.

This script tests whether that assumption is violated here, directly, rather than asserting it.
For each of the three evaluated targets (depression, antipsychotic, epilepsy) and the two CVS
schemes where the proposed method actually runs (CVS1, CVS2 -- CVS3 never computes A at all, see
06_run_evaluation.py), it fits the EXACT SAME pipeline objects (fit_shared_mf, chem_pca,
fit_per_indication_logreg, negative_control_area_factor) used by 06_run_evaluation.py and measures:

  overlap_r2 = R^2 of regressing the target's OWN true label y_k0 (train rows only) onto the area
  factor A estimated purely from its negative-control columns' residuals.

Under the method's identifying assumption this should be ~0 (A is unwanted variation, not signal).
A large value means the assumption is violated: A is predictive of the very thing the method is
trying to preserve, so removing it costs real precision -- exactly the mechanism that would explain
naive subtraction's advantage on the bias-diagnostic metrics in Section 4.

Writes: results/signal_overlap_diagnostic.csv (per target/cvs/seed), plus a printed summary
correlating overlap_r2 against the already-measured naive-vs-proposed AUPRC gap
(results/eval_auprc.csv).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)

        for cvs in ["CVS1", "CVS2"]:  # CVS3 never fits A (see module docstring)
            for seed in SEEDS:
                train_mask, test_rows = run_eval.make_masks(N, K, k0, cvs, seed)
                if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                    continue
                Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
                Zrep = np.hstack([Xchem, Ufree])
                clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
                if clfs[k0] is None:
                    continue
                A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)

                train_idx = np.where(train_mask[:, k0])[0]
                reg = LinearRegression().fit(A[train_idx], y_full[train_idx])
                r2 = reg.score(A[train_idx], y_full[train_idx])
                rows.append(dict(target=target, cvs=cvs, seed=seed, overlap_r2=r2,
                                  n_train=len(train_idx), n_pos_train=int(y_full[train_idx].sum())))

    diag = pd.DataFrame(rows)
    diag.to_csv("../results/signal_overlap_diagnostic.csv", index=False)
    summary = diag.groupby(["target", "cvs"]).overlap_r2.agg(["mean", "std", "count"])
    print("Signal-overlap R^2 (area factor A explaining the target's own true label, train-only):")
    print(summary)

    auprc = pd.read_csv("../results/eval_auprc.csv")
    gap_rows = []
    for (target, cvs), g in auprc[auprc.method.isin(["naive_subtract", "proposed_negctrl"])].groupby(["target", "cvs"]):
        piv = g.groupby(["seed", "method"]).auprc.mean().unstack()
        if "naive_subtract" not in piv.columns or "proposed_negctrl" not in piv.columns:
            continue
        gap = (piv["naive_subtract"] - piv["proposed_negctrl"]).mean()
        gap_rows.append(dict(target=target, cvs=cvs, naive_minus_proposed_auprc=gap))
    gap_df = pd.DataFrame(gap_rows)
    merged = summary.reset_index().merge(gap_df, on=["target", "cvs"], how="inner")
    merged.to_csv("../results/signal_overlap_vs_gap.csv", index=False)
    print("\nOverlap R^2 vs. measured naive-minus-proposed AUPRC gap, by (target, cvs):")
    print(merged[["target", "cvs", "mean", "naive_minus_proposed_auprc"]])
    if len(merged) >= 3:
        corr = merged["mean"].corr(merged["naive_minus_proposed_auprc"], method="spearman")
        print(f"\nSpearman correlation across {len(merged)} (target, cvs) cells: {corr:.3f} "
              f"(descriptive only, n too small for a hypothesis test)")

    print("\nWrote results/signal_overlap_diagnostic.csv, results/signal_overlap_vs_gap.csv")


if __name__ == "__main__":
    main()
