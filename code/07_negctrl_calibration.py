"""Negative-control validity check (literature_review.md section 5 item 5 / DATA_PLAN.md's final
caveat): after fitting the proposed method's area factor for a target indication, verify there is
NO residual association among the negative-control indications THEMSELVES once the same
correction machinery is applied to them -- a calibration-style diagnostic analogous to
Lipsitch/Tchetgen Tchetgen negative-control calibration.

For each target k0 and each ncj in NC(k0): refit ncj with the area factor estimated from
NC(k0) \\ {ncj} (never touching ncj's own labels) and compare AUPRC(plain) vs AUPRC(corrected) on a
held-out 20% of compounds. If the negative controls are mechanistically independent of one another
(as assumed), this correction should leave ncj's own predictability roughly UNCHANGED.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project)

TARGETS = ["depression", "antipsychotic", "epilepsy"]
# Bumped from 5 to 20 reruns per coordinator follow-up request, specifically to settle whether the
# anxiolytic calibration cells (large +0.127/+0.126 mean delta on only 4 reruns in the original
# run) are genuine calibration failures or small-sample noise (anxiolytic has only 7 positives in
# the whole drug pool, so a handful of its positives landing differently across the 20% row
# holdout per seed can swing AUPRC wildly).
SEEDS = list(range(20))
R_FREE, R_CHEM = 16, 32


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    rows = []

    for target in TARGETS:
        nc_names = negative_controls(target, diseases=cols)
        print(f"\ntarget={target}  NC={nc_names}")
        for ncj in nc_names:
            calib_set = [d for d in nc_names if d != ncj]
            ncj_idx = col_idx[ncj]
            calib_idx = [col_idx[d] for d in calib_set]
            y_ncj = Y[:, ncj_idx].astype(int)
            if y_ncj.sum() < 5 or len(calib_idx) == 0:
                continue
            for seed in SEEDS:
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
                s_plain = score_project(clfs[ncj_idx], Zrep, Zrep.shape[1])

                A = negative_control_area_factor(Y, P_hat, calib_idx, train_mask, r_area=2)
                train_idx = np.where(train_mask[:, ncj_idx])[0]
                clf_db, n_base = fit_with_extra(Zrep, A, y_ncj, train_idx)
                s_debiased = score_project(clf_db, Zrep, n_base)

                ap_plain = average_precision_score(y_ncj[test_rows], s_plain[test_rows])
                ap_db = average_precision_score(y_ncj[test_rows], s_debiased[test_rows])
                rows.append(dict(target=target, calibration_indication=ncj, seed=seed,
                                 auprc_plain=ap_plain, auprc_corrected=ap_db,
                                 delta=ap_db - ap_plain))

    df = pd.DataFrame(rows)
    df.to_csv("../results/negctrl_calibration.csv", index=False)
    summ = df.groupby(["target", "calibration_indication"]).delta.agg(["mean", "std", "count"]).reset_index()
    summ.to_csv("../results/negctrl_calibration_summary.csv", index=False)
    print("\nPer-(target, calibration indication) mean AUPRC delta (corrected - plain); "
          "should cluster near 0 if negative controls are mechanistically independent:")
    print(summ.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="gray", lw=1, ls="--")
    labels = summ.target + ":" + summ.calibration_indication
    ax.errorbar(range(len(summ)), summ["mean"],
                yerr=summ["std"] / np.sqrt(summ["count"].clip(lower=1)),
                fmt="o", capsize=3, color="C0")
    ax.set_xticks(range(len(summ))); ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("AUPRC delta (corrected - plain)\n(negative-control self-calibration)")
    ax.set_title("Negative-control calibration check\n(should scatter around 0)")
    fig.tight_layout()
    fig.savefig("../figures/negctrl_calibration.png", dpi=150)
    print("\nWrote results/negctrl_calibration.csv, negctrl_calibration_summary.csv, "
          "figures/negctrl_calibration.png")


if __name__ == "__main__":
    main()
