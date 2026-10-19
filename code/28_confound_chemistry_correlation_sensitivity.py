"""Confound-chemistry correlation sensitivity sweep (a sixth round of review, pasted inline, Issue
5/B, marked "desirable, rather than mandatory"): code/20_synthetic_groundtruth.py's correction-
application gap (fit-then-project leaves more true confound in the top 50 than direct subtraction
of the SAME estimated factor) is measured under one fixed structural assumption, z_area's noise
scale of 0.5 relative to its chemistry-subspace signal. This script asks whether that gap is an
artifact of that one choice, by sweeping confound_noise_scale in simulate() (code/20's new optional
argument, default unchanged so 20 and 26 still reproduce their prior numbers) across a range from a
confound almost entirely determined by chemistry to one almost independent of it, and reporting, at
each level, the realized R^2 of regressing z_area on X (the actual correlation strength, not the
noise-scale knob itself) alongside the paired fit-project vs. direct-subtract gap, 30 seeds each.

Writes: results/confound_chemistry_sensitivity.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, ".")
from models import fit_with_extra, score_project, sigmoid, project_area_factor_oos
import importlib
synth = importlib.import_module("20_synthetic_groundtruth")

NOISE_SCALES = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = synth.SEEDS


def main():
    rows = []
    for noise_scale in NOISE_SCALES:
        for seed in SEEDS:
            X, z_area, diseases, Y = synth.simulate(seed, confound_noise_scale=noise_scale)
            target = diseases[0]
            y_full = Y[target]

            train_mask_row, test_rows = synth.make_masks_cvs2(len(y_full), seed)
            train_idx = np.where(train_mask_row)[0]
            cand = test_rows[y_full[test_rows] == 0]
            high_confound = z_area > np.quantile(z_area, 2 / 3)

            realized_r2 = LinearRegression().fit(X, z_area).score(X, z_area)

            clf_plain = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                X[train_idx], y_full[train_idx])
            logit_plain = X @ clf_plain.coef_[0] + clf_plain.intercept_[0]

            clf_fp, n_base_fp = fit_with_extra(X, z_area[:, None], y_full, train_idx)
            s_fitproject = score_project(clf_fp, X, n_base_fp)

            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                z_area[train_idx, None], y_full[train_idx])
            coef_direct = calib.coef_[0][0]
            s_directsub = sigmoid(logit_plain - coef_direct * z_area)

            def top50(s):
                order = cand[np.argsort(-s[cand])]
                top = order[:min(50, len(order))]
                return high_confound[top].mean()

            rows.append(dict(noise_scale=noise_scale, seed=seed, realized_r2_confound_vs_chem=realized_r2,
                              top50_confound_fitproject=top50(s_fitproject),
                              top50_confound_directsub=top50(s_directsub)))

    out = pd.DataFrame(rows)
    out.to_csv("../results/confound_chemistry_sensitivity.csv", index=False)

    def ci95(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        return x.mean(), x.mean() - half, x.mean() + half

    print(f"=== paired (fitproject - directsub) across {len(SEEDS)} seeds, by confound-chemistry R^2 ===")
    for noise_scale in NOISE_SCALES:
        sub = out[out.noise_scale == noise_scale]
        r2_m = sub.realized_r2_confound_vs_chem.mean()
        diff = sub.top50_confound_fitproject.values - sub.top50_confound_directsub.values
        d_m, d_lo, d_hi = ci95(diff)
        d_z = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        print(f"noise_scale={noise_scale:.2f}  R^2(confound~chem)={r2_m:.3f}  "
              f"diff={d_m:.3f} ({d_lo:.3f}, {d_hi:.3f})  d_z={d_z:.2f}")
    print("\nWrote results/confound_chemistry_sensitivity.csv")


if __name__ == "__main__":
    main()
