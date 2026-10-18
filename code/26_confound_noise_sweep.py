"""Confound-estimation-noise sweep on the synthetic ground-truth experiment (a fourth round of
review, pasted inline, Issue B): the oracle/direct-subtract comparison in
code/20_synthetic_groundtruth.py shows that, at the method's actual estimation quality (0.83 AUROC
against the true confound), fit-then-project-out (the proposed method's real mechanism) leaves more
of the confound in the ranked list than a cruder direct subtraction of the SAME estimated factor
does. That compares the two application mechanisms at one noise level. It does not show whether
fit-then-project-out is simply MORE SENSITIVE to estimation error, i.e. whether its disadvantage
would shrink or grow as the confound estimate gets better or worse.

This script sweeps that directly: starting from the true confound z_area (reusing
code/20_synthetic_groundtruth.py's simulate()), add Gaussian noise at increasing relative scale,
sigma in {0, 0.25, 0.5, 1, 1.5, 2, 3} times std(z_area), and apply BOTH application mechanisms to
the SAME noisy estimate at each sigma:
  fitproject: fit_with_extra(X, z_hat[:,None], y, train_idx) -> score_project  (the proposed
    method's actual mechanism, with z_hat standing in for its own estimated factor)
  directsub:  calibrate a univariate logistic of y on z_hat (train rows), subtract that calibrated
    contribution from the plain logit directly  (the method's own estimate applied the simple way
    naive applies its raw rate)

sigma=0 reproduces 20_synthetic_groundtruth.py's "oracle" (fitproject) and an analogous
noise-free direct-subtract-on-the-true-confound baseline. As sigma increases, both methods' input
degrades identically; the question is which one's top-50 true-confound rate rises toward (or past)
naive's own level faster.

A fifth round of review (pasted inline, Issue B) asked for paired effect sizes and confidence
intervals for exactly this fitproject-vs-directsub comparison, and for more independent synthetic
datasets rather than 5. SEEDS is imported from 20_synthetic_groundtruth, where it was grown from 5
to 30 for the same reason, so this sweep now reuses those same 30 independent datasets at each
sigma; main() reports, per sigma, a paired 95% CI and Cohen's d_z for (fitproject - directsub)
across those 30 seeds.

Writes: results/confound_noise_sweep.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from sklearn.metrics import roc_auc_score
from models import fit_with_extra, score_project, sigmoid
import importlib
synth = importlib.import_module("20_synthetic_groundtruth")

SIGMAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
SEEDS = synth.SEEDS


def main():
    rows = []
    for seed in SEEDS:
        X, z_area, diseases, Y = synth.simulate(seed)
        target = diseases[0]
        y_full = Y[target]
        z_std = z_area.std()

        train_mask_row, test_rows = synth.make_masks_cvs2(len(y_full), seed)
        train_idx = np.where(train_mask_row)[0]
        cand = test_rows[y_full[test_rows] == 0]
        high_confound = z_area > np.quantile(z_area, 2 / 3)

        clf_plain = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
            X[train_idx], y_full[train_idx])
        logit_plain = X @ clf_plain.coef_[0] + clf_plain.intercept_[0]

        for sigma in SIGMAS:
            rng = np.random.default_rng(20000 + seed * 100 + int(sigma * 10))
            z_hat = z_area + sigma * z_std * rng.normal(size=len(z_area))
            realized_corr = np.corrcoef(z_hat, z_area)[0, 1]
            realized_auroc = roc_auc_score(high_confound[cand], z_hat[cand])

            clf_fp, n_base_fp = fit_with_extra(X, z_hat[:, None], y_full, train_idx)
            s_fitproject = score_project(clf_fp, X, n_base_fp)

            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                z_hat[train_idx, None], y_full[train_idx])
            coef_direct = calib.coef_[0][0]
            s_directsub = sigmoid(logit_plain - coef_direct * z_hat)

            def top50(s):
                order = cand[np.argsort(-s[cand])]
                top = order[:min(50, len(order))]
                return high_confound[top].mean()

            rows.append(dict(seed=seed, sigma=sigma, realized_corr_zhat_ztrue=realized_corr,
                              realized_auroc_zhat_vs_flag=realized_auroc,
                              top50_confound_fitproject=top50(s_fitproject),
                              top50_confound_directsub=top50(s_directsub)))

    out = pd.DataFrame(rows)
    out.to_csv("../results/confound_noise_sweep.csv", index=False)
    print(out.groupby("sigma")[["realized_corr_zhat_ztrue", "realized_auroc_zhat_vs_flag",
                                 "top50_confound_fitproject",
                                 "top50_confound_directsub"]].mean().round(3).to_string())

    def ci95(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        return x.mean(), x.mean() - half, x.mean() + half

    print(f"\n=== paired (fitproject - directsub), same {len(SEEDS)} seeds at each sigma ===")
    for sigma in SIGMAS:
        sub = out[out.sigma == sigma]
        diff = sub["top50_confound_fitproject"].values - sub["top50_confound_directsub"].values
        d_m, d_lo, d_hi = ci95(diff)
        cohens_dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        print(f"sigma={sigma:.2f}: mean diff={d_m:.3f} ({d_lo:.3f}, {d_hi:.3f})  d_z={cohens_dz:.2f}")
    print("\nWrote results/confound_noise_sweep.csv")


if __name__ == "__main__":
    main()
