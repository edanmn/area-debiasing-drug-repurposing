"""Synthetic ground-truth confounding experiment (Priority 2 in the assessment.txt fix plan): the
real-data experiments can show that a correction's penalty term is or is not associated with a
PROXY bias flag (same-ATC-area membership, a SIDER hard-negative term), but never with the true
organ-system confound itself, since that confound's real value is unobserved in real data. This
script builds a small simulation where the confound is a literal generative variable, so "does
this correction's own factor track the real confound" is answerable directly, not inferred.

Generative model, per compound i:
  X_chem[i]        ~ N(0, I_30)                              -- chemistry-like features
  z_area[i]        = X_chem[i, :5] @ w_area + eps_area[i]     -- organ-system confound, itself a
                      (noisy) function of a chemistry subspace, mirroring this paper's own premise
                      that structurally related compounds share both disease-specific chemistry
                      AND organ-system membership
  eta_d[i]         = S_MECH * (X_chem[i] @ beta_d) + S_CONF * z_area[i] * in_area[d]   -- disease d's
                      true log-odds: a disease-specific mechanism term (beta_d, different per
                      disease) plus the shared confound (only for diseases in the confounded area)
  Y_d[i]           ~ Bernoulli(sigmoid(eta_d[i] - threshold_d))

Target k0 and its negative controls are diseases sharing in_area=True but with different, randomly
drawn beta_d (same area, different mechanism, the same structure as negative_controls() in
indication_matrix.py). An out-of-area disease (in_area=False, no z_area term) is included as a
sanity check that corrections don't do anything strange when there is no real confound to find.

What this measures, that real data cannot:
  1. corr(naive's raw_rate penalty, TRUE z_area)      -- does naive's penalty track the real confound?
  2. corr(proposed's area factor A / A_oos, TRUE z_area) -- does the RUV-style factor recover it?
  3. AUPRC on the true k0 label for plain/naive/proposed (precision-preservation, mirrors Table 1)
  4. Top-50 false-positive rate among the TRUE top-tertile-z_area compounds (mirrors Table 2's
     same-ATC-area metric, but against ground truth instead of a proxy flag)

A fifth round of review (pasted inline, Issue B) asked for more independent synthetic datasets, paired
effect sizes, and confidence intervals, particularly for fit-then-project vs direct-subtract, since 5
seeds is limited evidence of stability for the paper's central methodological claim. SEEDS below grew
from 5 to 30: each seed redraws X, w_area, all betas, and every Bernoulli label from scratch (see
simulate()), so these are 30 independent synthetic datasets under the same generative family, not 30
resamples of one fixed dataset. main() now reports a 95% CI (seed-level, t-distribution) on every
mean below, and a paired effect size (Cohen's d_z, mean paired difference over its SD across the SAME
30 seeds) for proposed vs direct_subtract specifically, the comparison this round of review asked for.

A third round of review (pasted inline, Issue 2 of that round) asked for two more scoring approaches,
to separate "does estimation fail" from "does application of a correctly estimated factor fail":
  5. direct_subtract: fit plain, then subtract a calibrated multiple of the ALREADY-ESTIMATED
     proposed factor A_oos directly from the score (no joint refit-and-project step), i.e. apply
     the proposed method's own estimated confound the same crude way naive applies raw_rate. If
     this removes the confound where the proposed method's actual projection step does not, the gap
     is in HOW the estimated factor is applied, not in estimating it.
  6. oracle: the same fit-with-extra-and-project mechanics as the proposed method, but given the
     TRUE z_area (not an estimated factor) as the extra column. This is the ceiling: the best this
     application mechanism could do with a perfectly estimated confound. If oracle still leaves the
     true confound in the ranked list, the gap is in the projection mechanism, not in estimation
     quality at all; if oracle removes it but "proposed" does not, imperfect estimation (0.83 AUROC,
     not 1.0) is a meaningful part of the gap too.

Writes: results/synthetic_groundtruth.csv
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, ".")
from models import (negative_control_area_factor, fit_with_extra, score_project, sigmoid,
                     project_area_factor_oos)

def sign_free_auroc(score, flag):
    """SVD factors have an arbitrary sign (flip the factor and its fitted coefficient flips to
    compensate, same likelihood); an AUROC near 0 and one near 1 are the same finding under
    opposite sign conventions. Matches the convention already used in
    code/15_ruv2_genomics_generalization.py for the same reason."""
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


N = 3000
D_CHEM = 30
N_DISEASES_IN_AREA = 4   # target + 3 negative controls
S_MECH = 1.5
S_CONF = 1.3
SEEDS = list(range(30))
LAMBDA_NAIVE = 2.0


def simulate(seed, confound_noise_scale=0.5):
    """confound_noise_scale controls how strongly z_area correlates with X's chemistry subspace:
    0.5 is this script's original, fixed choice. A sixth round of review (pasted inline, Issue 5/B)
    asked whether the correction-application gap (code/28_confound_chemistry_correlation_sensitivity.py)
    holds only under this one structural assumption; the default is unchanged so every existing
    caller (this script, code/26_confound_noise_sweep.py) reproduces its prior numbers exactly."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D_CHEM))
    w_area = rng.normal(size=5)
    w_area /= np.linalg.norm(w_area)
    z_area = X[:, :5] @ w_area + rng.normal(scale=confound_noise_scale, size=N)

    diseases = [f"area_disease_{j}" for j in range(N_DISEASES_IN_AREA)] + ["other_disease"]
    betas = {d: rng.normal(size=D_CHEM) / np.sqrt(D_CHEM) for d in diseases}
    Y = {}
    for j, d in enumerate(diseases):
        in_area = d != "other_disease"
        eta = S_MECH * (X @ betas[d]) + (S_CONF * z_area if in_area else 0.0)
        thresh = np.quantile(eta, 0.95)  # ~5% positive rate, mirrors real data's sparsity
        Y[d] = (rng.random(N) < (1 / (1 + np.exp(-(eta - thresh))))).astype(int)
    return X, z_area, diseases, Y


def make_masks_cvs2(n, seed):
    rng = np.random.default_rng(1000 + seed)
    row_test = rng.random(n) < 0.2
    return ~row_test, np.where(row_test)[0]


def main():
    rows = []
    for seed in SEEDS:
        X, z_area, diseases, Y = simulate(seed)
        target = diseases[0]
        nc_names = diseases[1:N_DISEASES_IN_AREA]
        y_full = Y[target]

        train_mask_row, test_rows = make_masks_cvs2(N, seed)
        train_idx = np.where(train_mask_row)[0]

        # plain: one logistic regression on X_chem, train rows only
        clf_plain = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
            X[train_idx], y_full[train_idx])
        logit_plain = X @ clf_plain.coef_[0] + clf_plain.intercept_[0]
        s_plain = sigmoid(logit_plain)

        # naive subtraction: raw negative-control rate penalty
        nc_rate = np.mean([Y[d] for d in nc_names], axis=0)
        ar = (nc_rate - nc_rate.min()) / (nc_rate.max() - nc_rate.min() + 1e-9)
        s_naive = s_plain - LAMBDA_NAIVE * ar

        # proposed: negative-control area-factor removal, same mechanics as models.py, built on a
        # train_mask shaped (N, n_nc) like the real pipeline expects
        Y_nc = np.column_stack([Y[d] for d in nc_names])
        P_hat_nc = np.zeros_like(Y_nc, dtype=float)
        for j, d in enumerate(nc_names):
            clf_nc = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                X[train_idx], Y[d][train_idx])
            P_hat_nc[:, j] = sigmoid(X @ clf_nc.coef_[0] + clf_nc.intercept_[0])
        train_mask_nc = np.tile(train_mask_row[:, None], (1, len(nc_names)))
        A = negative_control_area_factor(Y_nc, P_hat_nc, list(range(len(nc_names))), train_mask_nc, r_area=2)
        clf_db, n_base = fit_with_extra(X, A, y_full, train_idx)
        s_proposed = score_project(clf_db, X, n_base)
        A_oos = project_area_factor_oos(A, X, train_idx)

        # direct_subtract: calibrate a univariate logistic fit of y on the ALREADY-ESTIMATED proposed
        # factor A_oos, then subtract that calibrated contribution straight from the plain logit --
        # the proposed method's own estimated confound, applied the same crude way naive applies
        # raw_rate, instead of jointly refit-and-project.
        calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
            A_oos[train_idx], y_full[train_idx])
        coef_direct = calib.coef_[0][0]
        s_direct_subtract = sigmoid(logit_plain - coef_direct * A_oos[:, 0])

        # oracle: same fit-with-extra-and-project mechanics as "proposed", but given the TRUE
        # z_area (not an estimated factor) as the extra column -- the ceiling for this application
        # mechanism under perfect estimation.
        clf_oracle, n_base_oracle = fit_with_extra(X, z_area[:, None], y_full, train_idx)
        s_oracle = score_project(clf_oracle, X, n_base_oracle)

        cand = test_rows[y_full[test_rows] == 0]
        high_confound = z_area > np.quantile(z_area, 2 / 3)  # ground-truth analog of "same-area flag"

        def top50_confound_rate(scores):
            order = cand[np.argsort(-scores[cand])]
            top50 = order[:min(50, len(order))]
            return high_confound[top50].mean()

        rows.append(dict(
            seed=seed,
            auprc_plain=average_precision_score(y_full[test_rows], s_plain[test_rows]),
            auprc_naive=average_precision_score(y_full[test_rows], s_naive[test_rows]),
            auprc_proposed=average_precision_score(y_full[test_rows], s_proposed[test_rows]),
            auprc_direct_subtract=average_precision_score(y_full[test_rows], s_direct_subtract[test_rows]),
            auprc_oracle=average_precision_score(y_full[test_rows], s_oracle[test_rows]),
            top50_confound_base_rate=high_confound[cand].mean(),
            top50_confound_plain=top50_confound_rate(s_plain),
            top50_confound_naive=top50_confound_rate(s_naive),
            top50_confound_proposed=top50_confound_rate(s_proposed),
            top50_confound_direct_subtract=top50_confound_rate(s_direct_subtract),
            top50_confound_oracle=top50_confound_rate(s_oracle),
            corr_naive_penalty_vs_truez=np.corrcoef(ar[cand], z_area[cand])[0, 1],
            auroc_naive_penalty_vs_truez=roc_auc_score(high_confound[cand], ar[cand]),
            auroc_proposed_factor_raw_vs_truez=sign_free_auroc(A[cand, 0], high_confound[cand]),
            auroc_proposed_factor_oos_vs_truez=sign_free_auroc(A_oos[cand, 0], high_confound[cand]),
        ))

    out = pd.DataFrame(rows)
    out.to_csv("../results/synthetic_groundtruth.csv", index=False)
    print(out.mean(numeric_only=True).round(3))

    def ci95(x):
        x = np.asarray(x, dtype=float)
        n = len(x)
        se = x.std(ddof=1) / np.sqrt(n)
        half = stats.t.ppf(0.975, n - 1) * se
        return x.mean(), x.mean() - half, x.mean() + half

    print(f"\n=== 95% CI over {len(SEEDS)} independent seeds (t-distribution) ===")
    for col in ["top50_confound_plain", "top50_confound_naive", "top50_confound_proposed",
                "top50_confound_direct_subtract", "top50_confound_oracle"]:
        m, lo, hi = ci95(out[col])
        print(f"{col}: {m:.3f} ({lo:.3f}, {hi:.3f})")

    diff = out["top50_confound_proposed"].values - out["top50_confound_direct_subtract"].values
    d_m, d_lo, d_hi = ci95(diff)
    cohens_dz = diff.mean() / diff.std(ddof=1)
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    print(f"\n=== paired proposed vs direct_subtract (fit-then-project vs direct subtraction), "
          f"same {len(SEEDS)} seeds ===")
    print(f"mean paired difference: {d_m:.3f} ({d_lo:.3f}, {d_hi:.3f})  "
          f"Cohen's d_z={cohens_dz:.2f}  paired t={t_stat:.2f}  p={p_val:.2e}")
    print("\nWrote results/synthetic_groundtruth.csv")


if __name__ == "__main__":
    main()
