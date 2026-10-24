"""Alignment-check reliability sweep (Priority 2 of a new reviewer roadmap, pasted inline): tests
whether the alignment check's AUROC(P,F) statistic actually PREDICTS when a bias-diagnostic metric
will fail to register a correction's real effect, and whether it tracks a correction's TRUE
confound removal (ground truth, not a proxy), rather than being an interesting diagnostic with no
demonstrated predictive value.

Builds on code/20_synthetic_groundtruth.py's generative model (3,000 compounds, chemistry X,
confound z_area tied to a chemistry subspace, negative controls sharing area with the target).
Does not modify 20/26/28; three additions are new to this script:

  1. A PROXY FLAG F, distinct from the ground-truth high-confound flag already used by 20/26/28.
     Real same-ATC-area / SIDER-hard-negative flags are imperfect proxies for the true organ-system
     confound; here that is modeled by adding fixed Gaussian noise (0.4 * std(z_area), held constant
     across every sweep below) before thresholding, so F correlates with but is not identical to
     ground truth. The alignment check and "metric movement" below are computed against F, mirroring
     the real pipeline's Table~\\ref{tab:main-bias}/Table~\\ref{tab:alignment} relationship; TRUE
     confound removal is computed against ground truth, mirroring 20/26/28.

  2. Two more free generative parameters, beyond 26's noise-sigma and 28's confound_noise_scale:
       - confound_strength: S_CONF, the confound's weight in same-area indication log-odds
         (a fixed module constant in 20; a parameter here).
       - nc_contamination: a negative control's beta vector is mixed with the TARGET's own beta
         (unit-norm interpolation by contam_frac in [0,1]) -- the control becomes progressively
         less mechanistically independent of the target, violating the paper's own stated
         negative-control assumption on purpose to see what that does to the check.

  3. A noise knob directly on the ESTIMATED area factor A_oos (26's convention, applied here to the
     factor the real correction actually sees, not to the true z).

Four dimensions, swept one at a time from a shared default (confound_strength=1.3, matching 20's
original S_CONF; confound_noise_scale=0.5, matching 20's original default; nc_contamination=0.0,
fully independent controls as in 20; estimate_noise_sigma=0.0, the factor's own real estimation
error with no extra injected noise) x 5 levels x 30 independent seeds = 600 synthetic datasets,
same per-seed-redraws-everything convention as 20/26/28.

Per dataset, per method in {naive, proposed (fit-then-project), direct_subtract}:
  alignment_auroc        = sign-free AUROC(penalty term, PROXY flag F) -- the paper's existing check
  metric_responsiveness  = top50-F-rate(plain) - top50-F-rate(method): how much the bias-diagnostic
                            metric itself moves, mirroring code/19's real-data intervention test,
                            against F rather than ground truth
  true_delta_confound     = top50-truth-rate(plain) - top50-truth-rate(method): TRUE confound
                            removal, mirroring 20/26/28's existing top50_confound_rate
  auprc                   = AUPRC on the true target label, test rows

Pre-specified decision rule (fixed before looking at this sweep's results, not tuned against it):
alignment_auroc <= 0.58 predicts metric_responsiveness within +/-0.02 of zero ("the metric shows no
measurable response"). 0.58 matches the paper's own already-reported real-data near-chance range
(0.54-0.58) for cells where the metric failed to respond (Section~\\ref{sec:alignment}); +/-0.02
matches the real paper's own "stays exactly flat" language for a top50 rate
(Section~\\ref{sec:intervention}). Evaluated on seeds 15-29 of each (dimension, level) cell; seeds
0-14 are reserved and never used for this report, since the rule itself was fixed independent of
any data, train or test, in advance.

Writes: results/alignment_reliability_sweep.csv, figures/fig3_alignment_vs_confound_reduction.png
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, ".")
from models import negative_control_area_factor, fit_with_extra, score_project, sigmoid, project_area_factor_oos
import importlib
synth = importlib.import_module("20_synthetic_groundtruth")

N = synth.N
D_CHEM = synth.D_CHEM
N_DISEASES_IN_AREA = synth.N_DISEASES_IN_AREA
S_MECH = synth.S_MECH
SEEDS = synth.SEEDS
LAMBDA_NAIVE = synth.LAMBDA_NAIVE
PROXY_NOISE_REL = 0.4   # fixed, not swept: F = noisy proxy of the true confound, held constant

DEFAULTS = dict(confound_strength=1.3, confound_noise_scale=0.5, nc_contam_frac=0.0, estimate_noise_sigma=0.0)
DIMENSIONS = {
    "confound_strength": [0.3, 0.7, 1.3, 2.0, 3.0],
    "confound_noise_scale": [0.1, 0.25, 0.5, 1.0, 2.0],   # inverse of R^2(z,X); realized R^2 logged too
    "nc_contam_frac": [0.0, 0.1, 0.25, 0.5, 0.8],
    "estimate_noise_sigma": [0.0, 0.5, 1.0, 2.0, 3.0],
}


def simulate_custom(seed, confound_strength, confound_noise_scale, nc_contam_frac):
    """Same generative family as 20_synthetic_groundtruth.simulate(), with confound_strength and
    nc_contam_frac exposed as parameters (20 only exposes confound_noise_scale)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D_CHEM))
    w_area = rng.normal(size=5)
    w_area /= np.linalg.norm(w_area)
    z_area = X[:, :5] @ w_area + rng.normal(scale=confound_noise_scale, size=N)

    diseases = [f"area_disease_{j}" for j in range(N_DISEASES_IN_AREA)] + ["other_disease"]
    betas = {d: rng.normal(size=D_CHEM) / np.sqrt(D_CHEM) for d in diseases}
    target = diseases[0]
    beta_target = betas[target]
    for d in diseases[1:N_DISEASES_IN_AREA]:
        if nc_contam_frac > 0:
            mixed = np.sqrt(1 - nc_contam_frac ** 2) * betas[d] + nc_contam_frac * beta_target
            betas[d] = mixed / np.linalg.norm(mixed) * np.linalg.norm(betas[d])

    Y = {}
    for d in diseases:
        in_area = d != "other_disease"
        eta = S_MECH * (X @ betas[d]) + (confound_strength * z_area if in_area else 0.0)
        thresh = np.quantile(eta, 0.95)
        Y[d] = (rng.random(N) < (1 / (1 + np.exp(-(eta - thresh))))).astype(int)
    return X, z_area, diseases, Y


def sign_free_auroc(score, flag):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


def top50_rate(scores, cand, flag):
    order = cand[np.argsort(-scores[cand])]
    top = order[:min(50, len(order))]
    return flag[top].mean()


def run_one(seed, confound_strength, confound_noise_scale, nc_contam_frac, estimate_noise_sigma):
    X, z_area, diseases, Y = simulate_custom(seed, confound_strength, confound_noise_scale, nc_contam_frac)
    target = diseases[0]
    nc_names = diseases[1:N_DISEASES_IN_AREA]
    y_full = Y[target]

    rng_proxy = np.random.default_rng(900000 + seed)
    z_proxy = z_area + rng_proxy.normal(scale=PROXY_NOISE_REL * z_area.std(), size=N)
    high_confound_truth = z_area > np.quantile(z_area, 2 / 3)
    high_confound_proxy = z_proxy > np.quantile(z_proxy, 2 / 3)

    train_mask_row, test_rows = synth.make_masks_cvs2(N, seed)
    train_idx = np.where(train_mask_row)[0]
    cand = test_rows[y_full[test_rows] == 0]

    clf_plain = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
        X[train_idx], y_full[train_idx])
    logit_plain = X @ clf_plain.coef_[0] + clf_plain.intercept_[0]
    s_plain = sigmoid(logit_plain)

    nc_rate = np.mean([Y[d] for d in nc_names], axis=0)
    ar = (nc_rate - nc_rate.min()) / (nc_rate.max() - nc_rate.min() + 1e-9)
    s_naive = s_plain - LAMBDA_NAIVE * ar

    Y_nc = np.column_stack([Y[d] for d in nc_names])
    P_hat_nc = np.zeros_like(Y_nc, dtype=float)
    for j, d in enumerate(nc_names):
        clf_nc = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
            X[train_idx], Y[d][train_idx])
        P_hat_nc[:, j] = sigmoid(X @ clf_nc.coef_[0] + clf_nc.intercept_[0])
    train_mask_nc = np.tile(train_mask_row[:, None], (1, len(nc_names)))
    A = negative_control_area_factor(Y_nc, P_hat_nc, list(range(len(nc_names))), train_mask_nc, r_area=2)
    A_oos = project_area_factor_oos(A, X, train_idx)
    if estimate_noise_sigma > 0:
        rng_est = np.random.default_rng(700000 + seed)
        A_oos = A_oos + estimate_noise_sigma * A_oos.std() * rng_est.normal(size=A_oos.shape)

    clf_db, n_base = fit_with_extra(X, A_oos, y_full, train_idx)
    s_proposed = score_project(clf_db, X, n_base)

    calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
        A_oos[train_idx], y_full[train_idx])
    coef_direct = calib.coef_[0][0]
    s_direct = sigmoid(logit_plain - coef_direct * A_oos[:, 0])

    from sklearn.linear_model import LinearRegression
    realized_r2 = LinearRegression().fit(X, z_area).score(X, z_area)

    out = []
    penalties = dict(naive=ar, proposed=A_oos[:, 0], direct_subtract=A_oos[:, 0])
    scores = dict(naive=s_naive, proposed=s_proposed, direct_subtract=s_direct)
    for method in ["naive", "proposed", "direct_subtract"]:
        p = penalties[method]
        s = scores[method]
        align_auroc = sign_free_auroc(p[cand], high_confound_proxy[cand])
        metric_resp = top50_rate(s_plain, cand, high_confound_proxy) - top50_rate(s, cand, high_confound_proxy)
        true_delta = top50_rate(s_plain, cand, high_confound_truth) - top50_rate(s, cand, high_confound_truth)
        auprc = average_precision_score(y_full[test_rows], s[test_rows])
        out.append(dict(seed=seed, method=method, alignment_auroc=align_auroc,
                         metric_responsiveness=metric_resp, true_delta_confound=true_delta,
                         auprc=auprc, realized_r2_confound_vs_chem=realized_r2,
                         confound_strength=confound_strength, confound_noise_scale=confound_noise_scale,
                         nc_contam_frac=nc_contam_frac, estimate_noise_sigma=estimate_noise_sigma))

    # oracle, for the true-effectiveness reference scale
    clf_oracle, n_base_o = fit_with_extra(X, z_area[:, None], y_full, train_idx)
    s_oracle = score_project(clf_oracle, X, n_base_o)
    oracle_true_delta = top50_rate(s_plain, cand, high_confound_truth) - top50_rate(s_oracle, cand, high_confound_truth)
    out.append(dict(seed=seed, method="oracle", alignment_auroc=np.nan, metric_responsiveness=np.nan,
                     true_delta_confound=oracle_true_delta, auprc=np.nan,
                     realized_r2_confound_vs_chem=realized_r2, confound_strength=confound_strength,
                     confound_noise_scale=confound_noise_scale, nc_contam_frac=nc_contam_frac,
                     estimate_noise_sigma=estimate_noise_sigma))
    return out


def main():
    rows = []
    for dim, levels in DIMENSIONS.items():
        for level in levels:
            params = dict(DEFAULTS)
            params[dim] = level
            for seed in SEEDS:
                for r in run_one(seed, params["confound_strength"], params["confound_noise_scale"],
                                  params["nc_contam_frac"], params["estimate_noise_sigma"]):
                    r["dimension"] = dim
                    r["level"] = level
                    rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv("../results/alignment_reliability_sweep.csv", index=False)
    print(f"Wrote results/alignment_reliability_sweep.csv, {len(out)} rows")

    method_rows = out[out.method != "oracle"].copy()

    print("\n=== pooled pattern: corr(alignment_auroc, true_delta_confound), corr(alignment_auroc, metric_responsiveness), by method ===")
    for method in ["naive", "proposed", "direct_subtract"]:
        sub = method_rows[method_rows.method == method]
        r_true = np.corrcoef(sub.alignment_auroc, sub.true_delta_confound)[0, 1]
        r_metric = np.corrcoef(sub.alignment_auroc, sub.metric_responsiveness)[0, 1]
        print(f"{method}: corr(align, true_delta_confound)={r_true:.3f}  corr(align, metric_responsiveness)={r_metric:.3f}  n={len(sub)}")

    print("\n=== oracle true_delta_confound (reference scale), by dimension/level ===")
    orc = out[out.method == "oracle"]
    print(orc.groupby(["dimension", "level"]).true_delta_confound.mean().round(3).to_string())

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"naive": "tab:red", "proposed": "tab:blue", "direct_subtract": "tab:green"}
    for method in ["naive", "proposed", "direct_subtract"]:
        sub = method_rows[method_rows.method == method]
        ax.scatter(sub.alignment_auroc, sub.true_delta_confound, s=8, alpha=0.35,
                   label=method, color=colors[method])
    ax.axvline(0.58, color="gray", linestyle="--", linewidth=1, label="prespecified AUROC=0.58 cutoff")
    ax.set_xlabel("alignment-check AUROC(P,F)")
    ax.set_ylabel("true confound reduction (ground truth)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../figures/fig3_alignment_vs_confound_reduction.png", dpi=150)
    print("\nWrote figures/fig3_alignment_vs_confound_reduction.png")

    # decision rule: alignment_auroc <= 0.58 predicts |metric_responsiveness| < 0.02, seeds 15-29 only
    test_rows_mask = method_rows.seed >= 15
    ev = method_rows[test_rows_mask].copy()
    ev["predicted_nonresponsive"] = ev.alignment_auroc <= 0.58
    ev["true_nonresponsive"] = ev.metric_responsiveness.abs() < 0.02

    tp = ((ev.predicted_nonresponsive) & (ev.true_nonresponsive)).sum()
    fn = ((~ev.predicted_nonresponsive) & (ev.true_nonresponsive)).sum()
    tn = ((~ev.predicted_nonresponsive) & (~ev.true_nonresponsive)).sum()
    fp = ((ev.predicted_nonresponsive) & (~ev.true_nonresponsive)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (np.nan, np.nan)
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
        return center - half, center + half

    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    print(f"\n=== decision rule (alignment_auroc<=0.58 -> predict metric non-responsive), test seeds 15-29 ===")
    print(f"n={len(ev)}  TP={tp} FN={fn} TN={tn} FP={fp}")
    print(f"sensitivity={sens:.3f} (95% CI {sens_ci[0]:.3f}-{sens_ci[1]:.3f})")
    print(f"specificity={spec:.3f} (95% CI {spec_ci[0]:.3f}-{spec_ci[1]:.3f})")


if __name__ == "__main__":
    main()
