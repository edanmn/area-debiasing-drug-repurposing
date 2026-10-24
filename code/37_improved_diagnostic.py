"""Reviewer request (pasted inline): Section 4.6 (code/35_alignment_check_reliability.py) showed
the existing alignment check -- AUROC(P,F) <= 0.58 predicting that a bias-diagnostic metric will
not move -- has poor sensitivity (11%, 6.5% for proposed alone), despite 75% precision when it
fires. The reviewer asks for a BETTER diagnostic: "an intervention-based responsiveness statistic,
computed on held-out validation data," compared against the existing check and trivial baselines on
independently generated datasets, with the bar being "predicts something genuinely useful, not
merely reproducing a metric response that was already measured."

THE NEW STATISTIC (val_dose_response): genuinely different from AUROC(P,F). AUROC(P,F) is a
standalone ASSOCIATION test -- does the penalty term alone discriminate the proxy flag F, on the
same rows used throughout. It never applies the correction and never touches a bias metric. The new
statistic instead mirrors code/19_intervention_doseresponse.py's real-data dose sweep: it actually
applies the correction's own real effect at alpha in {0, 0.5, 1} (0 = uncorrected, 1 = the method's
full-strength score) and measures how much the SAME top50-proxy-flag-rate bias metric moves, exactly
the quantity metric_responsiveness measures on the test set in script 35 -- except this sweep runs on
a VALIDATION subset, using a model that never saw those validation rows during fitting.

Three-way, non-overlapping split per dataset (N=3000, same generative family as 20/26/35):
  1. test_rows   (~20% of N): IDENTICAL to script 35's synth.make_masks_cvs2 test rows. Used only to
     compute metric_responsiveness (ground truth for this evaluation), exactly reproducing script
     35's own numbers. Never touched by anything below.
  2. train_idx   (~80% of N): script 35's convention. Further split (seed-local permutation, a new
     independent RNG keyed by 500000+seed) into:
       a. fit_idx (75% of train_idx, ~60% of N): used to fit every model that produces the new
          statistic (plain classifier, per-negative-control classifiers, the negative-control area
          factor, its out-of-sample ridge projection, the joint fit-with-extra model, the
          direct-subtract calibration).
       b. val_idx (25% of train_idx, ~20% of N): held out from every one of those fits. The dose
          sweep and the resulting top50-proxy-flag-rate movement are measured ONLY on val_idx rows,
          using fit_idx-fitted models applied out-of-sample to val_idx. val_idx is disjoint from
          fit_idx by construction and disjoint from test_rows because it is carved out of train_idx,
          which never overlaps test_rows.
  val_dose_response = top50_rate(score_at_alpha=0, val_candidates, F) - top50_rate(score_at_alpha=1,
  val_candidates, F), i.e. the same before/after top50-rate-movement definition as
  metric_responsiveness, just measured on validation rows with a validation-blind fit instead of on
  the final test rows.

PRE-SPECIFIED THRESHOLD, set before touching the evaluation seeds: seeds 0-4 (5 of script 35's
already-reserved, never-evaluated 0-14 seed pool) are used ONLY to characterize val_dose_response's
noise floor. Among pilot-seed (dimension, level, method, seed) cells where the TEST set's
metric_responsiveness is genuinely flat (|.| <= 0.02), the 95th percentile of |val_dose_response| is
taken as THRESH: a one-sided bound on how much this statistic fluctuates by chance alone when there
is truly nothing to detect. THRESH is then frozen and applied, unmodified, to the evaluation seeds
(15-29, identical to script 35's evaluation seeds) -- it is never swept against those seeds' own
outcomes.

Evaluated on seeds 15-29 (matching script 35 exactly), against:
  (a) the new statistic:      predict flat if |val_dose_response| <= THRESH
  (b) the existing check:     predict flat if alignment_auroc <= 0.58 (recomputed identically, must
                               reproduce script 35's 11% sensitivity / 75% precision)
  (c) always-predict-flat
  (d) base-rate-matched random guess, stratified by method

Builds on results/alignment_reliability_sweep.csv (code/35's own output, read directly rather than
rerun, since it already contains every seed/dimension/level/method this script needs for (b) and for
ground truth) plus a new simulate-and-fit pass (seeds 0-4 and 15-29 only, 20 of the 30 seeds per
cell) to compute val_dose_response.

Writes: results/improved_diagnostic_sweep.csv, results/improved_diagnostic_comparison.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from models import negative_control_area_factor, fit_with_extra, score_project, sigmoid, project_area_factor_oos
import importlib
synth = importlib.import_module("20_synthetic_groundtruth")
alignrel = importlib.import_module("35_alignment_check_reliability")

N = synth.N
N_DISEASES_IN_AREA = synth.N_DISEASES_IN_AREA
LAMBDA_NAIVE = synth.LAMBDA_NAIVE
DIMENSIONS = alignrel.DIMENSIONS
DEFAULTS = alignrel.DEFAULTS
PROXY_NOISE_REL = alignrel.PROXY_NOISE_REL
simulate_custom = alignrel.simulate_custom
top50_rate = alignrel.top50_rate

PILOT_SEEDS = list(range(5))      # subset of script 35's reserved (never-evaluated) 0-14 pool;
                                   # used ONLY to set THRESH, never to measure sensitivity/precision
EVAL_SEEDS = list(range(15, 30))  # identical to script 35's evaluation seeds
VAL_FRAC_OF_TRAIN = 0.25
DOSE_ALPHAS = [0.0, 0.5, 1.0]
METHODS = ["naive", "proposed", "direct_subtract"]


def run_validation_stat(seed, confound_strength, confound_noise_scale, nc_contam_frac, estimate_noise_sigma):
    X, z_area, diseases, Y = simulate_custom(seed, confound_strength, confound_noise_scale, nc_contam_frac)
    target = diseases[0]
    nc_names = diseases[1:N_DISEASES_IN_AREA]
    y_full = Y[target]

    rng_proxy = np.random.default_rng(900000 + seed)
    z_proxy = z_area + rng_proxy.normal(scale=PROXY_NOISE_REL * z_area.std(), size=N)
    high_confound_proxy = z_proxy > np.quantile(z_proxy, 2 / 3)

    train_mask_row, test_rows = synth.make_masks_cvs2(N, seed)
    train_idx = np.where(train_mask_row)[0]

    rng_val = np.random.default_rng(500000 + seed)
    perm = rng_val.permutation(len(train_idx))
    n_val = int(round(VAL_FRAC_OF_TRAIN * len(train_idx)))
    val_idx = train_idx[perm[:n_val]]
    fit_idx = train_idx[perm[n_val:]]
    fit_mask_row = np.zeros(N, dtype=bool)
    fit_mask_row[fit_idx] = True

    cand_val = val_idx[y_full[val_idx] == 0]
    if len(cand_val) < 10 or y_full[fit_idx].sum() < 2:
        return []

    clf_plain_val = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
        X[fit_idx], y_full[fit_idx])
    logit_plain_val = X @ clf_plain_val.coef_[0] + clf_plain_val.intercept_[0]
    s_plain_val = sigmoid(logit_plain_val)

    nc_rate = np.mean([Y[d] for d in nc_names], axis=0)
    ar = (nc_rate - nc_rate.min()) / (nc_rate.max() - nc_rate.min() + 1e-9)

    Y_nc = np.column_stack([Y[d] for d in nc_names])
    P_hat_nc_val = np.zeros_like(Y_nc, dtype=float)
    for j, d in enumerate(nc_names):
        clf_nc = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
            X[fit_idx], Y[d][fit_idx])
        P_hat_nc_val[:, j] = sigmoid(X @ clf_nc.coef_[0] + clf_nc.intercept_[0])
    fit_mask_nc = np.tile(fit_mask_row[:, None], (1, len(nc_names)))
    A_val = negative_control_area_factor(Y_nc, P_hat_nc_val, list(range(len(nc_names))), fit_mask_nc, r_area=2)
    A_oos_val = project_area_factor_oos(A_val, X, fit_idx)
    if estimate_noise_sigma > 0:
        rng_est = np.random.default_rng(710000 + seed)
        A_oos_val = A_oos_val + estimate_noise_sigma * A_oos_val.std() * rng_est.normal(size=A_oos_val.shape)

    clf_db_val, n_base_val = fit_with_extra(X, A_oos_val, y_full, fit_idx)
    if clf_db_val is None:
        return []
    coef_base_val = clf_db_val.coef_[0][:n_base_val]
    coef_A_val = clf_db_val.coef_[0][n_base_val:]
    base_logit_val = X @ coef_base_val + clf_db_val.intercept_[0]
    area_logit_val = A_oos_val @ coef_A_val

    calib_val = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
        A_oos_val[fit_idx], y_full[fit_idx])
    coef_direct_val = calib_val.coef_[0][0]

    def score_at_alpha(method, alpha):
        if method == "naive":
            return s_plain_val - alpha * LAMBDA_NAIVE * ar
        if method == "direct_subtract":
            return sigmoid(logit_plain_val - alpha * coef_direct_val * A_oos_val[:, 0])
        if method == "proposed":
            return sigmoid(base_logit_val + (1 - alpha) * area_logit_val)
        raise ValueError(method)

    out = []
    for method in METHODS:
        rates = {a: top50_rate(score_at_alpha(method, a), cand_val, high_confound_proxy) for a in DOSE_ALPHAS}
        out.append(dict(seed=seed, method=method,
                         val_dose_response=rates[0.0] - rates[1.0],
                         val_rate_alpha0=rates[0.0], val_rate_alpha_mid=rates[0.5], val_rate_alpha1=rates[1.0],
                         n_cand_val=len(cand_val)))
    return out


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def confusion_stats(pred_flat, true_flat):
    pred_flat = np.asarray(pred_flat, dtype=bool)
    true_flat = np.asarray(true_flat, dtype=bool)
    tp = int((pred_flat & true_flat).sum())
    fn = int((~pred_flat & true_flat).sum())
    tn = int((~pred_flat & ~true_flat).sum())
    fp = int((pred_flat & ~true_flat).sum())
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    spec_lo, spec_hi = wilson_ci(tn, tn + fp)
    prec_lo, prec_hi = wilson_ci(tp, tp + fp)
    return dict(n=len(pred_flat), tp=tp, fn=fn, tn=tn, fp=fp,
                sensitivity=sens, sens_ci_lo=sens_lo, sens_ci_hi=sens_hi,
                specificity=spec, spec_ci_lo=spec_lo, spec_ci_hi=spec_hi,
                precision=prec, prec_ci_lo=prec_lo, prec_ci_hi=prec_hi)


def fmt(s):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}


def main():
    legacy = pd.read_csv("../results/alignment_reliability_sweep.csv")
    legacy = legacy[legacy.method != "oracle"].copy()

    need_seeds = sorted(set(PILOT_SEEDS) | set(EVAL_SEEDS))
    val_rows = []
    for dim, levels in DIMENSIONS.items():
        for level in levels:
            params = dict(DEFAULTS)
            params[dim] = level
            for seed in need_seeds:
                for r in run_validation_stat(seed, params["confound_strength"], params["confound_noise_scale"],
                                              params["nc_contam_frac"], params["estimate_noise_sigma"]):
                    r["dimension"] = dim
                    r["level"] = level
                    val_rows.append(r)
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv("../results/improved_diagnostic_sweep.csv", index=False)
    print(f"Wrote results/improved_diagnostic_sweep.csv, {len(val_df)} rows")

    merged = legacy.merge(val_df, on=["dimension", "level", "seed", "method"], how="inner")
    expected = len(need_seeds) * sum(len(v) for v in DIMENSIONS.values()) * len(METHODS)
    print(f"merged rows: {len(merged)} (expected up to {expected}, less any degenerate-fit drops)")

    # --- pre-specify THRESH on pilot seeds 0-4 only, never touching eval seeds 15-29 ---
    pilot = merged[merged.seed.isin(PILOT_SEEDS)]
    pilot_true_flat = pilot[pilot.metric_responsiveness.abs() <= 0.02]
    THRESH = float(np.quantile(pilot_true_flat.val_dose_response.abs(), 0.95))
    print(f"\nPre-specified THRESH = 95th percentile of |val_dose_response| among {len(pilot_true_flat)} "
          f"pilot (seeds 0-4) cells with genuinely flat TEST metric_responsiveness: {THRESH:.4f}")
    print(f"(pilot pool size: {len(pilot)}, flat fraction in pilot: {pilot.metric_responsiveness.abs().le(0.02).mean():.3f})")

    ev = merged[merged.seed.isin(EVAL_SEEDS)].copy()
    ev["true_flat"] = ev.metric_responsiveness.abs() <= 0.02

    rng_bl = np.random.default_rng(20261023)
    ev["pred_new"] = ev.val_dose_response.abs() <= THRESH
    ev["pred_existing"] = ev.alignment_auroc <= 0.58
    ev["pred_always_flat"] = True
    ev["pred_random"] = False
    for meth in METHODS:
        msk = (ev.method == meth).values
        p = ev.loc[msk, "true_flat"].mean()
        ev.loc[msk, "pred_random"] = rng_bl.random(int(msk.sum())) < p

    strategies = {
        "new_validation_stat": "pred_new",
        "existing_check_AUROC(P,F)<=0.58": "pred_existing",
        "always_predict_flat": "pred_always_flat",
        "base_rate_random": "pred_random",
    }

    print(f"\n=== OVERALL, eval seeds 15-29, all methods pooled (n={len(ev)}, true flat rate={ev.true_flat.mean():.3f}) ===")
    for name, col in strategies.items():
        print(name, fmt(confusion_stats(ev[col], ev.true_flat)))

    print("\n=== BY METHOD ===")
    for meth in METHODS:
        sub = ev[ev.method == meth]
        print(f"-- {meth} (n={len(sub)}, flat_rate={sub.true_flat.mean():.3f}) --")
        for name, col in strategies.items():
            print(" ", name, fmt(confusion_stats(sub[col], sub.true_flat)))

    print("\n=== BY SWEPT DIMENSION ===")
    for dim in DIMENSIONS:
        sub = ev[ev.dimension == dim]
        print(f"-- {dim} (n={len(sub)}, flat_rate={sub.true_flat.mean():.3f}) --")
        for name, col in strategies.items():
            print(" ", name, fmt(confusion_stats(sub[col], sub.true_flat)))

    comp_rows = []

    def add_block(breakdown, value, sub):
        for name, col in strategies.items():
            s = confusion_stats(sub[col], sub.true_flat)
            s.update(breakdown=breakdown, breakdown_value=value, strategy=name,
                      true_flat_rate=sub.true_flat.mean())
            comp_rows.append(s)

    add_block("overall", "all", ev)
    for meth in METHODS:
        add_block("method", meth, ev[ev.method == meth])
    for dim in DIMENSIONS:
        add_block("dimension", dim, ev[ev.dimension == dim])

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv("../results/improved_diagnostic_comparison.csv", index=False)
    print("\nWrote results/improved_diagnostic_comparison.csv")
    print(f"\nTHRESH = {THRESH:.4f}")


if __name__ == "__main__":
    main()
