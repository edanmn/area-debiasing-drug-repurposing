"""Third-domain generalization test, harder than script 15's RUV-2/gender replication. That
replication used RUV-2's own introductory dataset: a single binary covariate (gender), fit by
unregularized OLS, exactly the idealized case the Proposition's exact-orthogonality claim was
originally (incorrectly) stated for. A reviewer pointed out this is weak evidence for
generality: one domain with a rich, regularized representation (drug repurposing) and one with
a single binary covariate and no regularization at all (genomics) leaves no intermediate case.

This script tests a genuinely harder case in the same negative-control/RUV family: the
bladderbatch dataset (Leek, Bioconductor, DOI 10.18129/B9.bioc.bladderbatch), standard worked
example for the sva/ComBat/RUV tutorial literature. 57 bladder samples, cancer status with
THREE levels (Normal=8, Biopsy=9, Cancer=40, not binary) as the factor of interest, batch with
FIVE levels (sizes 11/18/4/5/19) as the confound, and the fit here uses RIDGE regression
(L2-regularized, lambda > 0), not unregularized OLS. Negative controls are the 45 Affymetrix
bacterial/phage poly-A spike-in probes (see extract_bladderbatch_data.R for exactly which and
why), which measure only hybridization/technical variation and carry no possible biological
signal, a cleaner negative control than housekeeping genes in that specific sense.

Design, parallel to script 15/16: X = [intercept, Biopsy-dummy, Cancer-dummy] (Normal is the
reference level), genuinely multivariate, not a single covariate. For each of the 45
negative-control probes, fit ridge regression of its expression on X; the normal equation for
ridge is X^T r_j = alpha * beta_hat_j (the direct analog of the corrected Proposition's
X_tr^T W r_j = beta_hat_j / C for regularized logistic regression, same mechanism, OLS case),
so exact orthogonality fails here too unless alpha -> 0. Measure the real deviation rather than
assume it, and compare against a much weaker ridge penalty to confirm the deviation shrinks
as the fit approaches the unregularized limit.

Premise check: is batch (5-level) predictable from X (cancer status)? The raw cross-tab
(Biopsy only in batches 4-5, Normal only in batches 2-3) already suggests strong structural
correlation, unlike the gender/lab case where gender barely predicted lab. The check predicts
the opposite outcome from script 15's: if X is highly informative about the confound, the
orthogonalized factor What should lose much of its power to track batch, the same direction as
the drug-repurposing domain, not the original gender/lab genomics case.

Small-sample caveat, stated plainly rather than hidden: 57 samples and a 5-level confound with
one level as small as n=4 is a genuinely thin base for cross-validated AUROC. Results use 10
repeats of 3-fold stratified CV (the largest k a class of 4 supports) and should be read as
indicative, not precise, exactly the same caveat already given elsewhere in this project for
small negative-control classes.

Writes: results/bladderbatch_alignment_check.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

R_AREA = 2
ALPHA_ACTUAL = 10.0   # a real, non-trivial ridge penalty (sklearn's alpha, 1/C-like strength)
ALPHA_WEAK = 0.01      # 1000x weaker, approaching the unregularized limit
SEEDS = range(10)


def multiclass_ovr_auroc(feat, y_cat, seeds, n_splits=3):
    """Mean one-vs-rest AUROC across the 5 batch levels, averaged over repeated 3-fold CV
    (3 is the largest k the smallest batch, n=4, supports), since 57 samples is too few for a
    single reliable split."""
    classes = np.unique(y_cat)
    y_bin = label_binarize(y_cat, classes=classes)
    aucs_per_class = {c: [] for c in classes}
    for s in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        preds = np.zeros((len(y_cat), len(classes)))
        for tr, te in skf.split(feat, y_cat):
            for ci, c in enumerate(classes):
                ytr_c = y_bin[tr, ci]
                if ytr_c.sum() < 1 or ytr_c.sum() == len(ytr_c):
                    preds[te, ci] = ytr_c.mean() if len(ytr_c) else 0.5
                    continue
                clf = LogisticRegression(max_iter=2000, class_weight="balanced")
                clf.fit(feat[tr], ytr_c)
                preds[te, ci] = clf.predict_proba(feat[te])[:, 1]
        for ci, c in enumerate(classes):
            if y_bin[:, ci].sum() == 0 or y_bin[:, ci].sum() == len(y_bin):
                continue
            aucs_per_class[c].append(roc_auc_score(y_bin[:, ci], preds[:, ci]))
    means = {c: float(np.mean(v)) for c, v in aucs_per_class.items() if v}
    return float(np.mean(list(means.values()))), means


def main():
    expr_nc = pd.read_csv("../data/bladderbatch_controlgenes.csv", index_col=0)
    pheno = pd.read_csv("../data/bladderbatch_pheno.csv", index_col=0)
    assert list(expr_nc.columns) == list(pheno.index)

    cancer = pheno["cancer"].values
    batch = pheno["batch"].values.astype(str)
    Yc = expr_nc.values.T  # 57 samples x 45 negative-control probes

    # X: intercept + 2 dummies for the 3-level cancer factor (Normal = reference)
    is_biopsy = (cancer == "Biopsy").astype(float)
    is_cancer = (cancer == "Cancer").astype(float)
    X_nointercept = np.column_stack([is_biopsy, is_cancer])
    X = np.column_stack([np.ones(len(cancer)), is_biopsy, is_cancer])

    def fit_ridge_residuals(alpha):
        R = np.zeros_like(Yc)
        betas = np.zeros((Yc.shape[1], X_nointercept.shape[1]))
        for j in range(Yc.shape[1]):
            r = Ridge(alpha=alpha, fit_intercept=True).fit(X_nointercept, Yc[:, j])
            pred = r.predict(X_nointercept)
            R[:, j] = Yc[:, j] - pred
            betas[j] = r.coef_
        return R, betas

    R_actual, betas_actual = fit_ridge_residuals(ALPHA_ACTUAL)
    R_weak, betas_weak = fit_ridge_residuals(ALPHA_WEAK)

    def what_and_proj(R, betas, alpha):
        U, S, _ = np.linalg.svd(R, full_matrices=False)
        What = U[:, :R_AREA] * S[:R_AREA][None, :]
        max_proj = float(np.abs(X.T @ What).max())
        # normal-equation check: X_nointercept^T r_j should equal alpha * beta_hat_j
        gap = 0.0
        for j in range(R.shape[1]):
            lhs = X_nointercept.T @ R[:, j]
            rhs = alpha * betas[j]
            gap = max(gap, float(np.abs(lhs - rhs).max()))
        return What, max_proj, gap

    What_actual, max_proj_actual, gap_actual = what_and_proj(R_actual, betas_actual, ALPHA_ACTUAL)
    What_weak, max_proj_weak, gap_weak = what_and_proj(R_weak, betas_weak, ALPHA_WEAK)

    rng = np.random.default_rng(0)
    rand_vec = rng.standard_normal(What_actual.shape)
    rand_vec *= np.linalg.norm(What_actual, axis=0) / (np.linalg.norm(rand_vec, axis=0) + 1e-12)
    random_baseline = float(np.abs(X.T @ rand_vec).max())

    naive_penalty = Yc.mean(axis=1).reshape(-1, 1)

    cancer_to_batch_mean, cancer_to_batch_per = multiclass_ovr_auroc(X_nointercept, batch, SEEDS)
    naive_to_batch_mean, naive_to_batch_per = multiclass_ovr_auroc(naive_penalty, batch, SEEDS)
    what_actual_to_batch_mean, what_actual_to_batch_per = multiclass_ovr_auroc(What_actual, batch, SEEDS)
    what_weak_to_batch_mean, what_weak_to_batch_per = multiclass_ovr_auroc(What_weak, batch, SEEDS)

    out = pd.DataFrame([dict(
        n_samples=len(cancer), n_negctrl_probes=Yc.shape[1],
        alpha_actual=ALPHA_ACTUAL, alpha_weak=ALPHA_WEAK,
        max_proj_actual=max_proj_actual, max_proj_weak=max_proj_weak,
        normal_eq_gap_actual=gap_actual, normal_eq_gap_weak=gap_weak,
        random_baseline_proj=random_baseline,
        cancer_predicts_batch_auroc=cancer_to_batch_mean,
        naive_predicts_batch_auroc=naive_to_batch_mean,
        what_actual_predicts_batch_auroc=what_actual_to_batch_mean,
        what_weak_predicts_batch_auroc=what_weak_to_batch_mean,
    )])
    out.to_csv("../results/bladderbatch_alignment_check.csv", index=False)

    print(f"Step 1 (orthogonality): max|X^T What| at alpha={ALPHA_ACTUAL}: {max_proj_actual:.3f}")
    print(f"                         at alpha={ALPHA_WEAK}: {max_proj_weak:.3f}")
    print(f"                         random-vector baseline: {random_baseline:.3f}")
    print(f"normal-equation gap (should be ~0, confirms X^T r = alpha*beta identity): "
          f"{gap_actual:.3e} (actual), {gap_weak:.3e} (weak)")
    print()
    print(f"Step 2 premise: cancer status predicts batch, AUROC = {cancer_to_batch_mean:.3f}")
    print(f"  per-batch-level: {cancer_to_batch_per}")
    print()
    print(f"Step 3: naive (raw control-probe mean) predicts batch, AUROC = {naive_to_batch_mean:.3f}")
    print(f"        What (orthogonalized, alpha={ALPHA_ACTUAL}) predicts batch, AUROC = {what_actual_to_batch_mean:.3f}")
    print(f"        What (orthogonalized, alpha={ALPHA_WEAK}) predicts batch, AUROC = {what_weak_to_batch_mean:.3f}")
    print()
    print("Wrote results/bladderbatch_alignment_check.csv")


if __name__ == "__main__":
    main()
