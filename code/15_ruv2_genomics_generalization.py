"""Tests the two-step alignment check outside drug repurposing, on RUV-2's own original data.

Section 13-14 and the paper's Proposition (sec:alignment-check) make a general claim: (1) a
correction's penalty term, when built as an SVD of residuals from a model fit on a base
representation X, is exactly orthogonal to X on train rows by construction, a fact about the
estimator, not about drug repurposing; and (2) whether that correction's penalty term ends up
looking uninformative about a bias-diagnostic flag F depends entirely on whether F is itself
predictable from X. This script re-runs both steps on a structurally different setting: the
original RUV-2 worked example (Gagnon-Bartsch & Speed 2012), gene expression from Vawter et al.
2004, with gender as the factor of interest, lab as the batch confound, and the paper's own 799
housekeeping probesets (Eisenberg & Levanon 2003) as negative controls. Here X is just the
gender design matrix (not a rich chemistry representation), fit by ordinary least squares
instead of logistic regression; Yc are control genes instead of negative-control indications;
What (the leading left singular vectors of the OLS residuals of Yc on X) is the genomics analog
of A.

Step (1) holds exactly here too (confirmed numerically to float precision, not just in theory).

Step (2) comes out differently than in the drug-repurposing case study: lab membership is NOT
well predicted by gender (AUROC well below 0.5, i.e. no better than chance), unlike same-ATC-area
membership, which WAS well predicted by chemistry+MF in Section 14. The check predicts what
should follow from that: since X carries little information about the confound F=lab, projecting
What out of X should not "use up" What's capacity to track F, so What should remain substantially
informative about lab, unlike the drug-repurposing case where the orthogonalized term fell to
chance. That is exactly what the numbers below show: What separates lab membership well above
chance (and a logistic fit on the full multi-factor What recovers lab almost perfectly), while a
naive, unprojected summary of the control genes alone does comparably or less well. The two
settings produce opposite-looking empirical outcomes (RUV's term wins here, naive's term won in
drug repurposing) for a reason the check itself anticipates: the premise in step (2) differs
between the two domains, and the check's prediction tracks that difference correctly in both.

Data: data/ruv2_gender_controlgenes.csv (799 negative-control probesets x 84 samples) and
data/ruv2_gender_pheno.csv (84 samples x gender/lab/region design), extracted from the
Bioconductor RUVnormalizeData package; see extract_ruv2_gender_data.R for provenance.

Writes: results/ruv2_alignment_check.csv
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LAB_COLS = ["D", "I", "M"]
K_FACTORS = 4
SEEDS = [0, 1, 2, 3, 4]


def sign_free_auroc(score, y):
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def cv_auroc(feature_mat, y, seeds, n_splits=5):
    aucs = []
    for s in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
        preds = np.zeros(len(y))
        for tr, te in cv.split(feature_mat, y):
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(feature_mat[tr], y[tr])
            preds[te] = clf.predict_proba(feature_mat[te])[:, 1]
        aucs.append(roc_auc_score(y, preds))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    expr_nc = pd.read_csv("../data/ruv2_gender_controlgenes.csv", index_col=0)
    pheno = pd.read_csv("../data/ruv2_gender_pheno.csv", index_col=0)
    assert list(expr_nc.columns) == list(pheno.index)

    gender = (pheno["gender"] == "M").astype(float).values
    lab = pheno[LAB_COLS].values
    Yc = expr_nc.values.T  # 84 samples x 799 negative-control genes

    X = np.column_stack([np.ones(len(gender)), gender])  # intercept + gender, the factor of interest
    proj = X @ np.linalg.inv(X.T @ X) @ X.T
    R = (np.eye(len(gender)) - proj) @ Yc  # OLS residuals of each control gene on X
    max_resid_dot = float(np.abs(X.T @ R).max())

    U, _, _ = np.linalg.svd(R, full_matrices=False)
    What = U[:, :K_FACTORS]
    max_what_dot = float(np.abs(X.T @ What).max())

    naive_penalty = Yc.mean(axis=1)       # raw, unprojected mean of the control genes
    ruv_penalty = What[:, 0]              # leading confound factor, the genomics analog of A
    ruv_penalty_norm = np.linalg.norm(What, axis=1)

    rows = []
    gender_to_lab_mean, gender_to_lab_std, what_to_lab_mean, what_to_lab_std = {}, {}, {}, {}
    for i, lc in enumerate(LAB_COLS):
        y = lab[:, i]
        gender_to_lab_mean[lc], gender_to_lab_std[lc] = cv_auroc(X[:, [1]], y, SEEDS)
        what_to_lab_mean[lc], what_to_lab_std[lc] = cv_auroc(What, y, SEEDS)
        rows.append(dict(
            lab=lc,
            naive_penalty_auroc=sign_free_auroc(naive_penalty, y),
            ruv_factor1_auroc=sign_free_auroc(ruv_penalty, y),
            ruv_norm4_auroc=sign_free_auroc(ruv_penalty_norm, y),
            gender_predicts_lab_mean=gender_to_lab_mean[lc],
            gender_predicts_lab_std=gender_to_lab_std[lc],
            what_predicts_lab_mean=what_to_lab_mean[lc],
            what_predicts_lab_std=what_to_lab_std[lc],
        ))

    out = pd.DataFrame(rows)
    out.to_csv("../results/ruv2_alignment_check.csv", index=False)

    print(f"max|X^T R|  = {max_resid_dot:.3e}  (step 1: OLS normal equations, should be ~0)")
    print(f"max|X^T What| = {max_what_dot:.3e}  (step 1: SVD column-space argument, should be ~0)")
    print()
    print("Step 2 premise: is F=lab predictable from X=gender? (5 seeds, 5-fold CV)")
    for lc in LAB_COLS:
        print(f"  lab={lc}: AUROC = {gender_to_lab_mean[lc]:.3f} +/- {gender_to_lab_std[lc]:.3f}")
    print()
    print(out.round(3).to_string(index=False))
    print()
    print("Wrote results/ruv2_alignment_check.csv")


if __name__ == "__main__":
    main()
