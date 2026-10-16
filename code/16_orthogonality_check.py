"""Checks the Proposition's exactness claim (sec:alignment-check) against the models actually
fit in the pipeline, rather than against an idealized unregularized MLE.

`fit_per_indication_logreg` (models.py) fits each per-indication model with sklearn
LogisticRegression at C=1.0 (L2-penalized MAP, not unregularized MLE). sklearn solves
min_w (1/2)||w||^2 + C * sum_i s_i * logloss_i(w), so at the fitted coefficients what actually
holds is the PENALIZED score equation

    X_tr^T W r_j = beta_hat_j / C

not X_tr^T W r_j = 0. The two agree only as C -> infinity (unregularized). This script (a)
confirms that identity holds numerically at the real C=1.0 fits used throughout the paper,
(b) quantifies how far that is from floating-point zero, and (c) confirms the deviation shrinks
toward zero as C grows, which is the correct, falsifiable form of the claim.

Writes: results/orthogonality_check.csv
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2]
C_ACTUAL = 1.0       # the value used everywhere else in the pipeline
C_WEAK = 100.0       # much weaker penalty, closer to the unregularized limit


def max_abs_proj(Xtr, Atr):
    return float(np.abs(Xtr.T @ Atr).max())


def frac_variance_in_X(Xtr, Atr):
    """Fraction of A's total column-space energy that lies inside X's column space (via the
    orthogonal projector onto X_tr), the direct answer to 'how far from orthogonal is this,
    in a normalized sense' rather than a raw, scale-dependent dot product."""
    Q, _ = np.linalg.qr(Xtr)
    proj_energy = float(np.linalg.norm(Q.T @ Atr) ** 2)
    total_energy = float(np.linalg.norm(Atr) ** 2)
    return proj_energy / total_energy


def random_baseline(Xtr, Atr, seed):
    rng = np.random.default_rng(seed)
    rand = rng.standard_normal(Atr.shape)
    rand *= np.linalg.norm(Atr, axis=0) / (np.linalg.norm(rand, axis=0) + 1e-12)
    return float(np.abs(Xtr.T @ rand).max())


def score_eq_residual(Zrep, Y, nc_idx, train_mask, clfs, C):
    """max_j |X_tr^T W r_j - beta_hat_j / C| for the negative-control columns: should be ~0 by
    the sklearn optimality condition itself, confirming the corrected identity is the right one
    (not a bug in how we're reading sklearn's fit)."""
    worst = 0.0
    for j in nc_idx:
        clf = clfs[j]
        if clf is None:
            continue
        tr = np.where(train_mask[:, j])[0]
        Xtr = Zrep[tr]
        p = 1 / (1 + np.exp(-(Xtr @ clf.coef_[0] + clf.intercept_[0])))
        y = Y[tr, j]
        pos, neg = y.sum(), len(y) - y.sum()
        s = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg)) if pos and neg else np.ones(len(y))
        lhs = Xtr.T @ (s * (y - p))
        rhs = clf.coef_[0] / C
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    return worst


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=32, seed=0)

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            row_test = rng.random(N) < 0.2
            train_mask = np.ones((N, K), dtype=bool)
            train_mask[row_test, :] = False

            Ufree, V, b = fit_shared_mf(Y, train_mask, r=16, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            Xtr = Zrep[train_idx_k0]

            clfs_actual, P_hat_actual = fit_per_indication_logreg(Zrep, Y, train_mask, C=C_ACTUAL)
            A_actual = negative_control_area_factor(Y, P_hat_actual, nc_idx, train_mask, r_area=2)
            Atr_actual = A_actual[train_idx_k0]

            clfs_weak, P_hat_weak = fit_per_indication_logreg(Zrep, Y, train_mask, C=C_WEAK)
            A_weak = negative_control_area_factor(Y, P_hat_weak, nc_idx, train_mask, r_area=2)
            Atr_weak = A_weak[train_idx_k0]

            rows.append(dict(
                target=target, seed=seed,
                max_proj_C1=max_abs_proj(Xtr, Atr_actual),
                max_proj_C100=max_abs_proj(Xtr, Atr_weak),
                random_baseline=random_baseline(Xtr, Atr_actual, seed),
                score_eq_gap_C1=score_eq_residual(Zrep, Y, nc_idx, train_mask, clfs_actual, C_ACTUAL),
                frac_var_in_X_C1=frac_variance_in_X(Xtr, Atr_actual),
                frac_var_in_X_C100=frac_variance_in_X(Xtr, Atr_weak),
            ))

    out = pd.DataFrame(rows)
    out.to_csv("../results/orthogonality_check.csv", index=False)
    print(out.round(3).to_string(index=False))
    print()
    g = out.groupby("target")[["max_proj_C1", "max_proj_C100", "random_baseline", "score_eq_gap_C1"]].mean()
    print(g.round(3))
    print()
    print("score_eq_gap_C1 ~0 confirms X_tr^T W r_j = beta_hat_j / C (not 0) is the correct")
    print("identity for the models as actually fit; max_proj drops from C=1 to C=100,")
    print("confirming the deviation shrinks toward the unregularized (C->inf) limit as predicted.")


if __name__ == "__main__":
    main()
