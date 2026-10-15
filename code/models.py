"""The proposed method and all baselines.

Revision note (logged honestly rather than silently): an initial version of this module used ONLY
the raw compound x indication co-occurrence matrix (no chemical structure) to learn the shared
compound embedding U, exactly mirroring a textbook logistic matrix factorization. A diagnostic
(see logs/model_diagnostics.log) showed this gets essentially NO signal even on TRAINING data for
the three primary indications (epilepsy/depression/antipsychotic train AUPRC indistinguishable from
base rate), because with only 18 indications and 94% of the 9,878-compound library having zero
indications at all, there is nowhere near enough cross-indication co-occurrence for pure MF to
learn from -- a real, reportable negative finding, not a bug (confirmed by checking that increasing
rank/epochs/learning rate did not help). This matches why NRLMF/NRBdMF themselves do not rely on
the interaction matrix alone: they add drug/disease SIMILARITY-based "neighborhood regularization".
We do the same here in the simplest faithful way: a fixed low-rank PCA of each compound's ECFP4
fingerprint is concatenated to the free MF embedding to form the shared compound representation Z.
This is used identically by the plain baseline, the naive-subtraction baseline, and the proposed
method (so the only difference the evaluation can attribute to each method is its own mechanism,
not an unrelated change in feature set). NRBdMF is deliberately NOT given chemistry, matching
Azuma et al. 2022's own method, which factorizes only the signed drug x disease matrix.

Pipeline:
  1. fit_shared_mf: free compound embedding U_free (N x r) via logistic MF on Y (TRAIN cells only).
  2. chem_pca: fixed TruncatedSVD of ECFP4 (N x r_chem), computed once, no train/test leakage risk
     (it never looks at Y).
  3. Z = hstack([chem_pca, U_free])  -- the shared compound representation used by every method.
  4. fit_per_indication_logreg(Z, Y, train_mask): one logistic regression per indication (this IS
     the "plain" baseline).
  5. Proposed method: for target k0 with negative controls NC(k0), take the TRAIN residual of the
     already-fit per-indication models on NC(k0)'s columns, SVD it (RUV-2-style factor analysis
     restricted to negative controls) to get an area factor A (N x r_area), refit k0 WITH [Z, A],
     then "project out" A at scoring time (score uses only the Z-coefficients).
  6. naive_subtract: plain_score - lambda * raw mean indication-rate over NC(k0) (no factor
     analysis, no joint fit -- this is exactly what makes it "naive").
  7. nrbdmf: NRBdMF-style signed (-1 indication / +1 SIDER-opposite-effect) matrix factorization of
     Y and the hard-negative hit matrix, independent of Z/chemistry, within-drug only.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def chem_pca(ecfp, r_chem=32, seed=0):
    svd = TruncatedSVD(n_components=r_chem, random_state=seed)
    Z = svd.fit_transform(ecfp)
    return (Z - Z.mean(0)) / (Z.std(0) + 1e-6)


def fit_shared_mf(Y, train_mask, r=16, n_epochs=300, lr=0.1, l2=1e-3, seed=0):
    """Free logistic MF embedding, fit on TRAIN cells only. Kept as a genuine cross-indication
    latent-interaction component of the shared compound representation (see module docstring for
    why it is combined with chemistry rather than used alone)."""
    rng = np.random.default_rng(seed)
    N, K = Y.shape
    U = rng.normal(0, 0.05, size=(N, r))
    V = rng.normal(0, 0.05, size=(K, r))
    b = np.zeros(K)
    M = train_mask.astype(np.float32)
    for ep in range(n_epochs):
        Z = U @ V.T + b[None, :]
        P = sigmoid(Z)
        G = (P - Y) * M
        gU = G @ V / max(M.sum(), 1) + l2 * U
        gV = G.T @ U / max(M.sum(), 1) + l2 * V
        gb = G.sum(0) / np.maximum(M.sum(0), 1)
        U -= lr * gU; V -= lr * gV; b -= lr * gb
    return U, V, b


def fit_per_indication_logreg(Zrep, Y, train_mask, C=1.0):
    """One logistic regression per indication column, each fit on its own train rows. Returns the
    list of fitted classifiers (or None if degenerate) and the resulting P_hat (N x K) using TRAIN
    rows' own fitted model applied to every row (standard "plain" prediction)."""
    N, K = Y.shape
    clfs = []
    P_hat = np.zeros((N, K))
    for k in range(K):
        tr = np.where(train_mask[:, k])[0]
        ytr = Y[tr, k]
        if ytr.sum() < 2 or ytr.sum() == len(ytr):
            clfs.append(None)
            P_hat[:, k] = ytr.mean() if len(ytr) else 0.0
            continue
        clf = LogisticRegression(max_iter=500, C=C, class_weight="balanced").fit(Zrep[tr], ytr)
        clfs.append(clf)
        P_hat[:, k] = sigmoid(Zrep @ clf.coef_[0] + clf.intercept_[0])
    return clfs, P_hat


def negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2):
    """RUV-2-style step: SVD of the train-only residual on the negative-control columns, using the
    ALREADY-FIT per-indication P_hat (so chemistry + free-MF signal already explained is removed
    before estimating the area factor -- the area factor is UNWANTED variation left over, not raw
    indication membership)."""
    if len(nc_idx) == 0:
        return np.zeros((Y.shape[0], r_area))
    R = Y[:, nc_idx] - P_hat[:, nc_idx]
    mask_nc = train_mask[:, nc_idx]
    R = R * mask_nc
    Uu, S, Vt = np.linalg.svd(R, full_matrices=False)
    r_area = min(r_area, Uu.shape[1])
    return Uu[:, :r_area] * S[:r_area][None, :]


def fit_with_extra(Zrep, extra, y, train_idx, C=1.0):
    """Fit logistic regression on [Zrep, extra] (extra may be None). Returns (clf, n_base) where
    n_base=Zrep.shape[1], so callers can 'project out' any extra columns at scoring time."""
    X = Zrep if extra is None else np.hstack([Zrep, extra])
    Xtr, ytr = X[train_idx], y[train_idx]
    if ytr.sum() < 2 or ytr.sum() == len(ytr):
        return None, Zrep.shape[1]
    clf = LogisticRegression(max_iter=500, C=C, class_weight="balanced").fit(Xtr, ytr)
    return clf, Zrep.shape[1]


def score_project(clf, Zrep, n_base):
    """Score using only the first n_base coefficients (drops any appended 'extra' / area-factor
    coefficients) -- the RUV 'project out the unwanted factor' step. For the plain model (no
    extra features were ever appended) this is just the ordinary fitted score."""
    if clf is None:
        return np.zeros(len(Zrep))
    coef = clf.coef_[0][:n_base]
    return sigmoid(Zrep @ coef + clf.intercept_[0])


def raw_area_rate(Y, nc_idx):
    if len(nc_idx) == 0:
        return np.zeros(Y.shape[0])
    return Y[:, nc_idx].mean(1)


def nrbdmf_signed_matrix(Y, hard_hits, train_mask=None, r=12, n_epochs=300, lr=0.1, l2=1e-3, seed=0):
    """NRBdMF-style (Azuma et al. 2022) signed matrix factorization: -1 for ChEMBL/ATC indication,
    +1 for a SIDER characteristic-adverse-effect hit, 0/missing otherwise. Factorized directly
    (masked SGD, only nonzero entries contribute), independent of chemistry -- a WITHIN-DRUG
    bidirectional signal, never using other diseases as negative controls.

    train_mask (N x K boolean), if given, is intersected with the nonzero mask so TEST cells never
    contribute to the factorization -- required for a fair CVS1/CVS2/CVS3 comparison against the
    other methods (an earlier version of this function ignored train_mask entirely, which leaked
    test-set labels into the NRBdMF baseline's fit; logged in logs/model_diagnostics.log)."""
    N, K = Y.shape
    S = np.zeros((N, K), dtype=np.float32)
    S[Y.astype(bool)] = -1.0
    S[(hard_hits) & (S == 0)] = 1.0
    mask = (S != 0).astype(np.float32)
    if train_mask is not None:
        mask = mask * train_mask.astype(np.float32)
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.05, size=(N, r))
    V = rng.normal(0, 0.05, size=(K, r))
    for ep in range(n_epochs):
        Zhat = U @ V.T
        G = (Zhat - S) * mask
        denom = max(mask.sum(), 1)
        gU = G @ V / denom + l2 * U
        gV = G.T @ U / denom + l2 * V
        U -= lr * gU; V -= lr * gV
    Shat = U @ V.T
    return S, Shat, mask
