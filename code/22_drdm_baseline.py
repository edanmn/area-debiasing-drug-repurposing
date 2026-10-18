"""DRDM baseline (Priority 5 in the assessment.txt fix plan): "complete the comparison with DRDM"
(Meng et al. 2025, cited as meng2025drdmb, "Adaptive Debiasing Learning for Drug Repositioning").

Honesty note, read before using any number from this script: DRDM has no public code (checked:
no GitHub repo linked from the paper, Mount Sinai Scholars page, or PubMed record). Its
ScienceDirect page (checked again after a second external review questioned the "abstract-only"
claim) actually exposes a fair amount: full abstract, full introduction with literature context,
a "statement of significance" table, and a one-line "Materials and methods" stub ("Fig. 2
illustrates the overall framework of DRDM"). The exact loss equations for "dynamically adjust[ing]
association weights" and "dual-view contrastive learning" are still paywalled (no PMC, arXiv, or
author-manuscript copy found), so the first honesty issue (no equations) stands. But a second,
more basic one is now clear from the visible introduction: DRDM dynamically reweights messages at
each propagation step of a GNN over a heterogeneous drug-disease graph (popular entities get
smaller aggregation weight, long-tail entities larger), evaluated on Fdataset/Cdataset/LRSSL.
There is no graph or propagation step anywhere in this project; Zrep is a flat per-compound vector.
Full access to DRDM's text would not make a faithful reproduction possible here, since the
architecture itself has no counterpart in this pipeline, a separate and more fundamental gap than
the paywall. What follows is therefore a reconstruction of the *motivation* (down-weight popular
entities, up-weight long-tail ones, add a contrastive auxiliary signal), not of DRDM's actual
mechanism, which is GNN-specific and does not transfer. This is stated explicitly here and must be
stated explicitly in any paper text that cites this result, the same way NRBdMF (azuma2022nrbdmf)
is labeled "matching the original specification" when it IS fully specified from a public, read
paper -- the opposite label is required when it is not.

Reconstruction, adapted to this project's compound x indication matrix (using the same chemistry+MF
shared representation Zrep every other method in this paper uses, kept as two separate "views"
instead of one concatenated vector so a dual-view contrastive term has something to contrast):

  1. Dynamic reweighting: each compound's per-epoch training weight combines (a) a STATIC
     long-tail popularity term, 1/degree^p where degree is the compound's total positive-indication
     count across all 18 classes (DRDM's "node polarization" -- popular vs long-tail entities), and
     (b) a DYNAMIC per-epoch term that up-weights examples the current-epoch model still gets most
     wrong (an approximation of DRDM's "dynamically adjusts... during training", since the exact
     update rule is not available).
  2. Dual-view contrastive learning: an InfoNCE-style loss between a chemistry-view projection and
     an MF-view projection of the same batch of compounds (same compound = positive pair, other
     batch members = negatives), added to the weighted classification loss, providing the "extra
     supervisory signal" the abstract describes.

Writes: results/drdm_baseline.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import chem_pca, fit_shared_mf, sigmoid
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
N_EPOCHS = 200
LR = 0.1
L2 = 1e-3
LAMBDA_CONTRASTIVE = 0.1
POPULARITY_POWER = 0.3
PROJ_DIM = 16
TEMP = 0.1


def drdm_score(Zchem, Zmf, Y, train_idx, k0, seed=0):
    rng = np.random.default_rng(seed)
    N = Zchem.shape[0]
    Zrep = np.hstack([Zchem, Zmf])
    d, d_chem, d_mf = Zrep.shape[1], Zchem.shape[1], Zmf.shape[1]
    w = rng.normal(0, 0.01, size=d)
    b = 0.0
    Wc = rng.normal(0, 0.05, size=(d_chem, PROJ_DIM))
    Wm = rng.normal(0, 0.05, size=(d_mf, PROJ_DIM))

    degree = Y.sum(1)
    pop_weight = 1.0 / np.power(degree + 1.0, POPULARITY_POWER)
    pop_weight = pop_weight / pop_weight[train_idx].mean()
    y = Y[:, k0].astype(float)
    tr = train_idx
    batch_size = min(128, len(tr))

    for epoch in range(N_EPOCHS):
        logits = Zrep[tr] @ w + b
        p = sigmoid(logits)
        resid_weight = 0.5 + np.abs(y[tr] - p)
        sw = pop_weight[tr] * resid_weight
        sw = sw / sw.mean()
        grad_logit = (p - y[tr]) * sw
        gw = Zrep[tr].T @ grad_logit / len(tr) + L2 * w
        gb = grad_logit.mean()
        w -= LR * gw
        b -= LR * gb

        batch = rng.choice(tr, size=batch_size, replace=False)
        zc = Zchem[batch] @ Wc
        zm = Zmf[batch] @ Wm
        zc = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + 1e-8)
        zm = zm / (np.linalg.norm(zm, axis=1, keepdims=True) + 1e-8)
        sim = (zc @ zm.T) / TEMP
        sim_max = sim.max(1, keepdims=True)
        exp_sim = np.exp(sim - sim_max)
        softmax = exp_sim / exp_sim.sum(1, keepdims=True)
        grad_sim = softmax.copy()
        grad_sim[np.arange(batch_size), np.arange(batch_size)] -= 1
        grad_sim *= LAMBDA_CONTRASTIVE / TEMP
        gWc = Zchem[batch].T @ (grad_sim @ zm) / batch_size
        gWm = Zmf[batch].T @ (grad_sim.T @ zc) / batch_size
        Wc -= LR * gWc
        Wm -= LR * gWm

    return sigmoid(Zrep @ w + b)


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        y_full = Y[:, k0].astype(int)
        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Ufree, V, b_ = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            train_idx = np.where(train_mask[:, k0])[0]
            if y_full[train_idx].sum() < 2:
                continue
            s_drdm = drdm_score(Xchem, Ufree, Y, train_idx, k0, seed=seed)
            area_rate_top50, base_rate = run_eval.area_enrichment(cmp, s_drdm, y_full, target, test_rows)
            rows.append(dict(
                target=target, seed=seed,
                auprc=average_precision_score(y_full[test_rows], s_drdm[test_rows]),
                area_rate_top50=area_rate_top50, area_base_rate=base_rate,
            ))

    out = pd.DataFrame(rows)
    out.to_csv("../results/drdm_baseline.csv", index=False)
    print(out.groupby("target")[["auprc", "area_rate_top50", "area_base_rate"]].mean().round(3))
    print("\nWrote results/drdm_baseline.csv")
    print("Reminder: this is a reconstruction from the published abstract, not a verified "
          "faithful reproduction -- no public code or accessible full methods text exists.")


if __name__ == "__main__":
    main()
