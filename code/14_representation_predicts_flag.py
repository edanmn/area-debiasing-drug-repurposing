"""Generalizes Section "alignment" from a single-dataset finding into a checkable precondition.

Section 13's result (naive's penalty term separates the bias-metric flags at 0.60-0.79 AUROC,
the proposed method's residualized penalty term sits at ~0.50 on every target/flag cell) has a
clean algebraic reason: the area factor A is constructed, by the SVD step in
negative_control_area_factor, as the residual of negative-control membership AFTER the
already-fit per-indication model's prediction P_hat (itself a function of the shared
representation Zrep = [chemistry PCA, free MF embedding]) is subtracted off. A is therefore
uncorrelated with P_hat's column space on train rows by construction, not by assumption; this
is an algebraic fact about the SVD-of-a-residual step; see paper Proposition 1.

That fact alone does not explain why A ends up uninformative about the bias metrics' own flags
(same-ATC-area membership, SIDER-hard-negative/proconvulsant membership). It explains that only
if those flags are themselves well predicted by the SAME representation Zrep that P_hat already
uses. This script tests that premise directly rather than asserting it: for each target and the
same CVS2 splits used throughout, fit a plain classifier on Zrep (train rows only) to predict
each flag, and report its AUROC on the exact same candidate population (non-indicated test
compounds) used by the bias metrics and by Section "alignment".

If this AUROC is well above chance, the premise holds in this dataset: the flags ride on the same
information channel the correction already explains away, so projecting out that channel leaves a
residual with nothing left to say about the flags. That completes the general diagnostic recipe:
(1) confirm the correction's "unwanted variation" term is orthogonal to the base representation by
construction, (2) check empirically whether the metric's flag is itself explainable from that base
representation. Both steps are general; neither is specific to drug repurposing or to this method.

Writes: results/representation_predicts_flag.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from common import D
from indication_matrix import build_matrix, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import fit_shared_mf, chem_pca
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32


def safe_auroc(flag, score):
    flag = np.asarray(flag)
    if flag.sum() == 0 or flag.sum() == len(flag):
        return np.nan
    return roc_auc_score(flag, score)


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)

    atc = pd.read_csv(D + "atc_level1.csv")
    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    rows = []
    for target in TARGETS:
        k0 = col_idx[target]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]
        area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
        same_area_flag = cmp.ik14.isin(area_iks).values

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            train_idx = np.where(train_mask[:, k0])[0]

            cand = test_rows[y_full[test_rows] == 0]
            if len(cand) == 0:
                continue

            def fit_and_score(flag):
                ytr = flag[train_idx].astype(int)
                if ytr.sum() < 2 or ytr.sum() == len(ytr):
                    return np.nan
                clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Zrep[train_idx], ytr)
                score = clf.predict_proba(Zrep[cand])[:, 1]
                return safe_auroc(flag[cand], score)

            row = dict(
                target=target, seed=seed, n_cand=len(cand),
                auroc_zrep_samearea=fit_and_score(same_area_flag),
                auroc_zrep_hardneg=fit_and_score(hard_col),
            )
            if target == "epilepsy":
                row["auroc_zrep_proconv"] = fit_and_score(proconv_mask_all)
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../results/representation_predicts_flag.csv", index=False)

    print("Mean AUROC (5 seeds): can Zrep = [chemistry PCA, free MF] alone, the SAME")
    print("representation the proposed method's P_hat is fit on, predict each bias-metric flag?")
    summary = out.groupby("target")[[c for c in out.columns if c.startswith("auroc_")]].mean()
    print(summary.round(3))
    print("\n0.5 = Zrep carries no information about that flag. Higher = better separation.")
    print("Wrote results/representation_predicts_flag.csv")


if __name__ == "__main__":
    main()
