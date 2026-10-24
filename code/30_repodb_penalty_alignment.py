"""W1 (reviewer feedback round 3): does the alignment check's penalty term help distinguish
genuinely harmful confounding from biologically meaningful similarity, against a trusted outcome
rather than only the bias-diagnostic metric being criticized?

repoDB (code/21_repodb_validation.py, results/repodb_validation.csv) already gives an independent
approved/failed label for depression, antipsychotic, and epilepsy -- the only 3 targets where
repoDB's indication-term matching avoids the mechanism-contamination problem documented in that
script (betablocker/lipid were tried there and dropped; this script inherits that same restriction
and does not revisit it).

Scope, stated plainly: this tests only "approved vs failed for the target's OWN indication", not
"approved for a different same-area indication" (i.e. one of the target's own negative controls),
since repoDB's indication-term map was only ever built, and only safely so, for the 3 core targets
themselves. Extending it to each negative control's own indication family would hit the same
contamination risk repoDB_validation.py already found for betablocker/lipid, so it is not attempted
here. What this DOES test: among compounds the model treats as non-indicated candidates (y=0),
does a high penalty term disproportionately fall on compounds repoDB says truly failed for this
disease (supporting "the penalty flags a nuisance"), or on compounds repoDB says are actually
approved for this disease despite the existing ChEMBL-derived label missing them (supporting "the
penalty is suppressing a real, mislabeled positive")?

Penalty terms, same quantities and same out-of-sample fix as code/13_naive_vs_proposed_alignment.py:
  raw_rate     = naive subtraction's penalty (raw_area_rate)
  area_penalty = proposed method's penalty (A_oos @ dropped area-factor coefficients)

Pooling: the repoDB-evidence subset is small (see results/repodb_validation.csv's n_pooled), so
following that script's own precedent, this pools each compound's out-of-fold score across
whichever of the 5 CVS2 seeds held it out, rather than reporting 5 separate small-n AUROCs.

Writes: results/repodb_penalty_alignment.csv
"""
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, raw_area_rate, project_area_factor_oos)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGETS_AND_TERMS = {
    "depression": ["depress"],
    "antipsychotic": ["schizophreni", "psychosis", "psychotic"],
    "epilepsy": ["epilep", "seizure", "convuls"],
}
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32


def load_repodb_labels(terms):
    approved, failed = set(), set()
    with open("../data/repoDB.csv") as f:
        for row in csv.DictReader(f):
            name = row["ind_name"].lower()
            if any(t in name for t in terms):
                dn = row["drug_name"].lower().strip()
                if row["status"] == "Approved":
                    approved.add(dn)
                elif row["status"] in ("Terminated", "Suspended", "Withdrawn"):
                    failed.add(dn)
    return approved, failed


def safe_auroc(y, s):
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    cmp_name_lower = cmp.name.str.lower().str.strip()

    rows = []
    for target, terms in TARGETS_AND_TERMS.items():
        k0 = col_idx[target]
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)

        approved, failed = load_repodb_labels(terms)
        is_approved = cmp_name_lower.isin(approved).values
        is_failed = cmp_name_lower.isin(failed).values & ~is_approved
        evidence = is_approved | is_failed

        pooled_naive, pooled_area = {}, {}
        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            cand = test_rows[y_full[test_rows] == 0]
            ev_cand = cand[evidence[cand]]
            if len(ev_cand) == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            if clfs[k0] is None or len(nc_idx) == 0:
                continue
            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            if clf_db is None:
                continue
            coef_A = clf_db.coef_[0][n_base:]
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            area_penalty = A_oos @ coef_A
            raw_rate = raw_area_rate(Y, nc_idx)
            for i in ev_cand:
                pooled_naive.setdefault(i, []).append(raw_rate[i])
                pooled_area.setdefault(i, []).append(area_penalty[i])

        pooled_idx = sorted(pooled_naive.keys())
        n_evidence = len(pooled_idx)
        n_failed = int(is_failed[pooled_idx].sum()) if pooled_idx else 0
        n_approved_mislabeled = int(is_approved[pooled_idx].sum()) if pooled_idx else 0
        print(f"{target}: {n_evidence} non-indicated candidates with repoDB evidence pooled across "
              f"folds ({n_failed} failed, {n_approved_mislabeled} approved-but-ChEMBL-negative)")

        row = dict(target=target, n_evidence=n_evidence, n_failed=n_failed,
                   n_approved_mislabeled=n_approved_mislabeled)
        if n_evidence < 5 or n_failed == 0 or n_approved_mislabeled == 0:
            print(f"{target}: too small or one-sided to estimate AUROC reliably -- not reported, "
                  f"counts above are the only usable output.")
            row["auroc_naive_vs_failed"] = np.nan
            row["auroc_areapenalty_vs_failed"] = np.nan
        else:
            y_failed = is_failed[pooled_idx].astype(int)
            mean_naive = np.array([np.mean(pooled_naive[i]) for i in pooled_idx])
            mean_area = np.array([np.mean(pooled_area[i]) for i in pooled_idx])
            row["auroc_naive_vs_failed"] = safe_auroc(y_failed, mean_naive)
            row["auroc_areapenalty_vs_failed"] = safe_auroc(y_failed, mean_area)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../results/repodb_penalty_alignment.csv", index=False)
    print(out)
    print("\n0.5 = penalty carries no information distinguishing repoDB-failed from repoDB-approved-"
          "but-mislabeled among non-indicated candidates. >0.5 = higher penalty falls more on "
          "repoDB-failed (nuisance-consistent). <0.5 = higher penalty falls more on repoDB-approved-"
          "but-mislabeled (removing real signal).")
    print("Wrote results/repodb_penalty_alignment.csv")


if __name__ == "__main__":
    main()
