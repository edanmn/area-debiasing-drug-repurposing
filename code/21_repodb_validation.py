"""repoDB secondary benchmark (Priority 4 in the assessment.txt fix plan, Related Work's own
framing: "repoDB... is a standard secondary comparison point in this literature").

Data: repoDB~\cite{brown2017repodb} (Brown & Patel 2017, CC-BY 4.0), pulled from the original
author's companion repo (github.com/zietzm/repoDB/data/repoDB.csv, itself built from
github.com/adam-sam-brown/repoDB -- the official apps.chiragjpgroup.org/repoDB download requires
an interactive Shiny session with no static file endpoint, so this is the closest available static
copy of the same CC-BY-licensed release) into data/repoDB.csv.

Join: by lowercased exact drug name (no DrugBank/InChIKey crosswalk exists in this project's data;
see code/common.py's load_compounds, which carries only ik14/name/smiles). 892 of repoDB's 1,572
drug names match a name in our 9,752-named-compound library -- an approximate, name-based join
with the same known limitations as the ATC/SIDER joins elsewhere in this paper, stated plainly.

Design: this does NOT refit anything on repoDB labels. It reuses the EXACT SAME trained objects
06_run_evaluation.py produces (fit on the existing ChEMBL/LINCS-derived Y, same CVS2 splits), and
at EVALUATION time only, checks each method's AUPRC/bias-metric behavior against an independently
sourced label for the subset of test compounds repoDB gives a definite answer for (its own
Approved vs Terminated/Suspended/Withdrawn status for an indication-matched UMLS concept). This
tests whether the pipeline's qualitative pattern survives judging the SAME model's output against
a label source it was never fit to, rather than fitting a new model.

Coverage check first (mechanism-vs-indication mismatch): depression/antipsychotic/epilepsy map
fairly cleanly to a repoDB indication-name family (depression, schizophrenia/psychosis, epilepsy/
seizure/convulsion) and show 82-86% agreement with the existing ChEMBL-derived label among
in-pool repoDB-approved compounds. betablocker/lipid were tried and dropped: repoDB's indication
terms (hypertension, angina, arrhythmia, hypercholesterolemia) are symptom/disease-level and match
many OTHER drug mechanisms too (ACE inhibitors, calcium-channel blockers, statins of a different
class, etc.), giving only 8/34 and leaving the comparison too contaminated by mechanism-mismatch to
be informative; reported as a negative finding, not silently dropped.

Writes: results/repodb_validation.csv
"""
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor, fit_with_extra, raw_area_rate
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
LAMBDA_NAIVE = 2.0


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
        is_repodb_approved = cmp_name_lower.isin(approved).values
        is_repodb_failed = cmp_name_lower.isin(failed).values & ~is_repodb_approved
        evidence = is_repodb_approved | is_repodb_failed
        y_repodb = is_repodb_approved.astype(int)
        agreement = (is_repodb_approved & (y_full == 1)).sum() / max(is_repodb_approved.sum(), 1)
        print(f"{target}: repoDB evidence for {evidence.sum()} pool compounds "
              f"({is_repodb_approved.sum()} approved, {is_repodb_failed.sum()} failed-only), "
              f"agreement with existing label = {agreement:.3f}")

        # Pool out-of-fold scores across all 5 seeds rather than computing AUPRC within each
        # seed's held-out fold: the repoDB-evidence subset is small enough (22-42 compounds per
        # target) that a single CVS2 fold's intersection with it is often all-positive or
        # all-negative (AUPRC trivially 1.0 or undefined), which is exactly the small-sample-noise
        # failure mode already flagged elsewhere in this project for sparse classes. Pooling the
        # out-of-fold score each evidence compound gets whenever it happens to be held out gives
        # one evaluation over the full evidence set instead of five degenerate fragments of it.
        from models import score_project
        pooled_scores = {"plain": {}, "naive": {}, "proposed": {}}
        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            ev_test = test_rows[evidence[test_rows]]
            if len(ev_test) == 0:
                continue
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None:
                continue
            s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])
            area_rate = raw_area_rate(Y, nc_idx)
            ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
            s_naive = s_plain - LAMBDA_NAIVE * ar
            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            s_proposed = score_project(clf_db, Zrep, n_base) if clf_db is not None else s_plain
            for name, s in [("plain", s_plain), ("naive", s_naive), ("proposed", s_proposed)]:
                for i in ev_test:
                    pooled_scores[name].setdefault(i, []).append(s[i])

        pooled_idx = sorted(pooled_scores["plain"].keys())
        if len(pooled_idx) < 5 or y_repodb[pooled_idx].sum() < 2 or y_repodb[pooled_idx].sum() == len(pooled_idx):
            print(f"{target}: pooled evidence set too small or degenerate "
                  f"(n={len(pooled_idx)}, n_pos={y_repodb[pooled_idx].sum() if pooled_idx else 0}) "
                  f"-- AUPRC not reported, only the label-agreement number above is usable.")
            rows.append(dict(target=target, agreement=agreement, n_pooled=len(pooled_idx),
                              n_pos_repodb=int(y_repodb[pooled_idx].sum()) if pooled_idx else 0))
            continue
        row = dict(target=target, agreement=agreement, n_pooled=len(pooled_idx),
                   n_pos_repodb=int(y_repodb[pooled_idx].sum()))
        for name in ["plain", "naive", "proposed"]:
            mean_scores = np.array([np.mean(pooled_scores[name][i]) for i in pooled_idx])
            row[f"auprc_repodb_{name}"] = average_precision_score(y_repodb[pooled_idx], mean_scores)
            row[f"auprc_chembl_{name}"] = average_precision_score(y_full[pooled_idx], mean_scores)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv("../results/repodb_validation.csv", index=False)
    print(out)
    print("\nWrote results/repodb_validation.csv")


if __name__ == "__main__":
    main()
