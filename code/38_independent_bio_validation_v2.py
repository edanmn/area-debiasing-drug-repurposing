"""Reviewer request (second-round review, LLM reviewer): "Use a separate set of clinically
supported positives and negatives for epilepsy, excluded from model fitting and hyperparameter
selection. Establish whether the correction reduces the ranking of known seizure-provoking drugs
while retaining actual antiepileptics." This script is the Step-3 follow-up to
code/36_independent_biological_validation.py, which already used the 43-drug Chen/Albertson/Olson
2016 proconvulsant panel (cross-validated against Ruffmann/Bogliun/Beghi 2006). That docstring
noted a prior fork's WebSearch attempt at a SECOND independent source did not turn up anything
"distinct enough to be worth adding." This script reports a genuinely new attempt and what it found.

WHAT WAS CHECKED FOR A SECOND INDEPENDENT PROCONVULSANT SOURCE (seizure-risk side), and why each
candidate was accepted or rejected -- see the full search log in the fork's final report; summary:
  - Further peer-reviewed narrative reviews of drug-induced seizures (e.g. Larson et al. 2021,
    Int J Toxicol; Mula 2026, Pract Neurol; a 2020 Seizure-journal review of antibiotic-induced
    seizures) were REJECTED as not independent: they recursively cite the same case-report
    literature chain as Chen 2016/Ruffmann 2006 and name almost exactly the same drugs (checked by
    fetching the Larson et al. 2021 risk table, which lists penicillin/clozapine/flumazenil/
    meperidine/cephalosporins/carbapenems/fluoroquinolones as "high/medium risk" -- all already in
    the existing 43-drug panel).
  - DrugBank's structured ADR/seizure annotations: inaccessible without a paid license, and even if
    accessible would be sourced from the same package-insert ADR pipeline as SIDER (already used
    inside the existing panel as sider_convulsion_flag), so would not be independent either way.
  - WHO VigiBase/VigiAccess: no public bulk API (CAPTCHA-gated single-drug web lookups only);
    shares the same indication-confound problem as FAERS (below) without the ability to query it
    systematically.
  - FDA boxed warnings / drug-label "seizure" mentions via openFDA: REJECTED as an independent
    *source*, because SIDER's own ADR terms (already used in the existing panel) are themselves
    mined from these same FDA package-insert sections -- checking against them again would not be
    independent. (It WAS used downstream, below, as a specificity *filter* on a FAERS signal, which
    is a different, legitimate role: confirming a statistically-detected signal has a known
    mechanism of action documented in regulatory literature, not using the label as the discovery
    source itself.)
  - openFDA FAERS (raw co-report counts): tried first and found to be badly confounded by
    "indication channeling" -- the drugs most often reported alongside seizure-type MedDRA terms
    are the antiepileptic drugs THEMSELVES (levetiracetam, lamotrigine, carbamazepine, ...),
    because epilepsy patients already on an ASM are the ones who report breakthrough seizures. Raw
    FAERS counts are therefore unusable without (a) excluding known ASMs/ATC-N03 drugs and (b) a
    proper disproportionality statistic (PRR + chi-square), not raw counts.
  - openFDA FAERS with PRR/chi-square disproportionality analysis, restricted to the project's own
    562-drug "approved for >=1 of 18 indications" pool, EXCLUDING any compound already ATC-N03 or
    approved-for-epilepsy (to remove the indication-channeling confound above) and EXCLUDING the 43
    drugs already in the Chen/Ruffmann panel: this is the one candidate that worked. 56 non-AED,
    non-panel drug names matched the pool; 15 passed the standard pharmacovigilance significance
    screen (PRR>=2, chi2>=4, a>=3 seizure co-reports). Of those 15, 9 were ALSO independently
    confirmed to list seizure/convulsion in their own FDA structured product label's adverse-
    reactions section (a specificity filter, not the discovery source -- see above), and have a
    plausible pharmacological mechanism; the other 6 (tofacitinib, candesartan, diclofenac,
    temozolomide, naproxen, ramipril) were EXCLUDED despite passing the statistical screen: no FDA
    label confirmation (tofacitinib, candesartan, diclofenac) or strong suspicion of residual
    comorbidity/channeling confound given the indication itself (temozolomide is used almost
    exclusively in glioma patients, who have seizures from the tumor, not the chemo; naproxen/
    ramipril have no established convulsant mechanism and only a boilerplate rare-event label
    mention). The surviving 9-drug list is data/proconvulsants_v2_independent.csv. This falls short
    of the ~15-20 "ideal" size named in the task, but is a real, methodologically distinct
    (observational pharmacovigilance disproportionality, not literature curation) addition.

ANTIEPILEPTIC (true-positive, "must not be demoted") SIDE: no usable second independent list was
found. WHO's Model List of Essential Medicines names only 6 oral antiseizure medicines (too few to
be a standalone panel). The ILAE's 2024 Epilepsia position paper on medication terminology is a
naming-convention consensus document (whether to say "antiseizure medication" vs "antiepileptic
drug"), not an extensional drug list. Checking this project's own data/chembl_atc_n03.csv (WHO ATC
N03 codes) confirmed it is a strict SUBSET of the existing epilepsy "approved" label already used
as y_true (26 of 31 approved-for-epilepsy compounds carry an N03 code; the other 5 do not) -- using
it would be re-deriving the existing label, which the task explicitly warned against. This script
therefore reuses the SAME ATC/ChEMBL-approved epilepsy label as the true-positive target, but the
"retains real antiepileptics" claim rests on the CVS2 held-out test fold (those rows are excluded
from clf fitting, from negative_control_area_factor, and from the direct/naive lambda terms for
that fold, exactly as in 06_run_evaluation.py and 36_independent_biological_validation.py) -- i.e.
held out from model fitting in the correct cross-validation sense, even though it is not a second,
differently-curated clinical source.

Reports, for plain/naive_subtract/proposed_negctrl/direct_subtract (+ random-factor controls) on
epilepsy/CVS2/5 seeds: precision@10/@50 on the held-out epilepsy label; proconvulsant rate@10/@50
on BOTH the original 43-drug panel (for continuity with script 36) and the new 9-drug FAERS-PRR
panel; bemegride's rank (as in script 36); and quetiapine's rank (the new panel's strongest-PRR,
most clinically familiar entry) as an analogous named-drug sanity check.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, ".")
from common import D
from indication_matrix import build_matrix, negative_controls, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, project_area_factor_oos, sigmoid)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

TARGET = "epilepsy"
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
LAMBDA_NAIVE = 2.0
BEMEGRIDE_IK = "ORRZGUBHBVWWOP"
QUETIAPINE_IK = "URKOMYMAXPYINW"


def precision_at_k(scores, test_rows, y_true, k):
    order = test_rows[np.argsort(-scores[test_rows])]
    topk = order[:min(k, len(order))]
    return float(y_true[topk].mean()), topk


def named_drug_rank(scores, test_rows, ik, cmp_ik_values):
    rows = np.where(cmp_ik_values == ik)[0]
    row = int(rows[0]) if len(rows) else None
    if row is None or row not in test_rows:
        return None, np.nan, np.nan
    order = test_rows[np.argsort(-scores[test_rows])]
    rank = int(np.where(order == row)[0][0]) + 1
    pct = rank / len(order) * 100
    return row, rank, pct


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)

    proconv_v1 = pd.read_csv(D + "proconvulsants.csv")
    proconv_v1_iks = set(proconv_v1.loc[proconv_v1.in_lincs == True, "ik14"].dropna())
    mask_v1 = cmp.ik14.isin(proconv_v1_iks).values

    proconv_v2 = pd.read_csv(D + "proconvulsants_v2_independent.csv")
    proconv_v2_iks = set(proconv_v2.ik14.dropna())
    mask_v2 = cmp.ik14.isin(proconv_v2_iks).values

    print(f"v1 (Chen/Ruffmann) panel in pool: {mask_v1.sum()}/{len(proconv_v1_iks)} resolved IDs present")
    print(f"v2 (FAERS-PRR, independent) panel in pool: {mask_v2.sum()}/{len(proconv_v2_iks)} resolved IDs present")
    overlap = mask_v1 & mask_v2
    print(f"overlap between v1 and v2 panels in pool: {overlap.sum()} (should be 0 by construction)")

    k0 = col_idx[TARGET]
    nc_idx = [col_idx[d] for d in negative_controls(TARGET, diseases=cols)]
    y_full = Y[:, k0].astype(int)

    rows = []
    for seed in SEEDS:
        train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
        if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
            continue
        Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
        Zrep = np.hstack([Xchem, Ufree])
        clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
        clf_plain = clfs[k0]
        s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

        area_rate = raw_area_rate(Y, nc_idx)
        ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
        s_naive = s_plain - LAMBDA_NAIVE * ar

        A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
        train_idx_k0 = np.where(train_mask[:, k0])[0]
        clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
        s_proposed = score_project(clf_db, Zrep, n_base)

        A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
        ytr = y_full[train_idx_k0]
        logit_plain = clf_plain.decision_function(Zrep)
        s_direct = s_naive.copy()
        if ytr.sum() >= 2 and ytr.sum() < len(ytr):
            calib = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                A_oos[train_idx_k0], ytr)
            s_direct = sigmoid(logit_plain - A_oos @ calib.coef_[0])

        rng = np.random.RandomState(1000 + seed)
        R = rng.normal(0, A.std(), size=A.shape)
        clf_rdb, n_base_r = fit_with_extra(Zrep, R, y_full, train_idx_k0)
        s_rand_refit = score_project(clf_rdb, Zrep, n_base_r)
        s_rand_direct = s_naive.copy()
        if ytr.sum() >= 2 and ytr.sum() < len(ytr):
            calib_r = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced").fit(
                R[train_idx_k0], ytr)
            s_rand_direct = sigmoid(logit_plain - R @ calib_r.coef_[0])

        for name, s in [("plain", s_plain), ("naive_subtract", s_naive), ("proposed_negctrl", s_proposed),
                        ("direct_subtract", s_direct), ("random_refit_drop", s_rand_refit),
                        ("random_direct_subtract", s_rand_direct)]:
            p10, top10 = precision_at_k(s, test_rows, y_full, 10)
            p50, top50 = precision_at_k(s, test_rows, y_full, 50)
            v1_10 = float(mask_v1[top10].mean()) if len(top10) else np.nan
            v1_50 = float(mask_v1[top50].mean()) if len(top50) else np.nan
            v2_10 = float(mask_v2[top10].mean()) if len(top10) else np.nan
            v2_50 = float(mask_v2[top50].mean()) if len(top50) else np.nan
            _, beme_rank, beme_pct = named_drug_rank(s, test_rows, BEMEGRIDE_IK, cmp.ik14.values)
            _, quet_rank, quet_pct = named_drug_rank(s, test_rows, QUETIAPINE_IK, cmp.ik14.values)
            rows.append(dict(target=TARGET, seed=seed, method=name, precision_at_10=p10, precision_at_50=p50,
                             proconv_v1_rate_at_10=v1_10, proconv_v1_rate_at_50=v1_50,
                             proconv_v2_rate_at_10=v2_10, proconv_v2_rate_at_50=v2_50,
                             n_test=len(test_rows), n_pos=int(y_full[test_rows].sum()),
                             bemegride_rank=beme_rank, bemegride_pct=beme_pct,
                             quetiapine_rank=quet_rank, quetiapine_pct=quet_pct))

    df = pd.DataFrame(rows)
    df.to_csv("../results/independent_bio_validation_v2.csv", index=False)
    summary = df.groupby("method")[["precision_at_10", "precision_at_50",
                                     "proconv_v1_rate_at_10", "proconv_v1_rate_at_50",
                                     "proconv_v2_rate_at_10", "proconv_v2_rate_at_50",
                                     "bemegride_rank", "bemegride_pct",
                                     "quetiapine_rank", "quetiapine_pct"]].mean()
    print(summary.round(3).to_string())
    print("\nWrote results/independent_bio_validation_v2.csv")


if __name__ == "__main__":
    main()
