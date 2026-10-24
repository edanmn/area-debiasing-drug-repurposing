"""Permutation test for Table 3 (sec:alignment), prompted by a second external review
(assessment_updated.txt, Issue 6): max(AUROC, 1-AUROC) is upward-biased under the null when the
sign is picked on the same evaluation set the statistic is reported on, since |AUROC-0.5| has
positive expectation even when the true AUROC is exactly 0.5. This matters most for small,
imbalanced flag-positive counts, exactly the epilepsy hard-negative (36 positives) and
proconvulsant (11 positives) cells.

A third round of review (pasted inline, Issue 4 of that round) asked two further questions about this
test: (a) whether permutation respects the structure of each CVS2 split and how results are
aggregated across the 5 reruns, and (b) whether testing 7 cells (2 flags x depression, 2 x
antipsychotic, 3 x epilepsy) raises a multiple-comparisons concern.

On (a): permutation is done SEPARATELY within each seed's own candidate set (the real fitted area
penalty for that seed's CVS2 split, against 5000 reshufflings of that seed's own flag vector) --
never pooling candidate rows or penalty values across seeds, since the fitted penalty term and the
candidate set both depend on that seed's particular train/test split. The 5 per-seed p-values are
then combined with Fisher's method, -2*sum(log(p_i)) ~ chi2(2*5=10 dof), which is the standard way
to combine p-values from independent tests (here, independent random CVS2 splits) into one p-value,
rather than by averaging them (an earlier version of this script reported the arithmetic mean of
the 5 per-seed p-values, which has no formal justification as a combined significance test and is
replaced here).

On (b): with 7 cells tested, a Bonferroni-corrected threshold for the nominal 0.05 level is
0.05/7 = 0.0071. Four cells clear it on the full 5-seed Fisher combination: epilepsy's same-ATC-area
and proconvulsant flags, plus BOTH of antipsychotic's flags. The last two (and epilepsy's
hard-negative, which clears 0.05 uncorrected but not the Bonferroni threshold) collapse to clearly
non-significant when the single most extreme of the 5 seeds is excluded (p rises from <0.03 to
0.18-0.45): a seed-specific artifact of one random CVS2 split, not a stable effect. Epilepsy's
same-ATC-area and proconvulsant survive this leave-one-out check too (same-ATC-area: p<1e-9 either
way; proconvulsant: p=0.0004 all-5, p=0.01 with the most extreme seed removed).

Note: an earlier version of this script's "same-ATC-area" flag was built from this project's own
18-indication co-occurrence matrix rather than the external WHO ATC level-1 code that
run_eval.area_enrichment and 13_naive_vs_proposed_alignment.py (which produced Table 3's actual
numbers) use -- a different flag, silently swapped in. Caught by comparing this script's per-seed
AUROC against 13's output for the same seed/target. Fixed; numbers below are post-fix.

Result: depression's two flags and epilepsy's hard-negative are not confirmed on this test (the
hard-negative's apparent Bonferroni-level significance before the leave-one-out check is the
seed-specific artifact above). Two of epilepsy's three flags, same-ATC-area and proconvulsant, are
confirmed; so is neither of antipsychotic's two flags, despite looking Bonferroni-significant before
the robustness check -- the numeric AUROC values in Table 3 are not safe to read as "the
correction's own term carries real information" from a raw combined p-value alone, without also
checking it is not driven by a single seed.

Writes: results/permutation_check.csv (per-seed rows), results/permutation_check_combined.csv
(Fisher-combined p-value per target/flag cell, plus the Bonferroni-adjusted threshold)
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, project_area_factor_oos)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

R_FREE, R_CHEM = 16, 32
N_PERM = 5000
SEEDS = range(5)
N_CELLS = 7  # 2 flags x depression, 2 x antipsychotic, 3 x epilepsy -- for Bonferroni correction
P_FLOOR = 1.0 / (N_PERM + 1)  # a permutation p of exactly 0 means "< 1/(n_perm+1)", not literally 0


def fisher_combine(pvals):
    pvals = np.clip(np.asarray(pvals, dtype=float), P_FLOOR, 1.0)
    stat = -2.0 * np.log(pvals).sum()
    return stat, chi2.sf(stat, df=2 * len(pvals))


def sign_free_auroc(flag, score):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


def perm_pvalue(flag, score, n_perm=N_PERM, seed=0):
    obs = sign_free_auroc(flag, score)
    rng = np.random.default_rng(seed)
    flag = np.asarray(flag)
    count = sum(sign_free_auroc(rng.permutation(flag), score) >= obs for _ in range(n_perm))
    return obs, count / n_perm


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    col_idx = {d: j for j, d in enumerate(cols)}
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)
    hard_hits = run_eval.sider_hit_matrix(cmp, cols)
    proconv = pd.read_csv("../data/proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values

    atc = pd.read_csv("../data/atc_level1.csv")

    rows = []
    unique_compound_rows = []
    for target in ["depression", "antipsychotic", "epilepsy"]:
        k0 = col_idx[target]
        cand_iks_union = set()
        # same-ATC-area flag: external WHO ATC level-1 code, matching run_eval.area_enrichment and
        # 13_naive_vs_proposed_alignment.py (the script that actually produced Table 3's numbers) --
        # NOT this project's own 18-indication co-occurrence matrix, a different definition an
        # earlier version of this script used by mistake, which silently tested the wrong flag.
        area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
        same_area_flag = cmp.ik14.isin(area_iks).values
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]

        flags = {"samearea": same_area_flag, "hardneg": hard_col}
        if target == "epilepsy":
            flags["proconv"] = proconv_mask_all

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            if clfs[k0] is None:
                continue
            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            coef_A = clf_db.coef_[0][n_base:]
            area_penalty = A_oos @ coef_A
            cand = test_rows[y_full[test_rows] == 0]
            cand_iks_union.update(cmp.ik14.values[cand])

            for flag_name, flag in flags.items():
                obs, pv = perm_pvalue(flag[cand], area_penalty[cand], seed=seed)
                rows.append(dict(target=target, flag=flag_name, seed=seed,
                                  n_cand=len(cand), n_pos=int(flag[cand].sum()),
                                  observed_auroc=obs, perm_pvalue=pv))

        unique_compound_rows.append(dict(target=target, n_unique_test_compounds=len(cand_iks_union),
                                          n_total_pool=N))

    out = pd.DataFrame(rows)
    out.to_csv("../results/permutation_check.csv", index=False)

    combined_rows = []
    for (target, flag), g in out.groupby(["target", "flag"]):
        stat, p_fisher = fisher_combine(g.perm_pvalue.values)
        # Leave-one-out robustness: refuse to call a cell confirmed if its combined significance
        # depends on a single one of the 5 seeds -- drop whichever seed has the smallest individual
        # p (the most influential one) and recombine the remaining 4.
        imin = g.perm_pvalue.values.argmin()
        _, p_excl = fisher_combine(np.delete(g.perm_pvalue.values, imin))
        alpha = 0.05 / N_CELLS
        combined_rows.append(dict(target=target, flag=flag, n_seeds=len(g),
                                   mean_observed_auroc=g.observed_auroc.mean(),
                                   fisher_chi2=stat, fisher_pvalue=p_fisher,
                                   fisher_pvalue_excl_most_extreme_seed=p_excl,
                                   bonferroni_alpha=alpha,
                                   significant_bonferroni=p_fisher < alpha,
                                   robust_to_leave_one_out=(p_fisher < alpha) and (p_excl < 0.05)))
    combined = pd.DataFrame(combined_rows)
    combined.to_csv("../results/permutation_check_combined.csv", index=False)
    pd.DataFrame(unique_compound_rows).to_csv("../results/unique_compound_counts_permutation.csv", index=False)
    print(out.groupby(["target", "flag"])[["observed_auroc", "perm_pvalue"]].mean().round(3))
    print()
    print(combined.round(4))
    print("\nWrote results/permutation_check.csv, results/permutation_check_combined.csv,")
    print("      results/unique_compound_counts_permutation.csv")


if __name__ == "__main__":
    main()
