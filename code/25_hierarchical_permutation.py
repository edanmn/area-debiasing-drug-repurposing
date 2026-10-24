"""Compound-level (hierarchical) permutation test for Table 3 (a fourth round of review, pasted
inline, Issue C): code/23_permutation_check.py combines 5 per-seed p-values with Fisher's method,
which assumes the 5 tests are independent. They are not fully independent: all 5 CVS2 seeds are
re-splits of the SAME N-compound pool, so a given compound appears as a test-set candidate in
several seeds, with the same true flag value and a correlated (not independent) fitted area-penalty
value each time. Fisher's method, and the uncorrected 0.05 bar used for the leave-one-out
companion check in 23, both treat the 5 seeds as if they were 5 independent experiments.

This script builds a null distribution that respects that structure instead of assuming it away:
for each of M global permutations, draw ONE permutation sigma of the full N compound indices and
apply it to the ENTIRE flag vector once (shuffled_flag = flag[sigma]), then reuse that SAME
shuffled flag vector across all 5 seeds' own candidate sets. A compound that appears in several
seeds' candidate sets is reassigned the same shuffled flag value in every one of them, for a given
permutation draw, exactly mirroring how its TRUE flag value is fixed (not independently re-rolled)
across seeds in the real data. The combined test statistic is the sum of the 5 seeds' sign-free
AUROCs; the p-value is the fraction of the M global permutations whose combined statistic meets or
exceeds the one observed with the real (unpermuted) flag.

The per-seed fitting code below (representation, area-factor estimation, out-of-sample projection,
flag construction) is copied verbatim from code/23_permutation_check.py rather than re-derived, and
this script asserts its recomputed per-seed observed AUROCs match 23's saved
results/permutation_check.csv before trusting anything downstream -- the same "diagnostic quantity
re-derived instead of reused" failure mode that produced a real bug in an earlier version of 23
itself (see that script's docstring) is just as possible here if this script quietly used a
different flag or fit.

Writes: results/hierarchical_permutation.csv
"""
import sys
import zlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, project_area_factor_oos)
import importlib
run_eval = importlib.import_module("06_run_evaluation")

R_FREE, R_CHEM = 16, 32
M_GLOBAL_PERM = 20000
SEEDS = range(5)


def sign_free_auroc(flag, score):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


def sign_free_auroc_or_nan(flag, score):
    """Like sign_free_auroc, but returns nan instead of raising when a shuffled flag happens to be
    single-class within one seed's candidate set (undefined AUROC) -- a real possibility here since
    the global permutation can, by chance, assign a sparse flag (e.g. 6-14 proconvulsant positives)
    zero or all positives within a given seed's ~330-row candidate subset."""
    if flag.sum() == 0 or flag.sum() == len(flag):
        return np.nan
    return sign_free_auroc(flag, score)


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

    prior = pd.read_csv("../results/permutation_check.csv")

    rows = []
    unique_compound_rows = []
    for target in ["depression", "antipsychotic", "epilepsy"]:
        k0 = col_idx[target]
        area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
        same_area_flag = cmp.ik14.isin(area_iks).values
        nc_idx = [col_idx[d] for d in negative_controls(target, diseases=cols)]
        y_full = Y[:, k0].astype(int)
        hard_col = hard_hits[:, k0]

        flags = {"samearea": same_area_flag, "hardneg": hard_col}
        if target == "epilepsy":
            flags["proconv"] = proconv_mask_all

        per_seed = {}  # seed -> (area_penalty (len N), cand (indices))
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
            per_seed[seed] = (area_penalty, cand)

            for flag_name, flag in flags.items():
                if flag[cand].sum() == 0 or flag[cand].sum() == len(cand):
                    continue
                obs = sign_free_auroc(flag[cand], area_penalty[cand])
                ref = prior[(prior.target == target) & (prior.flag == flag_name)
                             & (prior.seed == seed)].observed_auroc
                if len(ref) and abs(obs - ref.values[0]) > 1e-9:
                    raise RuntimeError(
                        f"Recomputed observed AUROC for {target}/{flag_name}/seed{seed} = {obs} "
                        f"does not match permutation_check.csv's {ref.values[0]} -- this script's "
                        f"fit/flag has drifted from 23_permutation_check.py's. Fix before trusting "
                        f"any p-value below.")

        for flag_name, flag in flags.items():
            seeds_ok = [s for s in per_seed if flag[per_seed[s][1]].sum() not in (0, len(per_seed[s][1]))]
            if len(seeds_ok) < 2:
                continue
            penalty_cand = {s: per_seed[s][0][per_seed[s][1]] for s in seeds_ok}
            obs_sum = sum(sign_free_auroc(flag[per_seed[s][1]], penalty_cand[s]) for s in seeds_ok)

            rng = np.random.default_rng(zlib.crc32(f"{target}/{flag_name}".encode()))
            null_sums = np.empty(M_GLOBAL_PERM)
            for m in range(M_GLOBAL_PERM):
                sigma = rng.permutation(N)
                shuffled_flag = flag[sigma]
                null_sums[m] = sum(
                    sign_free_auroc_or_nan(shuffled_flag[per_seed[s][1]], penalty_cand[s])
                    for s in seeds_ok)
            valid = ~np.isnan(null_sums)
            n_dropped = int((~valid).sum())
            null_valid = null_sums[valid]
            # Permutation draws where a seed's shuffled flag degenerated to one class give an
            # undefined combined statistic for that draw; dropped from both numerator and
            # denominator (not silently counted as "did not exceed", which would bias p downward).
            p_hier = (1 + np.sum(null_valid >= obs_sum)) / (len(null_valid) + 1)

            rows.append(dict(target=target, flag=flag_name, n_seeds=len(seeds_ok),
                              observed_sum_auroc=obs_sum, p_hierarchical=p_hier,
                              n_valid_perms=len(null_valid), n_dropped_degenerate=n_dropped,
                              null_mean=null_valid.mean(), null_std=null_valid.std()))
            print(f"{target}/{flag_name}: observed_sum={obs_sum:.3f}  "
                  f"null mean={null_valid.mean():.3f} std={null_valid.std():.3f}  "
                  f"dropped={n_dropped}/{M_GLOBAL_PERM}  p_hierarchical={p_hier:.5f}")

        cand_iks_union = set()
        for seed, (_, cand) in per_seed.items():
            cand_iks_union.update(cmp.ik14.values[cand])
        unique_compound_rows.append(dict(target=target, n_unique_test_compounds=len(cand_iks_union),
                                          n_total_pool=N))

    out = pd.DataFrame(rows)
    out.to_csv("../results/hierarchical_permutation.csv", index=False)
    pd.DataFrame(unique_compound_rows).to_csv("../results/unique_compound_counts_hierarchical.csv", index=False)
    print("\nWrote results/hierarchical_permutation.csv, results/unique_compound_counts_hierarchical.csv")


if __name__ == "__main__":
    main()
