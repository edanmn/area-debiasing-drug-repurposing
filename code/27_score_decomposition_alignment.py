"""Three-score decomposition of the alignment check (a sixth round of review, pasted inline, Issue
1, High: "add a small table comparing the diagnostic metric under all three scores"). Table 3
reports AUROC(P, flag) where P = s_joint - s_proposed = gamma^T a, the dropped factor's own
contribution at scoring time. This review points out that a high top-50 overlap between s_plain and
s_proposed (code/19_intervention_doseresponse.py, a fifth-round addition) still leaves up to 16% of
top-50 membership differing between them, and asks how much of that full plain-to-proposed change is
attributable to dropping the factor specifically versus to REFITTING the base coefficients jointly
with it, a distinction P alone does not make since both s_joint and s_proposed already use the
refit coefficients.

Three logit-scale differences, named for which pair of the three scores they compare, not assumed
to decompose additively in AUROC space even though they do in raw logit space (AUROC is a nonlinear
functional of a score, so AUROC(a+b) is not generally a function of AUROC(a) and AUROC(b) alone;
checked empirically below, not assumed):
  diff_drop         = s_joint_logit - s_proposed_logit = P  (identical to Table 3's "Proposed's
                       term"; recomputed here and checked against results/permutation_check.csv's
                       observed_auroc, same "re-derived quantity drifts silently" guard as
                       code/25_hierarchical_permutation.py)
  diff_plain_to_proposed = s_proposed_logit - s_plain_logit  (A's own contribution excluded from
                       both sides, since s_proposed already drops it; this IS the complete,
                       directly-reported plain-to-proposed empirical gap, not a partial piece of it,
                       the quantity this review asked for by name)
  diff_plain_to_joint = s_joint_logit - s_plain_logit = diff_drop + diff_plain_to_proposed  (the
                       same comparison against the UNDROPPED joint fit, included for the cancellation
                       check below; this is not itself a score the paper reports anywhere, since the
                       proposed method always drops A at scoring time)

Each is scored, exactly as Table 3 scores P, as a sign-free AUROC classifier of each target's bias
flags (same-ATC-area, SIDER hard-negative, proconvulsant), on the same CVS2 candidate rows, 5 seeds.

Writes: results/score_decomposition_alignment.csv
"""
import sys
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
SEEDS = range(5)


def sign_free_auroc(flag, score):
    a = roc_auc_score(flag, score)
    return max(a, 1 - a)


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

        for seed in SEEDS:
            train_mask, test_rows = run_eval.make_masks(N, K, k0, "CVS2", seed)
            Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
            Zrep = np.hstack([Xchem, Ufree])
            clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
            clf_plain = clfs[k0]
            if clf_plain is None:
                continue
            plain_logit = Zrep @ clf_plain.coef_[0] + clf_plain.intercept_[0]

            A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
            train_idx_k0 = np.where(train_mask[:, k0])[0]
            clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
            A_oos = project_area_factor_oos(A, Zrep, train_idx_k0)
            coef_base = clf_db.coef_[0][:n_base]
            coef_A = clf_db.coef_[0][n_base:]
            base_logit = Zrep @ coef_base + clf_db.intercept_[0]
            diff_drop = A_oos @ coef_A
            diff_plain_to_proposed = base_logit - plain_logit
            diff_plain_to_joint = diff_plain_to_proposed + diff_drop

            cand = test_rows[y_full[test_rows] == 0]

            for flag_name, flag in flags.items():
                if flag[cand].sum() in (0, len(cand)):
                    continue
                auroc_drop = sign_free_auroc(flag[cand], diff_drop[cand])
                ref = prior[(prior.target == target) & (prior.flag == flag_name)
                             & (prior.seed == seed)].observed_auroc
                if len(ref) and abs(auroc_drop - ref.values[0]) > 1e-9:
                    raise RuntimeError(
                        f"diff_drop AUROC for {target}/{flag_name}/seed{seed} = {auroc_drop} != "
                        f"permutation_check.csv's {ref.values[0]}; this script's fit has drifted "
                        f"from 23_permutation_check.py's. Fix before trusting anything below.")
                rows.append(dict(
                    target=target, flag=flag_name, seed=seed,
                    auroc_diff_drop=auroc_drop,
                    auroc_diff_plain_to_proposed=sign_free_auroc(flag[cand], diff_plain_to_proposed[cand]),
                    auroc_diff_plain_to_joint=sign_free_auroc(flag[cand], diff_plain_to_joint[cand]),
                ))

    out = pd.DataFrame(rows)
    out.to_csv("../results/score_decomposition_alignment.csv", index=False)
    print(out.groupby(["target", "flag"])[
        ["auroc_diff_drop", "auroc_diff_plain_to_proposed", "auroc_diff_plain_to_joint"]
    ].mean().round(3).to_string())
    print("\nWrote results/score_decomposition_alignment.csv")


if __name__ == "__main__":
    main()
