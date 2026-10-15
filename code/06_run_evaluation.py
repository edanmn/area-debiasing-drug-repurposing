"""Main evaluation: runs plain / naive-subtract / NRBdMF-style / proposed(negative-control) on the
SAME compound x indication matrix (drug-candidate pool, chemistry + free-MF shared representation,
see models.py/indication_matrix.py and logs/model_diagnostics.log for why) and the SAME CVS1/CVS2/
CVS3 splits, for depression and antipsychotic (primary) and epilepsy (secondary).

Writes:
  results/eval_auprc.csv
  results/eval_area_enrichment.csv
  results/eval_hard_negatives.csv
  results/eval_summary.csv   -- means + bootstrap 95% CIs across 5 reruns (seeds)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import D, bootstrap_ci
from indication_matrix import build_matrix, negative_controls, AREA_OF, ALL_DISEASES
from hard_negatives import HARD_NEG_TERMS
from models import (fit_shared_mf, chem_pca, fit_per_indication_logreg, negative_control_area_factor,
                     fit_with_extra, score_project, raw_area_rate, nrbdmf_signed_matrix)
from sklearn.metrics import average_precision_score

TARGETS = ["depression", "antipsychotic", "epilepsy"]
SEEDS = [0, 1, 2, 3, 4]
R_FREE = 16
R_CHEM = 32
LAMBDA_NAIVE = 2.0


def make_masks(N, K, k0_idx, cvs, seed):
    rng = np.random.default_rng(seed)
    train_mask = np.ones((N, K), dtype=bool)
    if cvs == "CVS1":
        cell_test = rng.random((N, K)) < 0.2
        train_mask[cell_test] = False
        test_rows = np.where(cell_test[:, k0_idx])[0]
    elif cvs == "CVS2":
        row_test = rng.random(N) < 0.2
        train_mask[row_test, :] = False
        test_rows = np.where(row_test)[0]
    elif cvs == "CVS3":
        train_mask[:, k0_idx] = False
        test_rows = np.arange(N)
    else:
        raise ValueError(cvs)
    return train_mask, test_rows


def sider_hit_matrix(cmp, cols):
    sider = pd.read_parquet(D + "sider_full.parquet")
    ik_to_terms = sider.groupby("ik14").se_name.apply(lambda s: set(x.lower() for x in s))
    hit = np.zeros((len(cmp), len(cols)), dtype=bool)
    for j, d in enumerate(cols):
        terms = HARD_NEG_TERMS.get(d)
        if not terms:
            continue
        for i, ik in enumerate(cmp.ik14):
            s = ik_to_terms.get(ik)
            if s and (s & terms):
                hit[i, j] = True
    return hit


def area_enrichment(cmp, scores, y_true, target, test_rows, k=50):
    atc = pd.read_csv(D + "atc_level1.csv")
    area_iks = set(atc.loc[atc.atc_level1_code == AREA_OF[target], "ik14"])
    flag = cmp.ik14.isin(area_iks).values
    sub_rows = test_rows[y_true[test_rows] == 0]
    if len(sub_rows) == 0:
        return np.nan, np.nan
    order = sub_rows[np.argsort(-scores[sub_rows])]
    base_rate = flag[sub_rows].mean()
    topk = order[:min(k, len(order))]
    return flag[topk].mean(), base_rate


def hard_neg_rank_pct(scores, test_rows, hard_mask_col):
    rows = test_rows[hard_mask_col[test_rows]]
    if len(rows) == 0:
        return np.nan
    order = np.argsort(-scores[test_rows])
    ranks = np.empty(len(test_rows)); ranks[order] = np.arange(1, len(test_rows) + 1)
    pos_of_test = {r: i for i, r in enumerate(test_rows)}
    pct = [ranks[pos_of_test[r]] / len(test_rows) * 100 for r in rows]
    return float(np.median(pct))


def main():
    cmp, Y, cols, ecfp = build_matrix(ALL_DISEASES, pool="drugs")
    N, K = Y.shape
    print(f"Drug-pool indication matrix: {N} compounds x {K} indications")
    col_idx = {d: j for j, d in enumerate(cols)}
    hard_hits = sider_hit_matrix(cmp, cols)
    Xchem = chem_pca(ecfp, r_chem=R_CHEM, seed=0)  # fixed, leakage-free (never looks at Y)

    proconv = pd.read_csv(D + "proconvulsants.csv")
    proconv_iks = set(proconv.loc[proconv.in_lincs == True, "ik14"].dropna())
    proconv_mask_all = cmp.ik14.isin(proconv_iks).values
    print(f"Proconvulsant panel resolved in drug pool: {proconv_mask_all.sum()} compounds")

    auprc_rows, enrich_rows, hardneg_rows = [], [], []

    for target in TARGETS:
        k0 = col_idx[target]
        nc_names = negative_controls(target, diseases=cols)
        nc_idx = [col_idx[d] for d in nc_names]
        print(f"\n### target={target}  negative controls={nc_names}")
        hard_col = hard_hits[:, k0]
        y_full = Y[:, k0].astype(int)

        for cvs in ["CVS1", "CVS2", "CVS3"]:
            for seed in SEEDS:
                train_mask, test_rows = make_masks(N, K, k0, cvs, seed)
                if len(test_rows) == 0 or y_full[test_rows].sum() == 0:
                    continue
                Ufree, V, b = fit_shared_mf(Y, train_mask, r=R_FREE, seed=seed)
                Zrep = np.hstack([Xchem, Ufree])

                # --- plain: one logistic regression per indication on Zrep ---
                clfs, P_hat = fit_per_indication_logreg(Zrep, Y, train_mask)
                clf_plain = clfs[k0]
                cvs3_degenerate = clf_plain is None  # target column had 0 training rows (CVS3)
                if cvs3_degenerate:
                    # True cold-start-disease: no target labels exist to fit ANYTHING
                    # target-specific. The only available signal is transfer from the
                    # mechanistically-related negative-control indications (see
                    # logs/model_diagnostics.log for why this replaces an earlier, meaningless
                    # all-zero-score degenerate case).
                    s_plain = P_hat[:, nc_idx].mean(1) if len(nc_idx) else np.full(N, y_full.mean())
                else:
                    s_plain = score_project(clf_plain, Zrep, Zrep.shape[1])

                # --- naive area-score subtraction ---
                area_rate = raw_area_rate(Y, nc_idx)
                ar = (area_rate - area_rate.min()) / (area_rate.max() - area_rate.min() + 1e-9)
                s_naive = s_plain - LAMBDA_NAIVE * ar

                # --- NRBdMF-style signed MF (chemistry-free, within-drug only); train_mask
                # respected so no test-label leakage (see logs/model_diagnostics.log) ---
                S, Shat, maskS = nrbdmf_signed_matrix(Y, hard_hits, train_mask=train_mask, r=12, seed=seed)
                s_nrbdmf = -Shat[:, k0]

                # --- proposed: negative-control area-factor removal ---
                if len(nc_idx) > 0 and not cvs3_degenerate:
                    A = negative_control_area_factor(Y, P_hat, nc_idx, train_mask, r_area=2)
                    train_idx_k0 = np.where(train_mask[:, k0])[0]
                    clf_db, n_base = fit_with_extra(Zrep, A, y_full, train_idx_k0)
                    s_debiased = score_project(clf_db, Zrep, n_base)
                else:
                    # CVS3: the proposed method's own machinery needs target labels to estimate a
                    # target-specific residual; with zero available it provably reduces to the
                    # same cross-indication transfer as "plain" (documented limitation, not a bug).
                    s_debiased = s_plain.copy()

                for name, s in [("plain", s_plain), ("naive_subtract", s_naive),
                                 ("nrbdmf", s_nrbdmf), ("proposed_negctrl", s_debiased)]:
                    yt = y_full[test_rows]
                    if yt.sum() == 0 or yt.sum() == len(yt):
                        continue
                    ap = average_precision_score(yt, s[test_rows])
                    auprc_rows.append(dict(target=target, cvs=cvs, method=name, seed=seed, auprc=ap,
                                           n_test=len(test_rows), n_pos=int(yt.sum())))
                    top_rate, base_rate = area_enrichment(cmp, s, y_full, target, test_rows)
                    enrich_rows.append(dict(target=target, cvs=cvs, method=name, seed=seed,
                                            top50_rate=top_rate, base_rate=base_rate))
                    hn_pct = hard_neg_rank_pct(s, test_rows, hard_col)
                    row = dict(target=target, cvs=cvs, method=name, seed=seed,
                               hard_neg_median_pct=hn_pct, n_hard_neg=int(hard_col[test_rows].sum()),
                               proconv_median_pct=np.nan, n_proconv=0)
                    if target == "epilepsy":
                        row["proconv_median_pct"] = hard_neg_rank_pct(s, test_rows, proconv_mask_all)
                        row["n_proconv"] = int(proconv_mask_all[test_rows].sum())
                    hardneg_rows.append(row)
            print(f"  {cvs} done")

    auprc_df = pd.DataFrame(auprc_rows)
    enrich_df = pd.DataFrame(enrich_rows)
    hardneg_df = pd.DataFrame(hardneg_rows)
    auprc_df.to_csv("../results/eval_auprc.csv", index=False)
    enrich_df.to_csv("../results/eval_area_enrichment.csv", index=False)
    hardneg_df.to_csv("../results/eval_hard_negatives.csv", index=False)

    summary = []
    for (target, cvs, method), g in auprc_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.auprc.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="auprc",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=len(g)))
    for (target, cvs, method), g in enrich_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.top50_rate.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="top50_area_rate",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=len(g)))
        m2, lo2, hi2 = bootstrap_ci(g.base_rate.values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="area_base_rate",
                            mean=m2, ci_lo=lo2, ci_hi=hi2, n_seeds=len(g)))
    for (target, cvs, method), g in hardneg_df.groupby(["target", "cvs", "method"]):
        m, lo, hi = bootstrap_ci(g.hard_neg_median_pct.dropna().values)
        summary.append(dict(target=target, cvs=cvs, method=method, metric="hard_neg_median_pct",
                            mean=m, ci_lo=lo, ci_hi=hi, n_seeds=g.hard_neg_median_pct.notna().sum()))
        if target == "epilepsy" and g.proconv_median_pct.notna().any():
            m3, lo3, hi3 = bootstrap_ci(g.proconv_median_pct.dropna().values)
            summary.append(dict(target=target, cvs=cvs, method=method, metric="proconv_median_pct",
                                mean=m3, ci_lo=lo3, ci_hi=hi3, n_seeds=g.proconv_median_pct.notna().sum()))
    pd.DataFrame(summary).to_csv("../results/eval_summary.csv", index=False)
    print("\nWrote results/eval_auprc.csv, eval_area_enrichment.csv, eval_hard_negatives.csv, eval_summary.csv")


if __name__ == "__main__":
    main()
