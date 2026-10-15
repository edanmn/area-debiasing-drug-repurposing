"""Step 1 (paper Figure 1): reproduce the motivating evidence BEFORE any new method is built.

(a) Retrain the plain ECFP4 fingerprint model (XGBoost, same hyperparameters as
    ICHI2027_mgan/code/candidates.py) for epilepsy, depression and antipsychotics on the drug-only
    candidate pool, and confirm bemegride ranks at/near #1 among non-AED candidates for epilepsy.
(b) For depression and antipsychotics, compute the top-50 false-positive "neurology" (ATC Level-1
    'N', Nervous System) enrichment vs the base rate in the candidate pool, using data/atc_level1.csv
    built by 02_fetch_atc_level1.py (must be run first).

Mirrors candidates.py's model exactly (5-seed-averaged XGBoost on ECFP4, GPU, drug-only pool) so the
numbers are a genuine re-run, not a re-read of the old csv. Writes results/fig1_candidate_ranking_<d>.csv
and results/fig1_neurology_enrichment.csv.
"""
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, ".")
from common import D, load_compounds, load_disease_labels, load_pool, load_ecfp, tanimoto_max, bootstrap_ci

DISEASES = ["epilepsy", "depression", "antipsychotic"]
SEEDS = range(5)


def load_disease_pool(disease):
    """Same contract as evallib.load(pool='drugs', disease=...): drug-only candidate pool,
    y=1 for approved/positive compounds of this indication."""
    cmp = load_compounds()
    lab = load_disease_labels(disease)
    cmp = cmp.set_index("ik14")
    cmp["approved"] = lab["positive"].reindex(cmp.index).fillna(False).astype(bool)
    cmp["clinical"] = lab["clinical"].reindex(cmp.index).fillna(False).astype(bool)
    cmp["max_phase"] = lab["max_phase"].reindex(cmp.index)
    cmp = cmp.reset_index()
    keep = load_pool("drugs") & cmp.has_structure.values
    # load_pool computed approved from compounds.parquet's own 'approved' col (epilepsy); for other
    # diseases we must redo the "approved OR max_phase>=1" logic with THIS disease's own labels.
    ph = pd.read_csv(D + "chembl_max_phase_all.csv").set_index("ik14").chembl_max_phase
    keep = cmp.has_structure.values & ((cmp.ik14.map(ph).fillna(0).values >= 1) | cmp.approved.values)
    ecfp_all = load_ecfp()
    ecfp = ecfp_all[keep]
    cmp = cmp[keep].reset_index(drop=True)
    y = cmp.approved.values.astype(int)
    return cmp, ecfp, y


def rank_candidates(disease):
    cmp, ecfp, y = load_disease_pool(disease)
    print(f"[{disease}] drug pool n={len(y)}, positives={y.sum()}")
    scores = np.zeros(len(y))
    for seed in SEEDS:
        m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.3, scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
                           tree_method="hist", device="cuda", random_state=seed, verbosity=0).fit(ecfp, y)
        scores += m.predict_proba(ecfp)[:, 1] / len(list(SEEDS))
    cmp = cmp.copy()
    cmp["score"] = scores
    if y.sum() > 0:
        cmp["nn_sim_to_pos"] = tanimoto_max(ecfp, ecfp[y == 1])
        inter = ecfp @ ecfp[y == 1].T
        union = ecfp.sum(1)[:, None] + ecfp[y == 1].sum(1)[None, :] - inter
        cmp["nn_pos_name"] = cmp.name[y == 1].values[(inter / np.maximum(union, 1)).argmax(1)]
    neg = cmp[y == 0].sort_values("score", ascending=False).reset_index(drop=True)
    neg["rank"] = np.arange(1, len(neg) + 1)
    neg.to_csv(f"../results/fig1_candidate_ranking_{disease}.csv", index=False)
    return neg


def neurology_flag(ik14_series):
    atc = pd.read_csv(D + "atc_level1.csv")
    neuro_iks = set(atc.loc[atc.atc_level1_code == "N", "ik14"])
    return ik14_series.isin(neuro_iks)


def main():
    rankings = {}
    for d in DISEASES:
        rankings[d] = rank_candidates(d)

    print("\n=== (a) bemegride check (epilepsy) ===")
    ep = rankings["epilepsy"]
    beme = ep[ep.name.str.lower() == "bemegride"]
    print(beme[["rank", "name", "score"]].to_string(index=False) if len(beme) else "bemegride not found in pool")
    print("\nTop 10 non-AED candidates for epilepsy:")
    print(ep.head(10)[["rank", "name", "score"]].to_string(index=False))

    print("\n=== (b) neurology enrichment in top-50 false positives ===")
    rows = []
    for d in ["depression", "antipsychotic"]:
        neg = rankings[d]
        neg_flag = neurology_flag(neg.ik14) if "ik14" in neg.columns else neurology_flag(neg.ik14)
        base_rate = neg_flag.mean()
        top50 = neg_flag.iloc[:50]
        top50_rate = top50.mean()
        # bootstrap CI over which-50 by re-scoring with each of the 5 seeds individually (variance
        # across reruns), not just a single point estimate.
        seed_rates = []
        cmp_full = load_compounds()
        for seed in SEEDS:
            cmp, ecfp, y = load_disease_pool(d)
            m = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.3, scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
                               tree_method="hist", device="cuda", random_state=seed, verbosity=0).fit(ecfp, y)
            s = m.predict_proba(ecfp)[:, 1]
            cmp = cmp.copy(); cmp["score"] = s
            neg_s = cmp[y == 0].sort_values("score", ascending=False).head(50)
            seed_rates.append(neurology_flag(neg_s.ik14).mean())
        mean_r, lo, hi = bootstrap_ci(seed_rates)
        rows.append(dict(disease=d, n_candidates=len(neg), base_rate_neurology=base_rate,
                          top50_neurology_rate_pooled=top50_rate,
                          top50_neurology_rate_mean_over_seeds=mean_r,
                          top50_neurology_rate_ci_lo=lo, top50_neurology_rate_ci_hi=hi,
                          seed_rates=str(seed_rates)))
        print(f"{d}: base_rate={base_rate:.3f}  top50_pooled={top50_rate:.3f}  "
              f"top50_mean(seeds)={mean_r:.3f} [{lo:.3f},{hi:.3f}]")
    pd.DataFrame(rows).to_csv("../results/fig1_neurology_enrichment.csv", index=False)
    print("\nWrote results/fig1_neurology_enrichment.csv and results/fig1_candidate_ranking_<disease>.csv")


if __name__ == "__main__":
    main()
