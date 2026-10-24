"""Reviewer roadmap Gap 2: one merged five-method CVS2 table (plain, naive_subtract,
proposed_negctrl, direct_subtract, random_direct_subtract/random_refit_drop) with paired
differences and CIs, instead of the numbers scattered across 6+ separate CSVs from the prior
two reviewer rounds. Reads existing CSVs only -- no new experiments, no refitting.

Produces, for EACH representation (chemistry+MF, the paper's existing primary; and chemistry-only,
code/33's new primary candidate):
  results/merged_method_table_chemmf.csv       -- target x method, auprc/area_rate/hardneg_pct/
                                                    proconv_pct means+CIs, n_pos/n_neg_hardneg/n_pos_proconv
  results/merged_method_table_chemonly.csv     -- same, chemistry-only representation
  results/paired_contrasts.csv                 -- paired per-seed differences (both representations),
                                                    for proposed_negctrl-vs-naive_subtract and
                                                    {proposed_negctrl,direct_subtract}-vs-their matched
                                                    random-factor-control counterpart, with bootstrap CIs
                                                    over the 5 matched CVS2 seeds
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from common import bootstrap_ci

TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
METHODS = ["plain", "naive_subtract", "proposed_negctrl", "direct_subtract",
           "random_direct_subtract", "random_refit_drop"]


def load_chemmf():
    """Chemistry+MF numbers come from three separate pipelines (06, 31, 32), each with its own
    per-seed CSV triple (auprc, enrich, hardneg). Concatenate into one per-seed frame per metric."""
    auprc = pd.read_csv("../results/eval_auprc.csv")
    auprc = auprc[(auprc.cvs == "CVS2") & auprc.method.isin(["plain", "naive_subtract", "proposed_negctrl"])]
    auprc_ds = pd.read_csv("../results/direct_subtraction_real_data.csv")
    auprc_rand = pd.read_csv("../results/random_factor_control.csv")
    auprc_all = pd.concat([auprc[["target", "method", "seed", "auprc", "n_test", "n_pos"]],
                           auprc_ds[["target", "method", "seed", "auprc", "n_test", "n_pos"]],
                           auprc_rand[["target", "method", "seed", "auprc", "n_test", "n_pos"]]],
                          ignore_index=True)

    enrich = pd.read_csv("../results/eval_area_enrichment.csv")
    enrich = enrich[(enrich.cvs == "CVS2") & enrich.method.isin(["plain", "naive_subtract", "proposed_negctrl"])]
    enrich_ds = pd.read_csv("../results/direct_subtraction_real_data_enrich.csv")
    enrich_rand = pd.read_csv("../results/random_factor_control_enrich.csv")
    enrich_all = pd.concat([enrich[["target", "method", "seed", "top50_rate"]],
                            enrich_ds[["target", "method", "seed", "top50_rate"]],
                            enrich_rand[["target", "method", "seed", "top50_rate"]]],
                           ignore_index=True)

    hn = pd.read_csv("../results/eval_hard_negatives.csv")
    hn = hn[(hn.cvs == "CVS2") & hn.method.isin(["plain", "naive_subtract", "proposed_negctrl"])]
    hn_ds = pd.read_csv("../results/direct_subtraction_real_data_hardneg.csv")
    hn_rand = pd.read_csv("../results/random_factor_control_hardneg.csv")
    cols = ["target", "method", "seed", "hard_neg_median_pct", "n_hard_neg", "proconv_median_pct", "n_proconv"]
    hn_all = pd.concat([hn[cols], hn_ds[cols], hn_rand[cols]], ignore_index=True)
    return auprc_all, enrich_all, hn_all


def load_chemonly():
    auprc = pd.read_csv("../results/chemistry_only_five_method.csv")[
        ["target", "method", "seed", "auprc", "n_test", "n_pos"]]
    enrich = pd.read_csv("../results/chemistry_only_five_method_enrich.csv")[
        ["target", "method", "seed", "top50_rate"]]
    hn = pd.read_csv("../results/chemistry_only_five_method_hardneg.csv")[
        ["target", "method", "seed", "hard_neg_median_pct", "n_hard_neg", "proconv_median_pct", "n_proconv"]]
    return auprc, enrich, hn


def build_table(auprc, enrich, hn, label):
    rows = []
    for target in TARGETS:
        for method in METHODS:
            a = auprc[(auprc.target == target) & (auprc.method == method)]
            e = enrich[(enrich.target == target) & (enrich.method == method)]
            h = hn[(hn.target == target) & (hn.method == method)]
            if len(a) == 0:
                continue
            am, alo, ahi = bootstrap_ci(a.auprc.values)
            em, elo, ehi = bootstrap_ci(e.top50_rate.values) if len(e) else (np.nan,) * 3
            hvals = h.hard_neg_median_pct.dropna().values
            hm, hlo, hhi = bootstrap_ci(hvals) if len(hvals) else (np.nan,) * 3
            row = dict(target=target, method=method,
                       auprc_mean=am, auprc_ci_lo=alo, auprc_ci_hi=ahi,
                       area_rate_mean=em, area_rate_ci_lo=elo, area_rate_ci_hi=ehi,
                       hardneg_pct_mean=hm, hardneg_pct_ci_lo=hlo, hardneg_pct_ci_hi=hhi,
                       n_pos=int(a.n_pos.iloc[0]) if len(a) else np.nan,
                       n_neg_hardneg=int(h.n_hard_neg.mean()) if len(h) else np.nan)
            if target == "epilepsy":
                pvals = h.proconv_median_pct.dropna().values
                if len(pvals):
                    pm, plo, phi = bootstrap_ci(pvals)
                    row.update(proconv_pct_mean=pm, proconv_pct_ci_lo=plo, proconv_pct_ci_hi=phi,
                               n_pos_proconv=int(h.n_proconv.mean()))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"../results/merged_method_table_{label}.csv", index=False)
    return df


def paired_contrast(auprc, enrich, hn, target, method_a, method_b, metric_df, metric_col, metric_name):
    a = metric_df[(metric_df.target == target) & (metric_df.method == method_a)][["seed", metric_col]]
    b = metric_df[(metric_df.target == target) & (metric_df.method == method_b)][["seed", metric_col]]
    m = a.merge(b, on="seed", suffixes=("_a", "_b")).dropna()
    if len(m) < 2:
        return None
    diff = (m[metric_col + "_a"] - m[metric_col + "_b"]).values
    mean_diff, lo, hi = bootstrap_ci(diff)
    return dict(target=target, method_a=method_a, method_b=method_b, metric=metric_name,
                n_seeds=len(m), mean_diff=mean_diff, ci_lo=lo, ci_hi=hi)


def main():
    chemmf = load_chemmf()
    chemonly = load_chemonly()
    build_table(*chemmf, "chemmf")
    build_table(*chemonly, "chemonly")

    contrast_rows = []
    for label, (auprc, enrich, hn) in [("chemmf", chemmf), ("chemonly", chemonly)]:
        for target in TARGETS:
            for ma, mb, metric_df, metric_col, metric_name in [
                ("proposed_negctrl", "naive_subtract", auprc, "auprc", "auprc"),
                ("proposed_negctrl", "naive_subtract", enrich, "top50_rate", "area_rate"),
                ("proposed_negctrl", "naive_subtract", hn, "hard_neg_median_pct", "hardneg_pct"),
                ("proposed_negctrl", "random_refit_drop", auprc, "auprc", "auprc"),
                ("proposed_negctrl", "random_refit_drop", enrich, "top50_rate", "area_rate"),
                ("proposed_negctrl", "random_refit_drop", hn, "hard_neg_median_pct", "hardneg_pct"),
                ("direct_subtract", "random_direct_subtract", auprc, "auprc", "auprc"),
                ("direct_subtract", "random_direct_subtract", enrich, "top50_rate", "area_rate"),
                ("direct_subtract", "random_direct_subtract", hn, "hard_neg_median_pct", "hardneg_pct"),
            ]:
                r = paired_contrast(auprc, enrich, hn, target, ma, mb, metric_df, metric_col, metric_name)
                if r:
                    r["representation"] = label
                    contrast_rows.append(r)
            if target == "epilepsy":
                for ma, mb in [("proposed_negctrl", "naive_subtract"),
                                ("proposed_negctrl", "random_refit_drop"),
                                ("direct_subtract", "random_direct_subtract")]:
                    r = paired_contrast(auprc, enrich, hn, target, ma, mb, hn, "proconv_median_pct", "proconv_pct")
                    if r:
                        r["representation"] = label
                        contrast_rows.append(r)

    contrasts_df = pd.DataFrame(contrast_rows)
    contrasts_df.to_csv("../results/paired_contrasts.csv", index=False)
    print(contrasts_df.round(4).to_string(index=False))
    print("\nWrote results/merged_method_table_chemmf.csv, merged_method_table_chemonly.csv, paired_contrasts.csv")


if __name__ == "__main__":
    main()
