"""Figures for the paper, built strictly from results/*.csv (no numbers invented in-figure)."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "../results/"
FIG = "../figures/"

METHOD_ORDER = ["plain", "naive_subtract", "nrbdmf", "proposed_negctrl"]
METHOD_LABEL = {"plain": "Plain (no correction)", "naive_subtract": "Naive area subtraction",
                "nrbdmf": "NRBdMF-style signed MF", "proposed_negctrl": "Proposed (neg-control area factor)"}
COLORS = {"plain": "#888888", "naive_subtract": "#d62728", "nrbdmf": "#9467bd",
          "proposed_negctrl": "#2ca02c"}


def fig_bemegride():
    df = pd.read_csv(RES + "fig1_candidate_ranking_epilepsy.csv").sort_values("rank").head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(df.name[::-1], df.score[::-1], color="#1f77b4")
    ax.set_xlabel("plain ECFP4 model score")
    ax.set_title("Top 10 non-AED candidates for epilepsy\n(plain fingerprint model, reproduced)")
    fig.tight_layout()
    fig.savefig(FIG + "fig1_bemegride_top10.png", dpi=150)
    plt.close(fig)


def fig_neurology_enrichment():
    df = pd.read_csv(RES + "fig1_neurology_enrichment.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df.base_rate_neurology, width=0.35, label="base rate (candidate pool)", color="#888888")
    ax.bar(x + 0.18, df.top50_neurology_rate_mean_over_seeds, width=0.35,
           label="top-50 false positives", color="#d62728")
    ax.errorbar(x + 0.18, df.top50_neurology_rate_mean_over_seeds,
                yerr=[df.top50_neurology_rate_mean_over_seeds - df.top50_neurology_rate_ci_lo,
                      df.top50_neurology_rate_ci_hi - df.top50_neurology_rate_mean_over_seeds],
                fmt="none", color="black", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(df.disease)
    ax.set_ylabel("fraction ATC Level-1 'Nervous System'")
    ax.set_title("Neurology enrichment among top-50 false positives\n(reproduced, plain model)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG + "fig1_neurology_enrichment.png", dpi=150)
    plt.close(fig)


def fig_auprc_by_method():
    df = pd.read_csv(RES + "eval_summary.csv")
    sub = df[df.metric == "auprc"]
    targets = sub.target.unique()
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        s = sub[sub.target == target]
        cvss = sorted(s.cvs.unique())
        width = 0.2
        xx = np.arange(len(cvss))
        for j, m in enumerate(METHOD_ORDER):
            vals, los, his = [], [], []
            for c in cvss:
                row = s[(s.cvs == c) & (s.method == m)]
                if len(row):
                    vals.append(row["mean"].values[0])
                    los.append(row["mean"].values[0] - row["ci_lo"].values[0])
                    his.append(row["ci_hi"].values[0] - row["mean"].values[0])
                else:
                    vals.append(np.nan); los.append(0); his.append(0)
            ax.bar(xx + (j - 1.5) * width, vals, width=width, label=METHOD_LABEL[m], color=COLORS[m])
            ax.errorbar(xx + (j - 1.5) * width, vals, yerr=[los, his], fmt="none", color="black",
                        capsize=2, linewidth=0.7)
        ax.set_xticks(xx); ax.set_xticklabels(cvss)
        ax.set_title(target)
        ax.set_ylabel("AUPRC" if target == targets[0] else "")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("AUPRC on true indications, by method and CV scheme (mean +/- bootstrap 95% CI over 5 reruns)")
    fig.tight_layout()
    fig.savefig(FIG + "eval_auprc_by_method.png", dpi=150)
    plt.close(fig)


def fig_hard_negative_ranks():
    df = pd.read_csv(RES + "eval_summary.csv")
    sub = df[df.metric == "hard_neg_median_pct"]
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    targets = sub.target.unique()
    cvss = sorted(sub.cvs.unique())
    labels, vals, los, his, colors = [], [], [], [], []
    for target in targets:
        for c in cvss:
            for m in METHOD_ORDER:
                row = sub[(sub.target == target) & (sub.cvs == c) & (sub.method == m)]
                if len(row):
                    labels.append(f"{target}\n{c}\n{m}")
                    vals.append(row["mean"].values[0])
                    los.append(row["mean"].values[0] - row["ci_lo"].values[0])
                    his.append(row["ci_hi"].values[0] - row["mean"].values[0])
                    colors.append(COLORS[m])
    xx = np.arange(len(vals))
    ax.bar(xx, vals, color=colors)
    ax.errorbar(xx, vals, yerr=[los, his], fmt="none", color="black", capsize=2, linewidth=0.7)
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_ylabel("median rank percentile of hard negatives\n(lower = ranked higher = worse)")
    ax.set_title("Hard-negative (SIDER opposite-effect) ranking by method")
    fig.tight_layout()
    fig.savefig(FIG + "eval_hard_negative_ranks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_bemegride()
    fig_neurology_enrichment()
    fig_auprc_by_method()
    fig_hard_negative_ranks()
    print("Wrote figures to ../figures/")
