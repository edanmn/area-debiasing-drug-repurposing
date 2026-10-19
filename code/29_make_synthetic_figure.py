"""Figure for the synthetic ground-truth experiment and noise sweep (a sixth round of review,
pasted inline, Issue 2, High: "replace some descriptive text with a figure containing the
correction comparison and complete seven-level noise sweep", since these dense-prose results are
"among the strongest in the paper"). Built strictly from results/synthetic_groundtruth.csv and
results/confound_noise_sweep.csv (both already 30-seed), no numbers invented in-figure.

Writes: figures/fig2_synthetic_correction_gap.pdf
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES = "../results/"
FIG = "../figures/"


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    return m, half


def main():
    synth = pd.read_csv(RES + "synthetic_groundtruth.csv")
    sweep = pd.read_csv(RES + "confound_noise_sweep.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 1.85))

    # Panel A: correction comparison at the real (unswept) estimation quality, 30 seeds.
    methods = ["top50_confound_plain", "top50_confound_naive", "top50_confound_proposed",
               "top50_confound_direct_subtract", "top50_confound_oracle"]
    labels = ["plain", "naive\nsubtract", "proposed\n(fit-project)", "direct\nsubtract", "oracle"]
    colors = ["#888888", "#d62728", "#2ca02c", "#1f77b4", "#7f7f7f"]
    means, halfs = [], []
    for c in methods:
        m, h = ci95(synth[c])
        means.append(m); halfs.append(h)
    base_rate = synth["top50_confound_base_rate"].mean()
    x = np.arange(len(methods))
    ax1.bar(x, means, yerr=halfs, capsize=3, color=colors)
    ax1.axhline(base_rate, color="black", ls="--", lw=1, label=f"base rate ({base_rate:.2f})")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("top-50 true-confound rate")
    ax1.set_title("(a) correction comparison, 30 seeds\n(95% CI)")
    ax1.legend(fontsize=7, loc="upper left")

    # Panel B: complete 7-level noise sweep, both application mechanisms, 30 seeds each.
    sigmas = sorted(sweep.sigma.unique())
    fp_mean, fp_half, ds_mean, ds_half = [], [], [], []
    for s in sigmas:
        sub = sweep[sweep.sigma == s]
        m, h = ci95(sub.top50_confound_fitproject); fp_mean.append(m); fp_half.append(h)
        m, h = ci95(sub.top50_confound_directsub); ds_mean.append(m); ds_half.append(h)
    ax2.errorbar(sigmas, fp_mean, yerr=fp_half, marker="o", color="#2ca02c", capsize=3,
                 label="fit-then-project")
    ax2.errorbar(sigmas, ds_mean, yerr=ds_half, marker="s", color="#1f77b4", capsize=3,
                 label="direct subtract")
    ax2.axvline(1.0, color="gray", ls=":", lw=1)
    ax2.text(1.05, ax2.get_ylim()[0] + 0.02, "method's own\nAUROC region", fontsize=7, color="gray")
    ax2.set_xlabel(r"noise scale $\sigma$ (x std of true confound)")
    ax2.set_ylabel("top-50 true-confound rate")
    ax2.set_title("(b) noise sweep, both mechanisms\n(95% CI, 30 seeds/level)")
    ax2.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG + "fig2_synthetic_correction_gap.pdf")
    plt.close(fig)
    print(f"Wrote {FIG}fig2_synthetic_correction_gap.pdf")


if __name__ == "__main__":
    main()
