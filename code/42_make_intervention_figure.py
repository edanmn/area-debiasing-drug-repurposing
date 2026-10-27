"""Figure for the intervention dose-response experiment (results/intervention_doseresponse.csv),
added per reviewer request (Issue 4, assessment_updated.txt): the sweep was previously reported
only in prose. Plots, for each target, the bias-diagnostic metric Table "main-bias" reports
(area rate for all five targets; proconvulsant percentile for epilepsy, which has its own panel
since it is on a different scale) against alpha, naive vs proposed, mean +/- std over 5 seeds.

Writes: paper/fig4_intervention_doseresponse.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/intervention_doseresponse.csv")
TARGETS = ["depression", "antipsychotic", "epilepsy", "betablocker", "lipid"]
COLORS = {"naive": "#d55e00", "proposed": "#0072b2"}

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))

ax = axes[0]
for method in ["naive", "proposed"]:
    for target in TARGETS:
        sub = df[(df.target == target) & (df.method == method)]
        g = sub.groupby("alpha")["area_rate_top50"].agg(["mean", "std"])
        ls = "-" if method == "naive" else "--"
        ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COLORS[method], linestyle=ls,
                     marker="o", markersize=2.5, linewidth=1, capsize=1.5, alpha=0.9)
ax.set_xlabel(r"$\alpha$ (correction strength)")
ax.set_ylabel("top-50 same-area rate")
ax.set_title("(a) area rate, all five targets", fontsize=8)

ax = axes[1]
sub_e = df[df.target == "epilepsy"]
for method in ["naive", "proposed"]:
    sub = sub_e[sub_e.method == method]
    g = sub.groupby("alpha")["proconv_pct"].agg(["mean", "std"])
    ls = "-" if method == "naive" else "--"
    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COLORS[method], linestyle=ls,
                 marker="o", markersize=3, linewidth=1.3, capsize=2, label=method)
ax.set_xlabel(r"$\alpha$ (correction strength)")
ax.set_ylabel("proconvulsant rank percentile")
ax.set_title("(b) epilepsy proconvulsant metric", fontsize=8)
ax.legend(fontsize=7, loc="best")

for ax in axes:
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)

fig.tight_layout()
fig.savefig("paper/fig4_intervention_doseresponse.png", dpi=300)
print("wrote paper/fig4_intervention_doseresponse.png")
