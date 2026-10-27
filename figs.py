"""Four-panel figure: the period 2 efficient frontier of each model, with all
eight optima (4 models x 2 periods) plotted on top."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

with open("results.json") as fh:
    D = json.load(fh)

MODELS = ["Markowitz", "MAD", "VaR", "CVaR"]
COL = {"Markowitz": "#1f4fb0", "MAD": "#c0392b", "VaR": "#1e8c4a", "CVaR": "#7d3c98"}
XLAB = {"Markowitz": "Standard deviation",
        "MAD": "Mean absolute deviation",
        "VaR": "VaR$_{0.95}$ (loss)",
        "CVaR": "CVaR$_{0.95}$ (loss)"}
TITLE = {"Markowitz": "(a) Markowitz model",
         "MAD": "(b) Mean absolute deviation model",
         "VaR": "(c) VaR model", "CVaR": "(d) CVaR model"}

plt.rcParams.update({"font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "font.family": "DejaVu Sans"})

fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
for ax, panel in zip(axes.flat, MODELS):
    fr = D["frontier"][panel]
    risk = np.array(fr["risk"])
    ret = np.array(fr["ret"])
    order = np.argsort(ret)
    ax.plot(risk[order], ret[order], "-", color=COL[panel], lw=1.6, zorder=1)
    for pt in D["points"][panel]:
        c = COL[pt["model"]]
        mk = "s" if pt["period"] == 1 else "o"
        ax.scatter(pt["risk"], pt["ret"], marker=mk, s=46,
                   facecolor=c, edgecolor="black", linewidth=0.6, zorder=3)
    ax.axhline(D["rbar"], color="grey", ls=":", lw=1.0, zorder=0)
    ax.set_xlabel(XLAB[panel])
    ax.set_ylabel("Daily return")
    ax.set_title(TITLE[panel], fontsize=9.5)
    ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, -2))

leg = [Line2D([0], [0], color="grey", lw=1.4, label="Efficient frontier")]
for m in MODELS:
    leg.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[m],
                      markeredgecolor="black", markersize=8, label=m))
leg.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="grey",
                  markeredgecolor="black", markersize=8, label="Period 1 optimum"))
leg.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="grey",
                  markeredgecolor="black", markersize=8, label="Period 2 optimum"))
fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=8.2,
           frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("frontiers.png", dpi=170, bbox_inches="tight")
print("Done: frontiers.png")
