# %%
import matplotlib.pyplot as plt
import json
from matplotlib.lines import Line2D

# %%
legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Entrained",
        markerfacecolor="red",
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Not entrained",
        markerfacecolor="blue",
        markersize=8,
    ),
]
# %%
with open("./phase_osci_LSA_results/params.json", "r") as f:
    params = json.load(f)
with open("./phase_osci_LSA_results/entrainment.json", "r") as f:
    judge_entrainment = json.load(f)


# %%
fig, axes = plt.subplots(
    2, 2, figsize=(8, 6), sharex=True, sharey=True, constrained_layout=True
)
panel_labels = [f"({chr(97 + i)})" for i in range(4)]

for ax, label in zip(axes.flat, panel_labels):
    ax.text(
        -0.2,
        1.25,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
    )
fig.supxlabel(r"$\kappa_{SC}$", fontsize=30)
fig.supylabel(r"$\kappa_{CS}$", fontsize=30)
fig.legend(
    handles=legend_elements, loc="upper right", ncol=1, frameon=False, fontsize=12
)
for index, T_light in enumerate([10, 16, 24, 32]):
    ax = axes[index // 2, index % 2]
    for param_index in range(len(params)):
        if params[param_index][0] == T_light:
            if judge_entrainment[param_index]:
                plot_color = "r"
            else:
                plot_color = "b"
            ax.scatter(params[param_index][1], params[param_index][2], color=plot_color)
fig.legend(
    handles=legend_elements, loc="upper right", ncol=1, frameon=False, fontsize=12
)
fig.savefig("./phase_osci_LSA_resultsFigure3_2.png")
##############################################################################################
# %%
