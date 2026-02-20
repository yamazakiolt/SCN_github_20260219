# Figure3.1作成プログラム
# phase_osci.pyにより作成されたphase_osci_resultsを読み込んでFigure3.1を作成する
# %%
from matplotlib.lines import Line2D
import json
import matplotlib.pyplot as plt
import numpy as np

# %%
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 10,
        "axes.labelsize": 11,
    }
)


def judge_entrainment_color(phase_L, phase_C, phase_S):
    flag = False
    for i in range(len(phase_C) - 100):
        phase_std_LC = np.std(
            np.array(phase_L[i : i + 100]) - np.array(phase_C[i : i + 100]), ddof=1
        )
        phase_std_LS = np.std(
            np.array(phase_L[i : i + 100]) - np.array(phase_S[i : i + 100]), ddof=1
        )
        if np.abs(phase_std_LC) < 0.001 and np.abs(phase_std_LS) < 0.001:
            flag = True
            break
    if flag:
        return "red"
    else:
        return "blue"


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
LD_list = [10, 16, 24, 32]
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
# subplot 作成の直後あたりで追加
fig.supxlabel(r"$\kappa_{SC}$", fontsize=30)
fig.supylabel(r"$\kappa_{CS}$", fontsize=30)
fig.legend(
    handles=legend_elements, loc="upper right", ncol=1, frameon=False, fontsize=12
)

for index, LD_phase in enumerate(LD_list):
    ax = axes[index // 2, index % 2]
    print(LD_phase)
    with open(f"./phase_osci_results/C_{LD_phase}.json", "r") as f:
        C_phase = json.load(f)
    with open(f"./phase_osci_results/S_{LD_phase}.json", "r") as f:
        S_phase = json.load(f)
    with open(f"./phase_osci_results/L_{LD_phase}.json", "r") as f:
        L_phase = json.load(f)
    with open(f"./phase_osci_results/params_{LD_phase}.json", "r") as f:
        params = json.load(f)

    for i in range(len(params)):
        ksc = params[i][0]
        kcs = params[i][1]
        ax.scatter(
            ksc, kcs, color=judge_entrainment_color(L_phase[i], C_phase[i], S_phase[i])
        )

# subplot 作成の直後あたりで追加
fig.supxlabel(r"$\kappa_{SC}$", fontsize=30)
fig.supylabel(r"$\kappa_{CS}$", fontsize=30)
fig.legend(
    handles=legend_elements, loc="upper right", ncol=1, frameon=False, fontsize=12
)
fig.savefig("./phase_osci_results/Figure3_1.png")
plt.show()
# %%
