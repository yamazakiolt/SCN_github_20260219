# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.colors import Normalize

# =========================
# フォント設定（全体）
# =========================
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.titlesize": 10,
        "axes.labelsize": 20,
    }
)

# =========================
# 設定
# =========================
cmap = plt.get_cmap("cool")
norm = Normalize(vmin=22, vmax=26)

vip_core_list = [125, 100, 75, 50, 25]
vip_shell_list = [25, 50, 75, 100, 125]
light_strangth = "0.2"

# =========================
# Figure & Axes
# =========================
fig, axes = plt.subplots(
    len(vip_core_list),
    len(vip_shell_list),
    figsize=(18, 18),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

fig2, axes2 = plt.subplots(
    len(vip_core_list),
    len(vip_shell_list),
    figsize=(21, 18),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# =========================
# パネルラベル（a, b, c, ...）
# =========================
panel_labels = [f"({chr(97 + i)})" for i in range(25)]

for ax, label in zip(axes.flat, panel_labels):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )
for ax, label in zip(axes2.flat, panel_labels):
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )

# =========================
# 描画ループ
# =========================
for i, vip_core in enumerate(vip_core_list):
    for j, vip_shell in enumerate(vip_shell_list):
        ax = axes[i, j]
        ax2 = axes2[i, j]

        vip_condition = f"{vip_core}_{vip_shell}"
        print(vip_condition)

        x = []
        y = []
        x2 = []
        y2 = []
        c2 = []
        x3, y3 = [], []
        for gaba_sc in [str(round(g / 200, 3)) for g in range(40)]:
            aftereffect_result = pd.read_csv(
                os.path.join(
                    "VIP_GABA_1&1cell_aftereffect",
                    light_strangth,
                    vip_condition,
                    f"evaluate_period_summary_{gaba_sc}.csv",
                )
            )

            entrain_result = pd.read_csv(
                os.path.join(
                    "VIP_GABA_1&1cell_entrain",
                    light_strangth,
                    vip_condition,
                    f"evaluate_period_summary_{gaba_sc}.csv",
                )
            )

            rand_result = pd.read_csv(
                os.path.join(
                    "./VIP_GABA_1_1cell_randint",
                    vip_condition,
                    f"evaluate_period_summary_{gaba_sc}.csv",
                )
            )

            for k in range(0, len(aftereffect_result), 2):
                core = aftereffect_result.loc[k]
                shell = aftereffect_result.loc[k + 1]

                ecore = entrain_result.loc[k]
                eshell = entrain_result.loc[k + 1]

                period_core = core["period"]
                LD_phase = core["L_phase"] + core["D_phase"]

                if (
                    ecore["is_sync"] is not np.False_
                    and eshell["is_sync"] is not np.False_
                ):
                    x2.append(LD_phase)
                    y2.append(float(gaba_sc))
                    c2.append(cmap(norm(period_core)))
                    if (
                        core["is_sync"] is not np.False_
                        and shell["is_sync"] is not np.False_
                    ):
                        x.append(period_core)
                        y.append(float(gaba_sc))

            for rand_seed in set(rand_result["rand_seed"]):
                df = rand_result[rand_result["rand_seed"] == rand_seed]
                V0 = df[df["cell_name"] == "V0"]
                V1 = df[df["cell_name"] == "V1"]

                if len(V0) == 0 or len(V1) == 0:
                    continue

                if V0["is_sync"].iloc[0] and V1["is_sync"].iloc[0]:
                    phase_diff = (V0["phase"].iloc[0] - V1["phase"].iloc[0]) / V0[
                        "period"
                    ].iloc[0]
                    y3.append(V0["gaba_sc"].iloc[0])
                    x3.append(V0["period"].iloc[0])

        ax2.scatter(x2, y2, color=c2, s=10)
        if i == 0 and j == 0:
            ax.scatter(x3, y3, color="k", s=10, alpha=0.1, label="Figure3.11")
            ax.scatter(x, y, c="r", s=10, label="after effect", alpha=0.5)
        else:
            ax.scatter(x3, y3, color="k", s=10, alpha=0.1)
            ax.scatter(x, y, c="r", s=10, alpha=0.5)

        if i == len(vip_core_list) - 1:
            ax.set_xlabel("Period (h)\n\n" + rf"${vip_shell}$", fontsize=30)
            ax2.set_xlabel("T-cycle (h)\n\n" + rf"${vip_shell}$", fontsize=30)

        # 最左列に縦軸の値を表示
        if j == 0:
            ax.set_ylabel(rf"${vip_core}$" + "\n" + r"$g_CS$", fontsize=30)
            ax2.set_ylabel(rf"${vip_core}$" + "\n" + r"$g_CS$", fontsize=30)
        ax.set_xlim((20, 28))
        ax.set_ylim((0, 0.21))
        ax.set_xticks([20, 22, 24, 26, 28])
        ax2.set_xlim((20, 28))
        ax2.set_ylim((0, 0.21))
        ax2.set_xticks([20, 22, 24, 26, 28])

fig.legend(loc="lower left", fontsize=30)
fig.supxlabel(r"$w_S$", fontsize=50)
fig.supylabel(r"$w_C$", fontsize=50)
fig2.supxlabel(r"$w_S$", fontsize=50)
fig2.supylabel(r"$w_C$", fontsize=50)
# =========================
# 共通カラーバー（フォント大）
# =========================
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig2.colorbar(sm, ax=axes2, location="right", fraction=0.03, pad=0.02)

cbar.set_label("Period (h)", fontsize=30)
cbar.ax.tick_params(labelsize=30)

# =========================
# 保存
# =========================
os.makedirs("results", exist_ok=True)
fig.savefig("./results/Figure3_19.png", dpi=300)
# fig2.savefig("./results/Summary2.png", dpi=300)

plt.show()
# %%
