# %%
# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.colors import Normalize

# %%
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

vip_core_list = [125, 100, 75, 50, 25]
vip_shell_list = [25, 50, 75, 100, 125]

fig, axes = plt.subplots(
    5, 5, figsize=(20, 18), sharex=True, sharey=True, constrained_layout=True
)

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

cmap = plt.get_cmap("cool")
norm = Normalize(vmin=-0.2, vmax=0.2)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=axes,
    fraction=0.03,
    pad=0.02,
)
cbar.set_label("phase_diff", fontsize=30)
cbar.ax.tick_params(labelsize=30)
fig.supxlabel(r"$w_S$", fontsize=50)
fig.supylabel(r"$w_C$", fontsize=50)

for i, vip_core in enumerate(vip_core_list):
    for j, vip_shell in enumerate(vip_shell_list):
        x, y, c = [], [], []
        ax = axes[i, j]
        vip_condition = f"{vip_core}_{vip_shell}"
        print(vip_condition)

        analyzed_folder = f"./VIP_GABA_1_1cell_randint/{vip_condition}"

        csv_list = sorted(os.listdir(analyzed_folder))

        for csv_file in csv_list:
            print(csv_file)

            period_result = pd.read_csv(os.path.join(analyzed_folder, csv_file))

            for rand_seed in set(period_result["rand_seed"]):
                df = period_result[period_result["rand_seed"] == rand_seed]
                V0 = df[df["cell_name"] == "V0"]
                V1 = df[df["cell_name"] == "V1"]

                if len(V0) == 0 or len(V1) == 0:
                    continue

                if V0["is_sync"].iloc[0] and V1["is_sync"].iloc[0]:
                    phase_diff = (V0["phase"].iloc[0] - V1["phase"].iloc[0]) / V0[
                        "period"
                    ].iloc[0]
                    y.append(V0["gaba_sc"].iloc[0])
                    x.append(V0["period"].iloc[0])
                    c.append(cmap(norm(phase_diff)))
        ax.scatter(x, y, color=c, s=10)
        # ラベル系
        if i == 4:
            ax.set_xlabel(f"{vip_shell}", fontsize=30)
        if j == 0:
            ax.set_ylabel(f"{vip_core}", fontsize=30)
        if i == len(vip_core_list) - 1:
            ax.set_xlabel("Period (h)\n\n" + rf"${vip_shell}$", fontsize=30)
        # 最左列に縦軸の値を表示
        if j == 0:
            ax.set_ylabel(rf"${vip_core}$" + "\n" + r"$g_{SC}$", fontsize=30)
        ax.set_xlim((20, 28))
        ax.set_ylim((0, 0.21))
        ax.set_xticks([20, 22, 24, 26, 28])

# (q) の index（a=0, b=1, ..., q=16）
q_idx = 16
ax_q = axes.flat[q_idx]

for spine in ax_q.spines.values():
    spine.set_linewidth(4)  # 太さ（ここ調整）
    spine.set_color("black")

os.makedirs("./results", exist_ok=True)
fig.savefig("./results/Figure3_11.png")
# %%
