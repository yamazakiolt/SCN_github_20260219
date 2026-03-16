# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.colors import Normalize
from collections import Counter

light_strangth = "0.2"


def get_continuous_width(x_vals):
    """連続しているとみなせる最大の幅を返す補助関数"""
    if len(x_vals) < 2:
        return 0, 0, 0, 0

    x_sorted = np.sort(x_vals)
    # 隣り合う点との差が0.15h以上あれば「不連続」とみなす
    diffs = np.diff(x_sorted)
    threshold = 0.25
    # 隙間があるインデックスを取得
    gap_indices = np.where(diffs > threshold)[0]

    # 各区間の開始と終了インデックスをリスト化
    starts = np.insert(gap_indices + 1, 0, 0)
    ends = np.append(gap_indices, len(x_sorted) - 1)

    # 各区間の幅を計算し、最大のものを採用
    widths = x_sorted[ends] - x_sorted[starts]
    max_idx = np.argmax(widths)
    mean_xvals = (x_sorted[starts[max_idx]] + x_sorted[ends[max_idx]]) / 2

    return (
        x_sorted[starts[max_idx]],
        x_sorted[ends[max_idx]],
        widths[max_idx],
        mean_xvals,
    )


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

# =========================
# Figure & Axes
# =========================

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
        ax2 = axes2[i, j]

        vip_condition = f"{vip_core}_{vip_shell}"
        print(vip_condition)
        x2, y2 = [], []

        for gaba_sc in [str(round(g / 200, 3)) for g in range(40)]:
            entrain_result = pd.read_csv(
                os.path.join(
                    "VIP_GABA_5&5cell_entrain",
                    light_strangth,
                    vip_condition,
                    f"evaluate_period_summary_{gaba_sc}.csv",
                )
            )

            for L_phase in set(entrain_result["L_phase"]):
                df_entrain = entrain_result[entrain_result["L_phase"] == L_phase]
                LD_phase = 2 * L_phase

                if np.prod(df_entrain["is_sync"]) == 1:
                    x2.append(LD_phase)
                    y2.append(float(gaba_sc))

        ax2.scatter(x2, y2, color="k", s=10)
        most_common_value = Counter(y2).most_common(1)[0][0]
        # --- 最頻値(most_common_value)の行 ---
        x2 = np.array(x2)
        y2 = np.array(y2)
        mask = y2 == most_common_value
        x2_min, x2_max, width_most_common, mean_most_common = get_continuous_width(
            x2[mask]
        )

        ax2.hlines(
            y=most_common_value, xmin=x2_min, xmax=x2_max, colors="r", linestyle="--"
        )

        # --- y=0 の行 ---
        mask_0 = y2 == 0
        x2_0_min, x2_0_max, width_0, mean_0 = get_continuous_width(x2[mask_0])

        ax2.hlines(y=0, xmin=x2_0_min, xmax=x2_0_max, colors="b", linestyle="--")

        # --- テキスト表示 ---
        ax2.text(
            26, 0.0, rf"${mean_0:.1f} \pm {width_0 / 2:.1f} h$", fontsize=20, c="b"
        )
        ax2.text(
            26.5,
            most_common_value,
            rf"${mean_most_common:.1f} \pm {width_most_common / 2:.1f} h$",
            fontsize=20,
            c="r",
        )
        ax2.hlines(y=0, xmin=x2_0_min, xmax=x2_0_max, colors="b", linestyle="--")
        if i == len(vip_core_list) - 1:
            ax2.set_xlabel("T-cycle (h)\n\n" + rf"${vip_shell}$", fontsize=30)

        # 最左列に縦軸の値を表示
        if j == 0:
            ax2.set_ylabel(rf"${vip_core}$" + "\n" + r"$g_{SC}$", fontsize=30)
        ax2.set_xlim((20, 28))
        ax2.set_ylim((-0.005, 0.21))
        ax2.set_xticks([20, 22, 24, 26, 28])

fig2.supxlabel(r"$w_S$", fontsize=50)
fig2.supylabel(r"$w_C$", fontsize=50)


# =========================
# 保存
# =========================
os.makedirs("./results", exist_ok=True)
fig2.savefig("./results/Figure3_18.png", dpi=300)

plt.show()
# %%
