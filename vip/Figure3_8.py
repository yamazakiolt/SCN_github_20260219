# %%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# %%
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 1,  # 線の太さ
    }
)
# %%
VIP_list = sorted(
    [f for f in os.listdir("./results/multicell/20") if not f.startswith(".")],
    key=lambda x: int(x),
)
# %%
fig = plt.figure(figsize=(14, 14))

gs = GridSpec(
    nrows=4,
    ncols=1,
    width_ratios=[1],  # ← 左：右 = 1 : 3
    height_ratios=[1, 1, 1, 1],
    hspace=0.3,
    wspace=0.3,
)

ax_list = []

for i in range(4):
    ax = fig.add_subplot(gs[i, 0])
    ax_list.append(ax)

panel_labels_azzi = [f"({chr(97 + i)})" for i in range(4)]

t_range = (1600, 1700)
index = 0
for vip in VIP_list:
    if (int(vip) - 10) % 30 == 0:
        ax = ax_list[index]
        print(vip)
        with open(os.path.join("results", "multicell", "./20", vip, "V.csv")) as f:
            df = pd.read_csv(f)
        print(np.std(df.loc[:, "V0":"V19"].mean(axis=1)))
        t_eval = np.array(df["t"])
        t_mask = (t_eval >= t_range[0]) & (t_eval <= t_range[1])
        t_plt = t_eval[t_mask]
        for cell_name in [c for c in df.keys() if c.startswith("V")]:
            ax.plot(t_plt, np.array(df[cell_name])[t_mask], c="k")
        ax.plot(
            t_plt,
            np.array(df.loc[:, "V0":"V19"].mean(axis=1))[t_mask],
            c="r",
            linewidth=5,
            linestyle="--",
        )
        print(np.std(np.array(df.loc[:, "V0":"V19"].mean(axis=1))[t_mask]))
        ax.set_xlim(t_range)
        ax.set_ylim((0, 8))
        ax.set_yticks([0, 4, 8])
        ax.set_ylabel(r"$V$", fontsize=20)
        if index == 3:
            ax.set_xlabel("Time(h)", fontsize=30)
        ax.text(
            -0.1,
            1.25,
            panel_labels_azzi[index],
            transform=ax.transAxes,
            fontsize=30,
            fontweight="bold",
            va="top",
            ha="left",
        )
        ax.set_title(r"$w_C=w_S=$" + str(vip), fontsize=20)
        index += 1
fig.savefig("./results/Figure3_8.png")
plt.show()
# %%
