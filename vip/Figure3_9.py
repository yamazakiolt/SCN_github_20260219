# %%
import numpy as np
import matplotlib.pyplot as plt
from utils_fig39 import cellmodel  # 自作関数の読み込み
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec
import os

# %%
os.makedirs("results", exist_ok=True)
np.random.seed(222)
# %%
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 3,  # 線の太さ
    }
)


fig = plt.figure(figsize=(14, 9))

gs = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[1, 8],  # ← 左：右 = 1 : 3
    height_ratios=[1, 1, 1],
    hspace=0.3,
    wspace=0.3,
)

axes = []

for i in range(3):
    ax_left = fig.add_subplot(gs[i, 0])
    ax_right = fig.add_subplot(gs[i, 1])
    axes.append((ax_left, ax_right))

panel_labels_azzi = [f"({chr(97 + i)}-1)" for i in range(3)]

x = np.array([i / 1000 for i in range(24 * 1000)])
core = [9, 9.5, 13]
shell = [11.5, 9.5, 11]
LD_list = [22, 24, 26]
for index, LD in enumerate(LD_list):
    ax = axes[index][0]
    y_core = np.sin(((2 * np.pi * (x + 6 - core[index])) / 24))
    y_shell = np.sin(((2 * np.pi * (x + 6 - shell[index])) / 24))
    ax.plot(x, y_core, c="#FF0000")
    ax.plot(x, y_shell, c="#0070C0", linestyle="--")
    ax.set_xticks([0, 12, 24])
    ax.set_yticks([-1, 1])
    if index == 2:
        ax.set_xlabel("Time(h)", fontsize=30)
    ax.set_ylabel(r"$Per2 luc$", fontsize=20)
    ax.text(
        -0.9,
        1.25,
        panel_labels_azzi[index],
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_light_background(
    ax, t_eval, light_wave, threshold=0.1, color="yellow", alpha=0.2
):
    """light_waveがthreshold以上の区間に背景を描く"""
    in_light = False
    for i in range(1, len(t_eval)):
        if light_wave[i - 1] < threshold and light_wave[i] >= threshold:
            start = t_eval[i]
            in_light = True
        elif light_wave[i - 1] >= threshold and light_wave[i] < threshold and in_light:
            end = t_eval[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_light = False
    # 光が終わらずに最後まで続いた場合
    if in_light:
        ax.axvspan(start, t_eval[-1], color=color, alpha=alpha)


cell_params = {
    "vs": 2.8,
    "n_p": 5,
    "P0": 4.6,
    "vc": 9.1 * 0.06,
    "n_ca": 3,
    "Kc": 0.1,
    "myu": 0.65,
    "ks": 7.9,
    "k1": 0.1,
    "vd": 9.3,
    "Kd": 56.1,
    "betaCa": 49.3,
    "ap": 1,
    "kdp": 910,
    "Pv": 22,
    "m": 10,
    "tauCa": 1,
    "taur": 0.0002,
    "alpha": 15,
    "us": 1,
    "lambdaVIP": 1,
    "V0": 5,
    "eta": 1,
}

t_max = 4000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.01)

init1 = np.random.rand(11) * np.array([1.25, 40, 30, 700, 15, 1, 1.25, 40, 30, 700, 15])
LD_params_11 = {
    "LD_time": [0, t_max],
    "L_phase": 11,
    "D_phase": 11,
    "L_max": 0.4,
    "L_min": 0,
    "k_light": 11,
    "k_dark": 11,
}  # LD = 11
LD_params_12 = {
    "LD_time": [0, t_max],
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0.4,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
}  # LD = 12
LD_params_13 = {
    "LD_time": [0, t_max],
    "L_phase": 13,
    "D_phase": 13,
    "L_max": 0.4,
    "L_min": 0,
    "k_light": 13,
    "k_dark": 13,
}  # LD = 13
con_param_VIP = {
    "b_core": 0,  # VIP
    "b_shell": 60.0,
    "gsa": 0,
}
model_VIP_11 = cellmodel.CellModel(
    LD_params_11, cell_params, connect_param=con_param_VIP
)
model_VIP_12 = cellmodel.CellModel(
    LD_params_12, cell_params, connect_param=con_param_VIP
)
model_VIP_13 = cellmodel.CellModel(
    LD_params_13, cell_params, connect_param=con_param_VIP
)

sol_11 = solve_ivp(
    fun=lambda t, y: model_VIP_11.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init1,
    t_eval=t_eval,
    method="LSODA",
)  # light
sol_12 = solve_ivp(
    fun=lambda t, y: model_VIP_12.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init1,
    t_eval=t_eval,
    method="LSODA",
)  # light
sol_13 = solve_ivp(
    fun=lambda t, y: model_VIP_13.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init1,
    t_eval=t_eval,
    method="LSODA",
)  # light

n_conv = 100
window_conv = np.ones(n_conv) / n_conv

P_core_11 = np.convolve(sol_11.y[1, :] + sol_11.y[2, :], window_conv, mode="same")
P_shell_11 = np.convolve(sol_11.y[7, :] + sol_11.y[8, :], window_conv, mode="same")
P_core_12 = np.convolve(sol_12.y[1, :] + sol_12.y[2, :], window_conv, mode="same")
P_shell_12 = np.convolve(sol_12.y[7, :] + sol_12.y[8, :], window_conv, mode="same")
P_core_13 = np.convolve(sol_13.y[1, :] + sol_13.y[2, :], window_conv, mode="same")
P_shell_13 = np.convolve(sol_13.y[7, :] + sol_13.y[8, :], window_conv, mode="same")

light_wave_11 = np.array([model_VIP_11.Light_strangth(t) for t in t_eval])
light_wave_12 = np.array([model_VIP_12.Light_strangth(t) for t in t_eval])
light_wave_13 = np.array([model_VIP_13.Light_strangth(t) for t in t_eval])
# 時間軸の表示範囲
t_range = (3430, 3500)
t_mask = (t_eval >= t_range[0]) & (t_eval <= t_range[1])

P_core = [P_core_11, P_core_12, P_core_13]
P_shell = [P_shell_11, P_shell_12, P_shell_13]
light_wave = [light_wave_11, light_wave_12, light_wave_13]
ax_P = [axes[0][1], axes[1][1], axes[2][1]]
panel_labels_model = [f"({chr(97 + i)}-2)" for i in range(3)]

for index, ax in enumerate(ax_P):
    draw_light_background(
        ax,
        t_eval[t_mask],
        light_wave[index][t_mask],
        threshold=LD_params_11["L_max"] / 2,
    )
    if index == 0:
        ax.plot(t_eval[t_mask], P_core[index][t_mask], c="#FF0000", label="Core")
        ax.plot(
            t_eval[t_mask],
            P_shell[index][t_mask],
            c="#0070C0",
            linestyle="--",
            label="Shell",
        )
        fig.legend(loc="upper right", fontsize=20, handlelength=4)
    else:
        ax.plot(t_eval[t_mask], P_core[index][t_mask], c="#FF0000")
        ax.plot(
            t_eval[t_mask],
            P_shell[index][t_mask],
            c="#0070C0",
            linestyle="--",
        )
    ax.set_xlim(t_range)
    ax.set_ylim((40, 80))
    ax.set_ylabel(r"$Pc+Pn$", fontsize=20)
    if index == 2:
        ax.set_xlabel("Time(h)", fontsize=30)
    ax.text(
        -0.15,
        1.25,
        panel_labels_model[index],
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )
    ax.text(
        0.7,
        0.85,
        f"LD={LD_list[index]} h",
        transform=ax.transAxes,
        fontsize=30,
        # fontweight="bold",
        va="top",
        ha="left",
    )
fig.savefig("./results/Figure3_9.png")
plt.show()

# %%
