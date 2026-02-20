# Figure3.6の作成プログラム
# %%
from utils import neuron_model
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

np.random.seed(222)
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 5,
    }
)
# %%
os.makedirs("results", exist_ok=True)
# %%


def draw_daytime_background(ax, t_eval, L_phase, D_phase, color="yellow", alpha=0.2):
    """t_eval % 24 が 6〜18 の区間に背景を描く（日中を示す）"""
    in_daytime = False
    for i in range(1, len(t_eval)):
        hour_prev = t_eval[i - 1] % (L_phase + D_phase)
        hour_curr = t_eval[i] % (L_phase + D_phase)

        if hour_prev < L_phase / 2 and hour_curr >= L_phase / 2:
            start = t_eval[i]
            in_daytime = True
        elif (
            hour_prev < L_phase + D_phase / 2
            and hour_curr >= L_phase + D_phase / 2
            and in_daytime
        ):
            end = t_eval[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_daytime = False

    # 日中が最後まで続いた場合
    if in_daytime:
        ax.axvspan(start, t_eval[-1], color=color, alpha=alpha)


# %%
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

t_max = 2000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.001)
np.random.seed(2222)

init1 = np.random.rand(6) * np.array([1.25, 40, 30, 700, 15, 1])
init2 = np.random.rand(6) * np.array([1.25, 40, 30, 700, 15, 1])
init3 = np.random.rand(6) * np.array([1.25, 40, 30, 700, 15, 1])
init_all = [init1, init2, init3]

LD_params_base = {
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
    "LD_time": t_span,
}  # 光なし
LD_params_1 = {
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0.3,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
    "LD_time": t_span,
}  # 光あり


cell = [neuron_model.SingleCell(cell_params, cell_id=0, cell_type="core")]
network = neuron_model.CellNetwork(cell, vip_weight=[0], gaba_matrix=np.zeros((1, 1)))
# %%
light_func_args_base = (
    LD_params_base["L_phase"],
    LD_params_base["D_phase"],
    LD_params_base["L_max"],
    LD_params_base["L_min"],
    LD_params_base["k_light"],
    LD_params_base["k_dark"],
    LD_params_base["LD_time"][0],
    LD_params_base["LD_time"][1],
)
light_func_args_1 = (
    LD_params_1["L_phase"],
    LD_params_1["D_phase"],
    LD_params_1["L_max"],
    LD_params_1["L_min"],
    LD_params_1["k_light"],
    LD_params_1["k_dark"],
    LD_params_1["LD_time"][0],
    LD_params_1["LD_time"][1],
)

t_range = (1002, 1062)
t_mask = (t_eval >= t_range[0]) & (t_eval <= t_range[1])


V_base_list = []
V_light_list = []
for init_i in init_all:
    sol_base_i = solve_ivp(
        fun=lambda t, y: network.ode_all_neurons(
            t, y, light_func_args=light_func_args_base
        ),
        t_span=[0, t_max],
        t_eval=t_eval,
        y0=init_i,
        method="LSODA",
    )
    sol_1_i = solve_ivp(
        fun=lambda t, y: network.ode_all_neurons(
            t, y, light_func_args=light_func_args_1
        ),
        t_span=[0, t_max],
        t_eval=t_eval,
        y0=init_i,
        method="LSODA",
    )
    n_conv = 100
    window_conv = np.ones(n_conv) / n_conv
    V_base_i = np.convolve(sol_base_i.y[4, :], window_conv, mode="same")
    V_1_i = np.convolve(sol_1_i.y[4, :], window_conv, mode="same")
    V_base_list.append(V_base_i)
    V_light_list.append(V_1_i)
    light_func = neuron_model.LightModel(LD_params_1)
    light_strangth = [light_func.light_strength(t) for t in t_eval]

peaks_light = [find_peaks(V, prominence=0.5)[0] for V in V_light_list]
peaks_base = [find_peaks(V, prominence=0.5)[0] for V in V_base_list]

plt.figure(figsize=(8, 8))
ax1 = plt.subplot(2, 1, 2)
draw_daytime_background(
    ax1, t_eval[t_mask], L_phase=LD_params_1["L_phase"], D_phase=LD_params_1["D_phase"]
)
for i, V in enumerate(V_light_list):
    plt.plot(t_eval[t_mask], V[t_mask] + i * 10, c="k")
    plt.vlines(t_eval[peaks_light[i]], 10 * i + 4, 10 * i + 9, color="k")
plt.xlim(t_range)
plt.ylim(-2, 30)
plt.xticks(np.arange(t_range[0], t_range[1] + 1, 12))
plt.yticks([0, 10, 20, 30])
ax1.set_ylabel(r"$V$", fontsize=20)
ax1.set_xlabel("Time(h)", fontsize=20)

# === 下段：Lightなし ===
ax2 = plt.subplot(2, 1, 1)

for i, V in enumerate(V_base_list):
    plt.plot(t_eval[t_mask], V[t_mask] + i * 10, c="k")
    plt.vlines(t_eval[peaks_base[i]], 10 * i + 4, 10 * i + 9, color="k")
plt.xlim(t_range)
plt.ylim(-2, 30)
plt.xticks(np.arange(t_range[0], t_range[1] + 1, 12))
plt.yticks([0, 10, 20, 30])
ax2.set_ylabel(r"$V$", fontsize=20)

ax2.text(
    -0.2,
    1.25,
    "(b)",
    transform=ax2.transAxes,
    fontsize=30,
    fontweight="bold",
    va="top",
    ha="left",
)
ax1.text(
    -0.2,
    1.25,
    "(c)",
    transform=ax1.transAxes,
    fontsize=30,
    fontweight="bold",
    va="top",
    ha="left",
)
plt.tight_layout()
plt.savefig("./results/Figure3_6(b)(c).png", dpi=300)
plt.show()
# %%
LD_params_1 = {
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 1,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
    "LD_time": t_span,
}  # 光あり
t_range = (1008, 1032)
light_time_mask = (t_eval >= t_range[0]) & (t_eval <= t_range[1])
light_time = t_eval[light_time_mask]
light_func = neuron_model.LightModel(LD_params_1)
light_strangth = np.array([light_func.light_strength(t) for t in t_eval])[
    light_time_mask
]

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.plot(light_time, light_strangth, c="k")
ax3.set_xticks(np.arange(t_range[0], t_range[1] + 1, 6))
ax3.set_yticks([0, 0.5, 1])
ax3.axvspan(1014, 1026, color="yellow", alpha=0.2)
ax3.set_xlabel("Time(h)", fontsize=30)
ax3.set_ylabel("Light Strength", fontsize=30)
ax3.text(
    -0.2,
    1.25,
    "(a)",
    transform=ax3.transAxes,
    fontsize=30,
    fontweight="bold",
    va="top",
    ha="left",
)
plt.tight_layout()
fig3.savefig("./results/Figure3_6(a).png", dpi=300)
plt.show()
# %%
