# VIPによる位相同期
# %%
from utils import neuron_model  # 自作関数の読み込み
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

np.random.seed(222)
# %%
os.makedirs("./results", exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 3,  # 線の太さ
    }
)


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

# %%
init1 = np.random.rand(12) * np.array(
    [1.25, 40, 30, 700, 15, 1, 1.25, 40, 30, 700, 15, 1]
)

t_span = [0, 2000]
t_eval = np.arange(0, 2000, 0.001)
cell_types = ["core", "shell"]
cells_num_each_type = [1, 1]
cell_type_each_cell = []
for index, cell_type in enumerate(cell_types):
    for _ in range(cells_num_each_type[index]):
        cell_type_each_cell.append(cell_type)
cells_num = sum(cells_num_each_type)

vip_core = 50
vip_shell = 50
vip_weights_0 = [0] * cells_num
vip_weights = []
for cell_index in range(cells_num):
    if cell_type_each_cell[cell_index] == "core":
        vip_weights.append(vip_core)
    if cell_type_each_cell[cell_index] == "shell":
        vip_weights.append(vip_shell)
gaba_matrix = np.zeros((cells_num, cells_num))

# 光なし
LD_params_nolight = {
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
    "LD_time": [0, 0],
}
light_func_args_nolight = (
    LD_params_nolight["L_phase"],
    LD_params_nolight["D_phase"],
    LD_params_nolight["L_max"],
    LD_params_nolight["L_min"],
    LD_params_nolight["k_light"],
    LD_params_nolight["k_dark"],
    LD_params_nolight["LD_time"][0],
    LD_params_nolight["LD_time"][1],
)
cells = [
    neuron_model.SingleCell(cell_params, i, cell_type=cell_type_each_cell[i])
    for i in range(cells_num)
]

network_novip = neuron_model.CellNetwork(cells, vip_weights_0, gaba_matrix)
network_vip = neuron_model.CellNetwork(cells, vip_weights, gaba_matrix)

sol_noVIP = solve_ivp(
    fun=lambda t, y: network_novip.ode_all_neurons(
        t, y, light_func_args=light_func_args_nolight
    ),
    t_span=t_span,
    t_eval=t_eval,
    y0=init1,
    method="LSODA",
)

init_VIP = sol_noVIP.y[:, 300 * 1000]

# t=300から開始
sol_VIP = solve_ivp(
    fun=lambda t, y: network_vip.ode_all_neurons(
        t, y, light_func_args=light_func_args_nolight
    ),
    t_span=t_span,
    y0=init_VIP,
    t_eval=t_eval,
    method="LSODA",
)


n_conv = 100
window_conv = np.ones(n_conv) / n_conv

V_core_VIP = np.convolve(sol_VIP.y[4, :], window_conv, mode="same")
V_shell_VIP = np.convolve(sol_VIP.y[10, :], window_conv, mode="same")
V_core_noVIP = np.convolve(sol_noVIP.y[4, :], window_conv, mode="same")
V_shell_noVIP = np.convolve(sol_noVIP.y[10, :], window_conv, mode="same")

# 時間軸の表示範囲
t_range_VIP = (0, 250)
t_range_noVIP = (250, 550)
t_range_pre = (250, 300)
t_range_plt = (0, 300)
t_mask_VIP = (t_eval >= t_range_VIP[0]) & (t_eval <= t_range_VIP[1])
t_mask_noVIP = (t_eval >= t_range_noVIP[0]) & (t_eval <= t_range_noVIP[1])
t_mask_pre = (t_eval >= t_range_pre[0]) & (t_eval <= t_range_pre[1])
t_mask_plt = (t_eval >= t_range_plt[0]) & (t_eval <= t_range_plt[1])
t_eval_plt = t_eval[t_mask_plt]
V_core_VIP_plt = np.hstack((V_core_noVIP[t_mask_pre][:-1], V_core_VIP[t_mask_VIP]))
V_core_noVIP_plt = V_core_noVIP[t_mask_noVIP]
V_shell_VIP_plt = np.hstack((V_shell_noVIP[t_mask_pre][:-1], V_shell_VIP[t_mask_VIP]))
V_shell_noVIP_plt = V_shell_noVIP[t_mask_noVIP]

fig = plt.figure(figsize=(14, 9))

peak_index_core_VIP, _ = find_peaks(V_core_VIP)
peak_interval_core_VIP = np.diff(t_eval[peak_index_core_VIP])
peak_index_core_noVIP, _ = find_peaks(V_core_noVIP)
peak_interval_core_noVIP = np.diff(t_eval[peak_index_core_noVIP])

period_VIP = round(np.mean(peak_interval_core_VIP[-13:-3]), 2)
period_noVIP = round(np.mean(peak_interval_core_noVIP[-13:-3]), 2)

ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t_eval_plt, V_core_noVIP_plt, c="#FF0000", label="Core")
ax1.plot(t_eval_plt, V_shell_noVIP_plt, c="#0070C0", linestyle="--", label="Shell")
ax1.set_ylabel(r"$V$", fontsize=30)
ax1.axvline(50, c="k")
ax1.set_title("VIP -", fontsize=30)
ax1.text(
    0.05, 1.02, "VIP -", transform=ax1.transAxes, ha="left", va="bottom", fontsize=30
)
ax1.set_ylim((0, 8))
ax1.text(
    0.02,
    0.8,
    rf"${period_noVIP}~h$",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax1.text(
    0.8,
    0.8,
    rf"${period_noVIP}~h$",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t_eval_plt, V_core_VIP_plt, c="#FF0000")
ax2.plot(t_eval_plt, V_shell_VIP_plt, c="#0070C0", linestyle="--")
ax2.set_ylabel(r"$V$", fontsize=30)
ax2.set_xlabel("Time(h)", fontsize=30)
ax2.set_title("VIP +", fontsize=30)
ax2.text(
    0.05, 1.02, "VIP -", transform=ax2.transAxes, ha="left", va="bottom", fontsize=30
)
ax2.axvline(50, c="k")
ax2.set_ylim((0, 8))
ax2.text(
    0.02,
    0.8,
    rf"${period_noVIP}~h$",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax2.text(
    0.8,
    0.8,
    rf"${period_VIP}~h$",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

panel_labels = [f"({chr(97 + i)})" for i in range(2)]

for ax, label in zip([ax1, ax2], panel_labels):
    ax.text(
        -0.2,
        1.25,
        label,
        transform=ax.transAxes,
        fontsize=50,
        fontweight="bold",
        va="top",
        ha="left",
    )
fig.legend(loc="upper right", fontsize=20, handlelength=4)
plt.tight_layout()
fig.savefig("results/Figure3_7.png")
plt.show()

# %%
