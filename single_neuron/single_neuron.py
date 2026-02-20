# Figure3.3, Figure3.4作成用プログラム
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from utils import neuron_model
import os

# %%
os.makedirs("./results", exist_ok=True)
# %%
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "lines.linewidth": 5,  # 線の太さ
    }
)

# %%

cell_params = {
    "vs": 2.8,
    "n_p": 5,
    "P0": 4.6,
    "vc": 0.546,
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

y0 = [0, 0, 0, 0, 0, 0]
t_span = [0, 2000]
t_eval = np.arange(0, 2000, 0.001)
cell_types = ["core"]
cells_num_each_type = [1]
cell_type_each_cell = []
for index, cell_type in enumerate(cell_types):
    for _ in range(cells_num_each_type[index]):
        cell_type_each_cell.append(cell_type)
cells_num = sum(cells_num_each_type)
vip_core = 0
vip_weights = []
for cell_index in range(cells_num):
    if cell_type_each_cell[cell_index] == "core":
        vip_weights.append(vip_core)
gaba_matrix = np.zeros((cells_num, cells_num))

L_phase = 12
D_phase = 12
L_max = 0
L_min = 0
k_light = 12
k_dark = 12
LD_start = 0
LD_end = 0
light_func_args = (L_phase, D_phase, L_max, L_min, k_light, k_dark, LD_start, LD_end)


# 微分方程式の計算
start = time.time()
cells = [
    neuron_model.SingleCell(cell_params, i, cell_type=cell_types[i])
    for i in range(cells_num)
]
net = neuron_model.CellNetwork(cells, vip_weights, gaba_matrix)
sol = solve_ivp(
    fun=lambda t, y: net.ode_all_neurons(t, y, light_func_args),
    t_span=[0, 2000],
    t_eval=np.arange(0, 2000, 0.01),
    y0=y0,
    method="LSODA",
)
end = time.time()
print("ode solve time : ", end - start)

# 平滑化
n_conv = 100
b = np.ones(n_conv) / n_conv

times = sol.t
M = np.convolve(sol.y[0, :], b, mode="same")
Pc = np.convolve(sol.y[1, :], b, mode="same")
Pn = np.convolve(sol.y[2, :], b, mode="same")
Ca = np.convolve(sol.y[3, :], b, mode="same")
V = np.convolve(sol.y[4, :], b, mode="same")
pCREBp = []
for i0 in range(len(Ca)):
    pCREBp.append(
        cell_params["ap"] * Ca[i0] / (cell_params["ap"] * Ca[i0] + cell_params["kdp"])
    )
pCREBp = np.array(pCREBp)

fig1 = plt.figure(figsize=(16, 6))
ax1_0 = fig1.add_subplot(121)
ax1_0.plot(sol.t, M, color="#FF0000", lw=5)
ax1_0.plot(sol.t, pCREBp, color="#196B24", lw=5)
ax1_0.set_yticks([0, 0.5, 1, 1.5, 2])
ax1_0.set_xticks([520, 532, 544, 556, 568])
ax1_0.set_ylabel(r"$M~CREBp$", fontsize=30, labelpad=10)

ax1_1 = ax1_0.twinx()
ax1_1.plot(sol.t, Pc, color="#00B0F0", lw=5)
ax1_1.plot(sol.t, Pn, color="#7030A0", lw=5)
ax1_1.set_yticks([-10, 0, 10, 20, 30, 40, 50])
ax1_0.set_xlim((520, 568))
ax1_0.set_ylim((-0.2, 2.0))
ax1_1.set_ylim((-10, 50))
ax1_0.set_xlabel("Time (h)", fontsize=30)
ax1_1.set_ylabel(r"$Pc~Pn$", fontsize=30, rotation=270, labelpad=10)

ax2_0 = fig1.add_subplot(122)
ax2_0.plot(sol.t, V, color="k")
ax2_0.set_yticks([0, 5, 10])
ax2_0.set_xticks([520, 532, 544, 556, 568])
ax2_1 = ax2_0.twinx()
ax2_1.plot(sol.t, Ca, color="#E97132")
ax2_1.set_yticks([0, 250, 500])
ax2_1.set_xlim((520, 568))
ax2_0.set_ylim((0, 10))
ax2_1.set_ylim((0, 500))
ax2_0.set_xlabel("Time (h)", fontsize=30)
ax2_0.set_ylabel(r"$V$", fontsize=30, labelpad=-10)
ax2_1.set_ylabel(r"$[Ca^{2+}]$", fontsize=30, rotation=270, labelpad=30)

panel_labels = [f"({chr(97 + i)})" for i in range(2)]

for ax, label in zip([ax1_0, ax2_0], panel_labels):
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
plt.tight_layout()
fig1.savefig("./results/Figure3_3.png", dpi=300)
plt.show()


# %%
