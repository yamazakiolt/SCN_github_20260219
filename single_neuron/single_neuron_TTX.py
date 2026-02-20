# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from utils_TTX import neuron_model
import os
import time

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

TTX_start = 500
TTX_end = 600

# 微分方程式の計算
start = time.time()
cells = [
    neuron_model.SingleCell(cell_params, i, cell_type=cell_types[i])
    for i in range(cells_num)
]
net = neuron_model.CellNetwork(cells, vip_weights, gaba_matrix)
sol_TTX = solve_ivp(
    fun=lambda t, y: net.ode_all_neurons(t, y, light_func_args, TTX_start, TTX_end),
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

times = sol_TTX.t
M_TTX = np.convolve(sol_TTX.y[0, :], b, mode="same")
Pc_TTX = np.convolve(sol_TTX.y[1, :], b, mode="same")
Pn_TTX = np.convolve(sol_TTX.y[2, :], b, mode="same")
Ca_TTX = np.convolve(sol_TTX.y[3, :], b, mode="same")
V_TTX = np.convolve(sol_TTX.y[4, :], b, mode="same")
pCREBp_TTX = []
for i0 in range(len(Ca_TTX)):
    pCREBp_TTX.append(
        cell_params["ap"]
        * Ca_TTX[i0]
        / (cell_params["ap"] * Ca_TTX[i0] + cell_params["kdp"])
    )
pCREBp_TTX = np.array(pCREBp_TTX)


fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(sol_TTX.t, M_TTX, color="r")
ax2.axvspan(500, 600, color="#FFD4FF")
ax2.set_xticks([400, 500, 600, 700])
ax2.set_yticks([0, 0.5, 1, 1.5])
ax2.set_xlim((400, 700))
ax2.set_ylim((-0.3, 1.5))
ax2.set_xlabel("Time (h)", fontsize=30)
ax2.set_ylabel("$M$", fontsize=30, labelpad=0)
plt.tight_layout()
fig2.savefig("./results/Figure3_4.png", dpi=300)
plt.show()
# %%
