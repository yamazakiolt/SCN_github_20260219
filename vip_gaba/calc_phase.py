# 単一細胞で位相と初期値を計算
# %%
# %%
from utils_for_init import neuron_model
import numpy as np
from scipy.integrate import solve_ivp
import json
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator

np.random.seed(111)
# %%
# cell parameters
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

# time parameters
t_max = 2000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.001)

# cell numbers
cell_types = ["core"]
cells_per_type = 1
cells_num = len(cell_types) * cells_per_type
init = [0] * 6 * cells_num

# connection parameters
vip_weight = 0
gaba_matrix = np.zeros((cells_num, cells_num))
# %%
time_start = time.time()

cells = [
    neuron_model.SingleCell(cell_params, i, cell_type=cell_types[i // cells_per_type])
    for i in range(cells_num)
]

network = neuron_model.CellNetwork(cells, vip_weight, gaba_matrix, output_interval=100)
# %%
sol = solve_ivp(
    fun=lambda t, y: network.ode_all_neurons(t, y, light_func=lambda t: 0.0),  # 光なし
    t_span=t_span,
    y0=init,
    t_eval=t_eval,
    method="LSODA",
)
time_end = time.time()
print()
print(time_end - time_start)
# %%
M = sol.y[0, :]
Pc = sol.y[1, :]
Pn = sol.y[2, :]
Ca = sol.y[3, :]
V = sol.y[4, :]
VIP = sol.y[5, :]
t_eval = sol.t

# %%
V_peak, _ = find_peaks(V)
V_peaktime = t_eval[V_peak]
V_0 = 0
for index, value in enumerate(V_peaktime):
    if value > 1000:
        V_0 = int(value * 1000)
        V_2pi = int(V_peaktime[index + 1] * 1000)
        break
index_dic = dict()
for i in range(360):
    index_dic[i] = int(
        (V_2pi - V_0) / 360 * (i - 90) + V_0
    )  # それぞれの位相に対応するインデックスを作成
# %%
phase_initial_conditions = dict()
for key, value in index_dic.items():
    phase_initial_conditions[key] = dict()
    phase_initial_conditions[key]["M"] = M[value]
    phase_initial_conditions[key]["Pc"] = Pc[value]
    phase_initial_conditions[key]["Pn"] = Pn[value]
    phase_initial_conditions[key]["Ca"] = Ca[value]
    phase_initial_conditions[key]["V"] = V[value]
    phase_initial_conditions[key]["VIP"] = VIP[value]

with open("phase_initial_conditions.json", "w") as f:
    json.dump(phase_initial_conditions, f, indent=4)

# %%
# 確認用
with open("phase_initial_conditions.json", "r") as f:
    p_i_c = json.load(f)
fig, ax = plt.subplots()
ax.set_xlim((-10, 370))
ax.set_ylim((-0.1, 2.5))
ax.xaxis.set_major_locator(MultipleLocator(45))
for key, value in p_i_c.items():
    ax.scatter(key, value["M"], c="r", s=3)
    ax.scatter(key, value["Pc"] / 20, c="b", s=3)
    ax.scatter(key, value["Pn"] / 20, c="m", s=3)
    ax.scatter(key, value["Ca"] / 600, c="y", s=3)
    ax.scatter(key, value["V"] / 15, c="k", s=3)
    ax.scatter(key, value["VIP"], c="g", s=3)
fig.savefig("phase_initial_conditions.png")
# %%
