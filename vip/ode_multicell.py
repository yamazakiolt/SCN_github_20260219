# narutal_freq_allを変化させて定数倍にする
# %%
from utils_freq_2 import neuron_model
import numpy as np
from scipy.integrate import solve_ivp
import json
import pandas as pd
import time
import os

rng = np.random.default_rng(110)
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
with open("phase_initial_conditions.json", "r") as f:
    phase_initial_conditions = json.load(f)
t_max = 2000
time_interval = 0.01
cell_types = ["core", "shell"]
cells_num_each_type = [10, 10]
cell_type_each_cell = []
for index, cell_type in enumerate(cell_types):
    for _ in range(cells_num_each_type[index]):
        cell_type_each_cell.append(cell_type)
cells_num = sum(cells_num_each_type)
gaba_matrix = np.zeros((cells_num, cells_num))  # [receive, emit] 行列演算のため
L_max = 0
LD_time_max = 2000
# %%
T_mean = 24
T_std = 1.4
rng = np.random.default_rng(110)
cells_T = rng.normal(T_mean, T_std, cells_num)
print(np.mean(cells_T), np.std(cells_T, ddof=1))
natural_freq_all = 24 / cells_T
# %%
rand_index = 0
for w_vip in [i * 10 for i in range(11)]:
    print(f"W_VIP : {w_vip}")
    vip_weight = [w_vip] * cells_num
    L_phase = 12
    D_phase = 12
    time_start = time.time()
    outdir = f"./results/multicell/{cells_num}/{w_vip}"
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(110 + rand_index)
    phase_init = [rng.integers(360) for _ in range(cells_num)]
    y0 = []
    for phase in phase_init:
        phase = str(phase)
        y0.extend(
            [
                phase_initial_conditions[phase]["M"],
                phase_initial_conditions[phase]["Pc"],
                phase_initial_conditions[phase]["Pn"],
                phase_initial_conditions[phase]["Ca"],
                phase_initial_conditions[phase]["V"],
                phase_initial_conditions[phase]["VIP"],
            ]
        )
    cells = [
        neuron_model.SingleCell(
            cell_params,
            i,
            cell_type=cell_type_each_cell[i],
            natural_freq=natural_freq_all[i],
        )
        for i in range(cells_num)
    ]
    network = neuron_model.CellNetwork(cells, vip_weight, gaba_matrix)
    LD_params = {
        "L_phase": L_phase,
        "D_phase": D_phase,
        "L_max": L_max,
        "L_min": 0,
        "k_light": L_phase,
        "k_dark": D_phase,
        "LD_time": [0, LD_time_max // (L_phase + D_phase) * (L_phase + D_phase)],
    }
    light_func_args = (
        LD_params["L_phase"],
        LD_params["D_phase"],
        LD_params["L_max"],
        LD_params["L_min"],
        LD_params["k_light"],
        LD_params["k_dark"],
        LD_params["LD_time"][0],
        LD_params["LD_time"][1],
    )
    t_eval = np.round(np.arange(0, t_max, time_interval), 3)
    sol = solve_ivp(
        fun=lambda t, y: network.ode_all_neurons(t, y, light_func_args=light_func_args),
        t_span=[0, t_max],
        t_eval=t_eval,
        y0=y0,
        method="LSODA",
    )

    light_func = neuron_model.LightModel(LD_params)
    t_eval = sol.t
    light_strangth = [light_func.light_strength(t) for t in t_eval]
    # --- 保存（CSV形式で可視化しやすく） ---
    V_chunk = np.array([sol.y[i * 6 + 4, :] for i in range(cells_num)])
    df_chunk = pd.DataFrame(V_chunk.T, columns=[f"V{i}" for i in range(cells_num)])
    df_chunk.insert(0, "t", sol.t)
    df_chunk.to_csv(os.path.join(outdir, "V.csv"), index=False)

    del sol

    rand_index += 5
    time_end = time.time()
    print(f"    {round(time_end - time_start, 3)} s")
# %%
