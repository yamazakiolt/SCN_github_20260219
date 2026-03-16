# %%
import argparse

from utils import neuron_model
import numpy as np
from scipy.integrate import solve_ivp
import json
import pandas as pd
import time
import os

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--vip_core", type=int, required=True)
parser.add_argument("--vip_shell", type=int, required=True)
parser.add_argument("--core_cells", type=int, required=True)
parser.add_argument("--shell_cells", type=int, required=True)
parser.add_argument("--seed_base", type=int, required=True)
args = parser.parse_args()

vip_core = args.vip_core
vip_shell = args.vip_shell
cells_num_each_type = [args.core_cells, args.shell_cells]
seed_base = args.seed_base
# %%


def detect_threshold_crossings(t, V, threshold):
    """Vがthresholdを上向きに超える時刻を抽出"""
    crossings = []
    for i in range(1, len(V)):
        if V[i - 1] < threshold and V[i] >= threshold:
            # 線形補間でより正確な交点を求める
            frac = (threshold - V[i - 1]) / (V[i] - V[i - 1])
            t_cross = t[i - 1] + frac * (t[i] - t[i - 1])
            crossings.append(t_cross)
    return np.array(crossings)


def evaluate_period_all(df, threshold, LD_time_max):
    """閾値法で周期を判定する"""
    period_data = dict()
    period_data_entrain = dict()
    for col in [c for c in df.columns if c.startswith("V")]:
        t = df["t"].values
        V = df[col].values

        crossings = detect_threshold_crossings(t, V, threshold)
        crossings_entrain = crossings[crossings < LD_time_max]
        if len(crossings) < 5:
            period_data[col] = [False, 0]
        else:
            # 各交差間隔の平均を取る（＝推定周期）
            intervals = np.diff(crossings)
            mean_period = np.mean(intervals[-13:-3])
            std_period = np.std(intervals[-13:-3])

            # 周期が光周期に近ければ同調とみなす
            period_data[col] = [std_period < 0.1, mean_period]

        if len(crossings_entrain) < 5:
            period_data_entrain[col] = [False, 0]
        else:
            # 各交差間隔の平均を取る（＝推定周期）
            intervals_entrain = np.diff(crossings_entrain)
            mean_period_entrain = np.mean(intervals_entrain[-13:-3])
            std_period_entrain = np.std(intervals_entrain[-13:-3])

            # 周期が光周期に近ければ同調とみなす
            period_data_entrain[col] = [std_period_entrain < 0.1, mean_period_entrain]
    return period_data, period_data_entrain


def evaluate_phase(df, threshold):
    """閾値法で位相差を判定する"""
    phase_data = dict()
    t = df["t"].values
    V0 = df["V0"].values

    crossings_0 = detect_threshold_crossings(t, V0, threshold)
    crossing_0_3500 = crossings_0[crossings_0 > 3500]
    if crossing_0_3500.size > 0:
        crossing_0_3500_min = crossing_0_3500.min()
    else:
        crossing_0_3500_min = None
    for col in [c for c in df.columns if c.startswith("V")]:
        t = df["t"].values
        V = df[col].values

        crossings = detect_threshold_crossings(t, V, threshold)
        if len(crossings) < 5:
            period_data[col] = [False, 0]
            continue
        if crossing_0_3500_min:
            phase_data[col] = crossings[
                np.abs(crossings - crossing_0_3500_min).argmin()
            ]
        else:
            phase_data[col] = None

    return phase_data


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
t_max = 4000
time_interval = 0.01
cell_types = ["core", "shell"]
cell_type_each_cell = []
for index, cell_type in enumerate(cell_types):
    for _ in range(cells_num_each_type[index]):
        cell_type_each_cell.append(cell_type)
cells_num = sum(cells_num_each_type)
L_max = 0.2
LD_time_max = 2000
# %%
# %%

vip_weight = []
for cell_index in range(cells_num):
    if cell_type_each_cell[cell_index] == "core":
        vip_weight.append(vip_core)
    elif cell_type_each_cell[cell_index] == "shell":
        vip_weight.append(vip_shell)

for gaba_sc in np.arange(0, 0.205, 0.005):
    gaba_sc = round(gaba_sc, 3)
    gaba_matrix = np.zeros((cells_num, cells_num))
    for receive in range(cells_num):
        for emit in range(cells_num):
            if (
                cell_type_each_cell[emit] == "shell"
                and cell_type_each_cell[receive] == "core"
            ):
                gaba_matrix[receive][emit] = gaba_sc / cells_num * 2
    time_start = time.time()
    print(f"    gaba_sc:{gaba_sc}")
    all_period_results = []
    all_period_results_entrain = []
    outdir = f"./VIP_GABA_{cells_num_each_type[0]}&{cells_num_each_type[1]}cell_aftereffect/{L_max}/{vip_core}_{vip_shell}"
    os.makedirs(outdir, exist_ok=True)
    outdir_entrain = f"./VIP_GABA_{cells_num_each_type[0]}&{cells_num_each_type[1]}cell_entrain/{L_max}/{vip_core}_{vip_shell}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_entrain, exist_ok=True)
    for rand, L_phase in enumerate(np.arange(10, 14.05, 0.1)):
        L_phase = round(L_phase, 2)
        D_phase = L_phase
        rng = np.random.default_rng(seed_base + rand * 5)
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
            neuron_model.SingleCell(cell_params, i, cell_type=cell_type_each_cell[i])
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
            "LD_time": [
                0,
                LD_time_max // (L_phase + D_phase) * (L_phase + D_phase),
            ],
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
            fun=lambda t, y: network.ode_all_neurons(
                t, y, light_func_args=light_func_args
            ),
            t_span=[0, t_max],
            t_eval=t_eval,
            y0=y0,
            method="LSODA",
        )

        light_func = neuron_model.LightModel(LD_params)
        t_eval = sol.t
        light_strangth = [light_func.light_strength(t) for t in t_eval]
        for i0 in range(cells_num):
            linestyle = "-" if cell_type_each_cell[i0] == "core" else "--"
            M_i0 = sol.y[i0 * 6, :]
            Pc_i0 = sol.y[i0 * 6 + 1, :]
            Pn_i0 = sol.y[i0 * 6 + 2, :]
            Ca_i0 = sol.y[i0 * 6 + 3, :]
            V_i0 = sol.y[i0 * 6 + 4, :]
            VIP_i0 = sol.y[i0 * 6 + 5, :]

        # --- 保存（CSV形式で可視化しやすく） ---
        V_chunk = np.array([sol.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk = pd.DataFrame(V_chunk.T, columns=[f"V{i}" for i in range(cells_num)])
        df_chunk.insert(0, "t", sol.t)
        period_data, period_data_entrain = evaluate_period_all(
            df=df_chunk, threshold=3, LD_time_max=LD_time_max
        )
        phase_data = evaluate_phase(df=df_chunk, threshold=3)
        for cell_name, (is_sync, period) in period_data.items():
            all_period_results.append(
                {
                    "gaba_sc": gaba_sc,
                    "rand_seed": seed_base + rand * 5,
                    "cell_name": cell_name,
                    "is_sync": is_sync,
                    "period": period,
                    "L_phase": L_phase,
                    "D_phase": D_phase,
                    "phase": phase_data[cell_name],
                }
            )
        for cell_name, (is_sync, period_entrain) in period_data_entrain.items():
            all_period_results_entrain.append(
                {
                    "gaba_sc": gaba_sc,
                    "rand_seed": seed_base + rand * 5,
                    "cell_name": cell_name,
                    "is_sync": is_sync,
                    "period": period_entrain,
                    "L_phase": L_phase,
                    "D_phase": D_phase,
                }
            )

        del sol

    if all_period_results:
        df_period = pd.DataFrame(all_period_results)
        df_period.to_csv(
            os.path.join(outdir, f"evaluate_period_summary_{gaba_sc}.csv"), index=False
        )
    if all_period_results_entrain:
        df_period_entrain = pd.DataFrame(all_period_results_entrain)
        df_period_entrain.to_csv(
            os.path.join(outdir_entrain, f"evaluate_period_summary_{gaba_sc}.csv"),
            index=False,
        )
    seed_base += 1000
    time_end = time.time()
    print(f"    {round(time_end - time_start, 3)} s")


# %%
