import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def ode_singleneuron_numba(M, Pc, Pn, Ca, V, VIP, light, vip_input, gaba_input, p_arr):
    # p_arr は shape=(24,) の配列
    # [cell.params['ap'], cell.params['kdp'], cell.params['vs'], cell.params['P0'], cell.params['n_p'],
    # cell.params['vc'], cell.params['Kc'], cell.params['n_ca'], cell.params['myu'], cell.params['ks'],
    # cell.params['k1'], cell.params['vd'], cell.params['Kd'], cell.params['tauCa'], cell.params['betaCa'],
    # cell.params['taur'], cell.params['alpha'], cell.params['Pv'], cell.params['m'], cell.params['us'],
    # cell.params['lambdaVIP'], cell.params['V0'], cell.params['eta']]
    pCREBp = (p_arr[0] * Ca + vip_input) / (p_arr[0] * Ca + vip_input + p_arr[1])
    dM = p_arr[2] / (1 + (Pn / p_arr[3]) ** p_arr[4])
    dM += (
        p_arr[5]
        * (pCREBp / p_arr[6]) ** p_arr[7]
        / (1 + (pCREBp / p_arr[6]) ** p_arr[7])
    )
    dM -= p_arr[8] * M
    dPc = p_arr[9] * M - p_arr[10] * Pc
    dPn = p_arr[10] * Pc - p_arr[11] * Pn / (p_arr[12] + Pn)
    dCa = (p_arr[14] * V - Ca) / p_arr[13]
    V_activation = np.maximum(
        p_arr[16] / (1 + (Pn / p_arr[17]) ** p_arr[18]) - gaba_input + light, 0
    )
    dV = (-V + V_activation) / p_arr[15]
    dVIP = p_arr[19] / (1 + np.exp(-p_arr[20] * (V - p_arr[21]))) - p_arr[22] * VIP
    return np.array([dM, dPc, dPn, dCa, dV, dVIP])


@jit(nopython=True, fastmath=True)
def ode_all_neurons_numba(
    t,
    flat_state,
    cell_types,
    vip_weight,
    gaba_conn,
    params_arr,
    light_func_args,
    TTX_start,
    TTX_end,
):
    N = flat_state.size // 6
    states = flat_state.reshape(N, 6)
    dY = np.empty_like(states)
    core_count = np.sum(cell_types)
    L_phase, D_phase, L_max, L_min, k_light, k_dark, LD_start, LD_end = light_func_args
    vip_mean = np.sum(states[:, 5] * cell_types) / core_count if core_count > 0 else 0.0

    V_vals = np.ascontiguousarray(states[:, 4])

    LD_active = LD_start < t < LD_end
    for i in range(N):
        light = (
            light_strength_numba(
                t, L_phase, D_phase, L_max, L_min, k_light, k_dark, LD_start, LD_end
            )
            if (LD_active and cell_types[i] == 1)
            else 0.0
        )
        vip_input = vip_weight[i] * vip_mean
        gaba_input = np.dot(gaba_conn[i], V_vals)
        p = params_arr[i]

        if TTX_start < t < TTX_end:
            states[i, 4] = 0.01  # V = 0.01
        d_state = ode_singleneuron_numba(
            states[i, 0],
            states[i, 1],
            states[i, 2],
            states[i, 3],
            states[i, 4],
            states[i, 5],
            light,
            vip_input,
            gaba_input,
            p,
        )
        dY[i, :] = d_state

    return dY.reshape(-1)


@jit(nopython=True, fastmath=True)
def light_strength_numba(
    t, L_phase, D_phase, L_max, L_min, k_light, k_dark, LD_start, LD_end
):
    period = L_phase + D_phase
    t_remainder = t % period
    if LD_start < t < LD_end:
        if t_remainder < D_phase / 2:
            tanh_phase = k_dark * (t_remainder / (D_phase / 2) - 1)
        elif t_remainder < D_phase / 2 + L_phase / 2:
            tanh_phase = k_light / (L_phase / 2) * (t_remainder - (D_phase / 2))
        elif t_remainder < D_phase / 2 + L_phase:
            t_remainder_mirror = (L_phase + D_phase) - t_remainder
            tanh_phase = k_light / (L_phase / 2) * (t_remainder_mirror - (D_phase / 2))
        else:
            t_remainder_mirror = (L_phase + D_phase) - t_remainder
            tanh_phase = k_dark * (t_remainder_mirror / (D_phase / 2) - 1)
        return (np.tanh(tanh_phase) + 1) / 2 * (L_max - L_min) + L_min
    else:
        return 0.0


# Pythonクラスはそのまま（呼び出しはnumba関数を使うこと）


class SingleCell:
    def __init__(self, params: dict, cell_id: int, cell_type: str = "core"):
        self.params = params
        self.id = cell_id
        self.cell_type = cell_type
        self.state_size = 6
        self.state = np.zeros(self.state_size)

    def ode_singleneuron(
        self,
        t: float,
        state=np.array,
        light: float = 0.0,
        vip_input: float = 0.0,
        gaba_input: float = 0.0,
    ) -> np.ndarray:
        params_arr = np.array(
            [
                self.params["ap"],
                self.params["kdp"],
                self.params["vs"],
                self.params["P0"],
                self.params["n_p"],
                self.params["vc"],
                self.params["Kc"],
                self.params["n_ca"],
                self.params["myu"],
                self.params["ks"],
                self.params["k1"],
                self.params["vd"],
                self.params["Kd"],
                self.params["tauCa"],
                self.params["betaCa"],
                self.params["taur"],
                self.params["alpha"],
                self.params["Pv"],
                self.params["m"],
                self.params["us"],
                self.params["lambdaVIP"],
                self.params["V0"],
                self.params["eta"],
            ],
            dtype=np.float64,
        )
        M, Pc, Pn, Ca, V, VIP = state
        return ode_singleneuron_numba(
            M, Pc, Pn, Ca, V, VIP, light, vip_input, gaba_input, params_arr
        )


class CellNetwork:
    def __init__(
        self,
        cell_list: list,
        vip_weight: list,
        gaba_matrix: np.ndarray,
    ):
        self.cells = cell_list
        self.vip_weight = np.array(vip_weight)
        self.gaba_conn = gaba_matrix

    def unflatten_states(self, flat_state: np.ndarray):
        idx = 0
        for cell in self.cells:
            size = cell.state_size
            cell.state = flat_state[idx : idx + size]
            idx += size

    def ode_all_neurons(
        self,
        t: float,
        flat_state: np.ndarray,
        light_func_args,
        TTX_start,
        TTX_end,
    ) -> np.ndarray:
        # cell_types: 'core'->1, 'shell'->0
        cell_types = np.array(
            [1 if cell.cell_type == "core" else 0 for cell in self.cells], dtype=np.int8
        )
        params_arr = np.array(
            [
                [
                    cell.params["ap"],
                    cell.params["kdp"],
                    cell.params["vs"],
                    cell.params["P0"],
                    cell.params["n_p"],
                    cell.params["vc"],
                    cell.params["Kc"],
                    cell.params["n_ca"],
                    cell.params["myu"],
                    cell.params["ks"],
                    cell.params["k1"],
                    cell.params["vd"],
                    cell.params["Kd"],
                    cell.params["tauCa"],
                    cell.params["betaCa"],
                    cell.params["taur"],
                    cell.params["alpha"],
                    cell.params["Pv"],
                    cell.params["m"],
                    cell.params["us"],
                    cell.params["lambdaVIP"],
                    cell.params["V0"],
                    cell.params["eta"],
                ]
                for cell in self.cells
            ],
            dtype=np.float64,
        )
        return ode_all_neurons_numba(
            t,
            flat_state,
            cell_types,
            self.vip_weight,
            self.gaba_conn,
            params_arr,
            light_func_args,
            TTX_start,
            TTX_end,
        )


class LightModel:
    def __init__(self, LD_params=None, pulse_param=None):
        self.LD_params = LD_params
        self.pulse_param = pulse_param

    def light_strength(self, t):
        if self.LD_params is None:
            raise ValueError("LD_params is not set.")
        L_phase = self.LD_params["L_phase"]
        D_phase = self.LD_params["D_phase"]
        L_max = self.LD_params["L_max"]
        L_min = self.LD_params["L_min"]
        k_light = self.LD_params["k_light"]
        k_dark = self.LD_params["k_dark"]
        LD_time = self.LD_params["LD_time"]
        return light_strength_numba(
            t, L_phase, D_phase, L_max, L_min, k_light, k_dark, LD_time[0], LD_time[1]
        )
