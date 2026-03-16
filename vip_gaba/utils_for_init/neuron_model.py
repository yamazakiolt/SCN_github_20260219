import numpy as np


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
        state: np.ndarray,
        light: float = 0.0,
        vip_input: float = 0.0,
        gaba_input: float = 0.0,
        ttx_active: bool = False,
    ) -> np.ndarray:
        p = self.params
        M, Pc, Pn, Ca, V, VIP = state

        def pCREBp(Ca, vip_input):
            return (p["ap"] * Ca + vip_input) / (p["ap"] * Ca + vip_input + p["kdp"])

        dM = p["vs"] / (1 + (Pn / p["P0"]) ** p["n_p"])
        dM += (
            p["vc"]
            * (pCREBp(Ca, vip_input) / p["Kc"]) ** p["n_ca"]
            / (1 + (pCREBp(Ca, vip_input) / p["Kc"]) ** p["n_ca"])
        )
        dM -= p["myu"] * M

        dPc = p["ks"] * M - p["k1"] * Pc
        dPn = p["k1"] * Pc - p["vd"] * Pn / (p["Kd"] + Pn)
        dCa = 1 / p["tauCa"] * (p["betaCa"] * V - Ca)

        if ttx_active:
            dV = -V / p["taur"]
        else:
            V_activation = (
                p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"]) + light - gaba_input
            )
            dV = 1 / p["taur"] * (-V + max(V_activation, 0))

        dVIP = p["us"] / (1 + np.exp(-p["lambdaVIP"] * (V - p["V0"]))) - p["eta"] * VIP

        return np.array([dM, dPc, dPn, dCa, dV, dVIP])


class CellNetwork:
    def __init__(
        self,
        cell_list: list,
        vip_weight: float,
        gaba_matrix: np.ndarray,
        output_interval=None,
    ):
        self.cells = cell_list
        self.vip_weight = vip_weight
        self.gaba_conn = gaba_matrix
        self.output_interval = output_interval
        self.outputed_time = set()

    def unflatten_states(self, flat_state: np.ndarray):
        idx = 0
        for cell in self.cells:
            size = cell.state_size
            cell.state = flat_state[idx : idx + size]
            idx += size

    def ode_all_neurons(
        self, t: float, flat_state: np.ndarray, light_func, ttx_times=[]
    ) -> np.ndarray:
        self.unflatten_states(flat_state)
        dY = []

        # 各細胞の VIP 値と V 値を収集
        vip_vals = [cell.state[5] for cell in self.cells if cell.cell_type == "core"]
        V_vals = [cell.state[4] for cell in self.cells]

        for i, cell in enumerate(self.cells):
            # 光入力（core細胞にのみ）
            light = light_func(t) if cell.cell_type == "core" else 0.0

            # VIPとGABAの結合入力（接続行列 × 値）
            vip_input = self.vip_weight * np.mean(vip_vals)
            gaba_input = np.dot(self.gaba_conn[:, i], V_vals)

            # TTXのタイミング判定
            ttx_active = bool(ttx_times) and (ttx_times[0] <= t <= ttx_times[1])

            dydt = cell.ode_singleneuron(
                t,
                cell.state,
                light=light,
                vip_input=vip_input,
                gaba_input=gaba_input,
                ttx_active=ttx_active,
            )
            dY.append(dydt)

        if self.output_interval:
            if int(t) % self.output_interval == 0 and int(t) not in self.outputed_time:
                print(int(t))
                self.outputed_time.add(int(t))

        return np.concatenate(dY)


class LightModel:
    def __init__(self, LD_params, pulse_param):
        self.LD_params = LD_params
        self.pulse_param = pulse_param

    def light_strength(self, t):
        L_phase = self.LD_params["L_phase"]  # 明期時間
        D_phase = self.LD_params["D_phase"]  # 暗期時間
        L_max = self.LD_params["L_max"]
        L_min = self.LD_params["L_min"]
        k_light = self.LD_params["k_light"]
        LD_time = self.LD_params["LD_time"]

        t_remainder = t % (L_phase + D_phase)

        if LD_time[0] < t < LD_time[1]:
            if t_remainder < D_phase / 2:
                val = (np.tanh(4 / D_phase * t_remainder - 4) + 1) / 2
            elif t_remainder < D_phase / 2 + L_phase / 2:
                val = (
                    np.tanh(
                        (2 * k_light + 4) / L_phase * t_remainder
                        - (k_light + 2) * D_phase / L_phase
                        - 2
                    )
                    + 1
                ) / 2
            elif t_remainder < D_phase / 2 + L_phase:
                val = (
                    -np.tanh(
                        (2 * k_light + 4) / L_phase * t_remainder
                        - k_light
                        - (k_light + 2) / L_phase * (D_phase + L_phase)
                    )
                    + 1
                ) / 2
            else:
                val = (
                    -np.tanh(4 / D_phase * t_remainder - 4 * L_phase / D_phase) + 1
                ) / 2
            return val * (L_max - L_min) + L_min
        else:
            return 0

    def light_pulse(self, t):
        L_max = self.pulse_param["L_max"]
        pulse_width = self.pulse_param["pulse_width"]
        pulse_start = self.pulse_param["pulse_start"]

        if pulse_start <= t <= pulse_start + pulse_width:
            return L_max
        else:
            return 0.0
