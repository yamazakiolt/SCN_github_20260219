# 光入力と細胞モデルの微分方程式を記述
# %%
import numpy as np
# %%


class CellModel:
    def __init__(self, LD_params, cell_params, connect_param=False, pulse_param=False):
        self.LD_params = LD_params
        self.cell_params = cell_params
        self.pulse_param = pulse_param
        if connect_param:
            self.connect_param = connect_param

    def Light_strangth(self, t):
        L_phase = self.LD_params["L_phase"]  # 明るい時間
        D_phase = self.LD_params["D_phase"]  # 暗い時間
        L_max = self.LD_params["L_max"]  # 最大の明るさ
        L_min = self.LD_params["L_min"]  # 最小の明るさ
        k_light = self.LD_params["k_light"]  # 関数の形を規定
        t_remainder = t % (L_phase + D_phase)
        if t_remainder < D_phase / 2:
            ans = (np.tanh(4 / D_phase * t_remainder - 4) + 1) / 2 * (
                L_max - L_min
            ) + L_min
        elif t_remainder < D_phase / 2 + L_phase / 2:
            ans = (
                np.tanh(
                    (2 * k_light + 4) / L_phase * t_remainder
                    - (k_light + 2) * D_phase / L_phase
                    - 2
                )
                + 1
            ) / 2 * (L_max - L_min) + L_min
        elif t_remainder < D_phase / 2 + L_phase:
            ans = (
                -np.tanh(
                    (2 * k_light + 4) / L_phase * t_remainder
                    - k_light
                    - (k_light + 2) / L_phase * (D_phase + L_phase)
                )
                + 1
            ) / 2 * (L_max - L_min) + L_min
        else:
            ans = (
                -np.tanh(4 / D_phase * t_remainder - 4 * L_phase / D_phase) + 1
            ) / 2 * (L_max - L_min) + L_min
        return ans

    def Light_pulse(self, t):
        L_max = self.pulse_param["L_max"]
        pulse_width = self.pulse_param["pulse_width"]
        pulse_start = self.pulse_param["pulse_start"]
        if pulse_start <= t <= pulse_start + pulse_width:
            return L_max
        else:
            return 0

    def Onecell_model(
        self, t, Y, LD=False, TTX_effect=False, effect_time_TTX=[], light_pulse=False
    ):
        p = self.cell_params
        M, Pc, Pn, Ca, V = Y

        def pCREBp(Ca):
            return p["ap"] * Ca / (p["ap"] * Ca + p["kdp"])

        dM = (
            p["vs"] / (1 + (Pn / p["P0"]) ** p["n_p"])
            + p["vc"]
            * (pCREBp(Ca) / p["Kc"]) ** p["n_ca"]
            / (1 + (pCREBp(Ca) / p["Kc"]) ** p["n_ca"])
            - p["myu"] * M
        )
        dPc = p["ks"] * M - p["k1"] * Pc
        dPn = p["k1"] * Pc - p["vd"] * Pn / (p["Kd"] + Pn)
        dCa = 1 / p["tauCa"] * (p["betaCa"] * V - Ca)
        if light_pulse:
            dV = (
                1
                / p["taur"]
                * (
                    -V
                    + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"])
                    + self.Light_pulse(t)
                )
            )
        elif TTX_effect and effect_time_TTX[0] < t < effect_time_TTX[1]:
            dV = 1 / p["taur"] * (-V)
        elif (
            LD and t > self.LD_params["LD_time"][0] and t < self.LD_params["LD_time"][1]
        ):
            dV = (
                1
                / p["taur"]
                * (
                    -V
                    + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"])
                    + self.Light_strangth(t)
                )
            )
        else:
            dV = 1 / p["taur"] * (-V + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"]))

        return [dM, dPc, dPn, dCa, dV]


# %%
