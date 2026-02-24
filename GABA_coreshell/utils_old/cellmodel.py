# 光入力と細胞モデルの微分方程式を記述
# %%
import numpy as np
# %%


class CellModel:
    def __init__(self, LD_params, cell_params, connect_param=None, pulse_param=None):
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
        k_dark = self.LD_params["k_dark"]
        t_remainder = t % (L_phase + D_phase)
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

    def Light_pulse(self, t):
        if self.pulse_param is not None:
            L_max = self.pulse_param["L_max"]
            pulse_width = self.pulse_param["pulse_width"]
            pulse_start = self.pulse_param["pulse_start"]
            if pulse_start <= t <= pulse_start + pulse_width:
                return L_max
            else:
                return 0
        else:
            return None

    # def Onecell_model(
    #     self, t, Y, LD=False, TTX_effect=False, effect_time_TTX=[], light_pulse=False
    # ):
    #     p = self.cell_params
    #     M, Pc, Pn, Ca, V = Y

    #     def pCREBp(Ca):
    #         return p["ap"] * Ca / (p["ap"] * Ca + p["kdp"])

    #     dM = (
    #         p["vs"] / (1 + (Pn / p["P0"]) ** p["n_p"])
    #         + p["vc"]
    #         * (pCREBp(Ca) / p["Kc"]) ** p["n_ca"]
    #         / (1 + (pCREBp(Ca) / p["Kc"]) ** p["n_ca"])
    #         - p["myu"] * M
    #     )
    #     dPc = p["ks"] * M - p["k1"] * Pc
    #     dPn = p["k1"] * Pc - p["vd"] * Pn / (p["Kd"] + Pn)
    #     dCa = 1 / p["tauCa"] * (p["betaCa"] * V - Ca)
    #     if light_pulse:
    #         dV = (
    #             1
    #             / p["taur"]
    #             * (
    #                 -V
    #                 + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"])
    #                 + self.Light_pulse(t)
    #             )
    #         )
    #     elif TTX_effect and effect_time_TTX[0] < t < effect_time_TTX[1]:
    #         dV = 1 / p["taur"] * (-V)
    #     elif (
    #         LD and t > self.LD_params["LD_time"][0] and t < self.LD_params["LD_time"][1]
    #     ):
    #         dV = (
    #             1
    #             / p["taur"]
    #             * (
    #                 -V
    #                 + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"])
    #                 + self.Light_strangth(t)
    #             )
    #         )
    #     else:
    #         dV = 1 / p["taur"] * (-V + p["alpha"] / (1 + (Pn / p["Pv"]) ** p["m"]))

    #     return [dM, dPc, dPn, dCa, dV]

    def Twocell_model(self, t, Y, LD=False, TTX_effect=False, effect_time_TTX=[]):
        p = self.cell_params
        c = self.connect_param
        (
            M_core,
            Pc_core,
            Pn_core,
            Ca_core,
            V_core,
            VIP,
            M_shell,
            Pc_shell,
            Pn_shell,
            Ca_shell,
            V_shell,
        ) = Y

        def pCREBp(Ca, b, VIP):
            return (p["ap"] * Ca + b * VIP) / (p["ap"] * Ca + b * VIP + p["kdp"])

        # --- core cell ---
        dM_core = (
            p["vs"] / (1 + (Pn_core / p["P0"]) ** p["n_p"])
            + p["vc"]
            * (pCREBp(Ca_core, c["b_core"], VIP) / p["Kc"]) ** p["n_ca"]
            / (1 + (pCREBp(Ca_core, c["b_core"], VIP) / p["Kc"]) ** p["n_ca"])
            - p["myu"] * M_core
        )
        dPc_core = p["ks"] * M_core - p["k1"] * Pc_core
        dPn_core = p["k1"] * Pc_core - p["vd"] * Pn_core / (p["Kd"] + Pn_core)
        dCa_core = 1 / p["tauCa"] * (p["betaCa"] * V_core - Ca_core)
        dVIP = (
            p["us"] / (1 + np.exp(-p["lambdaVIP"] * (V_core - p["V0"])))
            - p["eta"] * VIP
        )
        if TTX_effect and effect_time_TTX[0] < t < effect_time_TTX[1]:
            dV_core = -V_core / p["taur"]
        elif LD and self.LD_params["LD_time"][0] < t < self.LD_params["LD_time"][1]:
            dV_core = (
                1
                / p["taur"]
                * (
                    -V_core
                    + max(
                        p["alpha"] / (1 + (Pn_core / p["Pv"]) ** p["m"])
                        + self.Light_strangth(t)
                        - c["gsa"] * V_shell,
                        0,
                    )
                )
            )
        else:
            dV_core = (
                1
                / p["taur"]
                * (
                    -V_core
                    + max(
                        p["alpha"] / (1 + (Pn_core / p["Pv"]) ** p["m"])
                        - c["gsa"] * V_shell,
                        0,
                    )
                )
            )

        # --- shell cell ---
        dM_shell = (
            p["vs"] / (1 + (Pn_shell / p["P0"]) ** p["n_p"])
            + p["vc"]
            * (pCREBp(Ca_shell, c["b_shell"], VIP) / p["Kc"]) ** p["n_ca"]
            / (1 + (pCREBp(Ca_shell, c["b_shell"], VIP) / p["Kc"]) ** p["n_ca"])
            - p["myu"] * M_shell
        )
        dPc_shell = p["ks"] * M_shell - p["k1"] * Pc_shell
        dPn_shell = p["k1"] * Pc_shell - p["vd"] * Pn_shell / (p["Kd"] + Pn_shell)
        dCa_shell = 1 / p["tauCa"] * (p["betaCa"] * V_shell - Ca_shell)
        dV_shell = (
            -V_shell / p["taur"]
            if TTX_effect and effect_time_TTX[0] < t < effect_time_TTX[1]
            else 1
            / p["taur"]
            * (-V_shell + p["alpha"] / (1 + (Pn_shell / p["Pv"]) ** p["m"]))
        )

        return [
            dM_core,
            dPc_core,
            dPn_core,
            dCa_core,
            dV_core,
            dVIP,
            dM_shell,
            dPc_shell,
            dPn_shell,
            dCa_shell,
            dV_shell,
        ]


# %%
