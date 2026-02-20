# Figure3.1のデータを作成する
# phase_osci_resultsが作成され、そこに位相の時間変化が保存される
# %%
import numpy as np
from scipy.integrate import solve_ivp
import json
import os

# %%
os.makedirs("./phase_osci_results", exist_ok=True)
# %%


def phase_osci(t, Y, ω, ksc, kL, kcs, ωL):
    ΦC, ΦS, ΦL = Y
    dΦCdt = ω + ksc * np.sin(ΦS - ΦC) + kL * np.sin(ΦL - ΦC)
    dΦSdt = ω + kcs * np.sin(ΦC - ΦS)
    dΦLdt = ωL

    return [dΦCdt, dΦSdt, dΦLdt]


def list_sparse(list, span):
    new_list = []
    for i in range(len(list)):
        if i % span == 0:
            new_list.append(list[i])
    return new_list


# %%
ω = 2 * np.pi / 24
kL = 0.1

init = [0, 0, 0]
t_span = [0, 3000]
t_eval = np.arange(0, 3000, 0.001)


for T_light in [10, 16, 24, 32]:
    params = []
    phase_diffs_CS = []
    phase_diffs_LC = []
    list_L = []
    list_C = []
    list_S = []
    print(f"T_light = {T_light}")
    ωL = 2 * np.pi / T_light
    for ksc in np.arange(-1, 1.1, 0.1):
        ksc = round(ksc, 1)
        print(f" ksc = {ksc}")
        for kcs in np.arange(-1, 1.1, 0.1):
            kcs = round(kcs, 1)
            print(f"  kcs = {kcs}")
            sol_phase_osci = solve_ivp(
                phase_osci,
                t_span,
                init,
                t_eval=t_eval,
                method="LSODA",
                args=(ω, ksc, kL, kcs, ωL),
            )
            # 平滑化
            n_conv = 100
            b = np.ones(n_conv) / n_conv
            times = sol_phase_osci.t
            ΦC = np.convolve(sol_phase_osci.y[0, :], b, mode="same")
            ΦS = np.convolve(sol_phase_osci.y[1, :], b, mode="same")
            ΦL = np.convolve(sol_phase_osci.y[2, :], b, mode="same")

            phase_diff_CS = ΦS - ΦC
            phase_diff_LC = ΦC - ΦL
            params.append([ksc, kcs])
            list_C.append(list_sparse(ΦC, 1000))
            list_L.append(list_sparse(ΦL, 1000))
            list_S.append(list_sparse(ΦS, 1000))

    with open(f"./phase_osci_results/params_{T_light}.json", "w") as f:
        json.dump(params, f, indent=4)
    with open(f"./phase_osci_results/C_{T_light}.json", "w") as f:
        json.dump(list_C, f, indent=4)
    with open(f"./phase_osci_results/L_{T_light}.json", "w") as f:
        json.dump(list_L, f, indent=4)
    with open(f"./phase_osci_results/S_{T_light}.json", "w") as f:
        json.dump(list_S, f, indent=4)

# %%
