# %%
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# %%
os.makedirs("./phase_osci_LSA_results", exist_ok=True)

# %%
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.titlesize": 10,
        "axes.labelsize": 11,
    }
)


# 式から導出された同調範囲をプロット
kL = 0.1
ω = 2 * np.pi / 24
params = []
judge_entrainment = []

for T_light in [10, 16, 24, 32]:
    print(f"T_light = {T_light}")
    ωL = 2 * np.pi / T_light
    for ksc in np.arange(-1, 1.1, 0.1):
        print(f" ksc = {ksc}")
        for kcs in np.arange(-1, 1.1, 0.1):
            print(f"  kcs = {kcs}")
            ans = False
            sinCS = (ωL - ω) / kcs
            sinCL = -(ksc + kcs) / kL * sinCS
            if -1 <= sinCS <= 1 and -1 <= sinCL <= 1:  # 三角関数の成立条件
                ans = True
            params.append([T_light, ksc, kcs])
            judge_entrainment.append(ans)


with open("./phase_osci_LSA_results/params.json", "w") as f:
    json.dump(params, f, indent=4)
with open("./phase_osci_LSA_results/entrainment.json", "w") as f:
    json.dump(judge_entrainment, f, indent=4)


# %%
