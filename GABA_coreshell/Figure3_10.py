# GABAありとなしで比較
# %%
# %%
from utils_old import cellmodel  # 自作関数の読み込み
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

# %%
os.makedirs("results", exist_ok=True)
# %%


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 3,  # 線の太さ
    }
)


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

t_max = 4000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.01)
# %%
seed_ab = 222
seed_cd = 221
np.random.seed(seed_ab)
init_ab = np.random.rand(11) * np.array(
    [1.25, 40, 30, 700, 15, 1, 1.25, 40, 30, 700, 15]
)
np.random.seed(seed_cd)
init_cd = np.random.rand(11) * np.array(
    [1.25, 40, 30, 700, 15, 1, 1.25, 40, 30, 700, 15]
)

LD_params_0 = {
    "LD_time": [0, t_max],
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
}  # no light

con_param_noGABA = {
    "b_core": 0,  # noVIP
    "b_shell": 0,
    "gsa": 0,
}
con_param_GABA = {
    "b_core": 0,  # noVIP
    "b_shell": 0,
    "gsa": 0.1,
}


model_noGABA = cellmodel.CellModel(
    LD_params_0, cell_params, connect_param=con_param_noGABA
)
model_GABA = cellmodel.CellModel(LD_params_0, cell_params, connect_param=con_param_GABA)


sol_noGABA_ab = solve_ivp(
    fun=lambda t, y: model_noGABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_ab,
    t_eval=t_eval,
    method="LSODA",
)  # no light no GABA

sol_noGABA_cd = solve_ivp(
    fun=lambda t, y: model_noGABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_cd,
    t_eval=t_eval,
    method="LSODA",
)  # no light no GABA

init_GABA_ab = sol_noGABA_ab.y[:, 300 * 100]
init_GABA_cd = sol_noGABA_cd.y[:, 300 * 100]

sol_GABA_ab = solve_ivp(
    fun=lambda t, y: model_GABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_GABA_ab,
    t_eval=t_eval,
    method="LSODA",
)

sol_GABA_cd = solve_ivp(
    fun=lambda t, y: model_GABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_GABA_cd,
    t_eval=t_eval,
    method="LSODA",
)

caption_list = [["(a)", "(b)"], ["(c)", "(d)"]]
sol_noGABA_list = [sol_noGABA_ab, sol_noGABA_cd]
sol_GABA_list = [sol_GABA_ab, sol_GABA_cd]

for caption, sol_noGABA, sol_GABA in zip(caption_list, sol_noGABA_list, sol_GABA_list):
    n_conv = 100
    window_conv = np.ones(n_conv) / n_conv

    V_core_noGABA = np.convolve(sol_noGABA.y[4, :], window_conv, mode="same")
    V_shell_noGABA = np.convolve(sol_noGABA.y[10, :], window_conv, mode="same")
    V_core_GABA = np.convolve(sol_GABA.y[4, :], window_conv, mode="same")
    V_shell_GABA = np.convolve(sol_GABA.y[10, :], window_conv, mode="same")

    # 時間軸の表示範囲
    t_range_GABA = (0, 250)
    t_range_noGABA = (250, 550)
    t_range_pre = (250, 300)
    t_range_plt = (0, 300)
    t_mask_noGABA = (t_eval >= t_range_noGABA[0]) & (t_eval <= t_range_noGABA[1])
    t_mask_GABA = (t_eval >= t_range_GABA[0]) & (t_eval <= t_range_GABA[1])
    t_mask_pre = (t_eval >= t_range_pre[0]) & (t_eval <= t_range_pre[1])
    t_mask_plt = (t_eval >= t_range_plt[0]) & (t_eval <= t_range_plt[1])
    t_eval_plt = t_eval[t_mask_plt]
    V_core_GABA_plt = np.hstack(
        (V_core_noGABA[t_mask_pre][:-1], V_core_GABA[t_mask_GABA])
    )
    V_core_noGABA_plt = V_core_noGABA[t_mask_noGABA]
    V_shell_GABA_plt = np.hstack(
        (V_shell_noGABA[t_mask_pre][:-1], V_shell_GABA[t_mask_GABA])
    )
    V_shell_noGABA_plt = V_shell_noGABA[t_mask_noGABA]

    fig = plt.figure(figsize=(14, 6))

    peak_index_core_GABA, _ = find_peaks(V_core_GABA, prominence=4)
    peak_interval_core_GABA = np.diff(t_eval[peak_index_core_GABA])
    peak_index_core_noGABA, _ = find_peaks(V_core_noGABA)
    peak_interval_core_noGABA = np.diff(t_eval[peak_index_core_noGABA])

    period_GABA = round(np.mean(peak_interval_core_GABA[-13:-3]), 2)
    period_noGABA = round(np.mean(peak_interval_core_noGABA[-13:-3]), 2)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t_eval_plt, V_core_noGABA_plt, c="#FF0000", label="Core")
    ax1.plot(t_eval_plt, V_shell_noGABA_plt, c="#0070C0", linestyle="--", label="Shell")
    ax1.set_ylabel(r"$V$", fontsize=30)
    ax1.axvline(50, c="k")
    ax1.set_title("GABA -", fontsize=30)
    ax1.text(
        0.05,
        1.02,
        "GABA -",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )
    ax1.text(
        0.02,
        0.8,
        rf"${period_noGABA}~h$",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )
    ax1.text(
        0.8,
        0.8,
        rf"${period_noGABA}~h$",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )

    ax1.set_ylim((0, 8.5))

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t_eval_plt, V_core_GABA_plt, c="#FF0000")
    ax2.plot(t_eval_plt, V_shell_GABA_plt, c="#0070C0", linestyle="--")
    ax2.set_ylabel(r"$V$", fontsize=30)
    ax2.set_xlabel("Time(h)", fontsize=20)
    ax2.text(
        0.44,
        0.8,
        "GABA +",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )
    ax2.text(
        0.02,
        0.8,
        rf"${period_noGABA}~h$",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )
    ax2.text(
        0.8,
        0.8,
        rf"${period_GABA}~h$",
        transform=ax2.transAxes,
        ha="left",
        va="bottom",
        fontsize=30,
    )
    ax2.axvline(50, c="k")
    ax2.set_ylim((0, 8.5))

    for ax, label in zip([ax1, ax2], caption):
        ax.text(
            -0.15,
            1.25,
            label,
            transform=ax.transAxes,
            fontsize=50,
            fontweight="bold",
            va="top",
            ha="left",
        )

    fig.legend(loc="upper right", fontsize=20, handlelength=4)
    plt.savefig(f"./results/Figure3_10_{caption[0]}_{caption[1]}.png")
    plt.show()

# %%
