# GABAありとなしで比較
# %%
# %%
from utils_fig39 import cellmodel  # 自作関数の読み込み
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

# %%
# (a)(b)はseed=222で(c)(d)はseed=221
seed_ab = 222  # (a)(b)
seed_cd = 221  # (c)(d)
# %%


def draw_light_background(
    ax, t_eval, light_wave, threshold=0.1, color="yellow", alpha=0.2
):
    """light_waveがthreshold以上の区間に背景を描く"""
    in_light = False
    for i in range(1, len(t_eval)):
        if light_wave[i - 1] < threshold and light_wave[i] >= threshold:
            start = t_eval[i]
            in_light = True
        elif light_wave[i - 1] >= threshold and light_wave[i] < threshold and in_light:
            end = t_eval[i]
            ax.axvspan(start, end, color=color, alpha=alpha)
            in_light = False
    # 光が終わらずに最後まで続いた場合
    if in_light:
        ax.axvspan(start, t_eval[-1], color=color, alpha=alpha)


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
}  # LD = 12

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


n_conv = 100
window_conv = np.ones(n_conv) / n_conv

V_core_noGABA_ab = np.convolve(sol_noGABA_ab.y[4, :], window_conv, mode="same")
V_shell_noGABA_ab = np.convolve(sol_noGABA_ab.y[10, :], window_conv, mode="same")
V_core_GABA_ab = np.convolve(sol_GABA_ab.y[4, :], window_conv, mode="same")
V_shell_GABA_ab = np.convolve(sol_GABA_ab.y[10, :], window_conv, mode="same")
V_core_noGABA_cd = np.convolve(sol_noGABA_cd.y[4, :], window_conv, mode="same")
V_shell_noGABA_cd = np.convolve(sol_noGABA_cd.y[10, :], window_conv, mode="same")
V_core_GABA_cd = np.convolve(sol_GABA_cd.y[4, :], window_conv, mode="same")
V_shell_GABA_cd = np.convolve(sol_GABA_cd.y[10, :], window_conv, mode="same")

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
V_core_GABA_plt_ab = np.hstack(
    (V_core_noGABA_ab[t_mask_pre][:-1], V_core_GABA_ab[t_mask_GABA])
)
V_core_noGABA_plt_ab = V_core_noGABA_ab[t_mask_noGABA]
V_shell_GABA_plt_ab = np.hstack(
    (V_shell_noGABA_ab[t_mask_pre][:-1], V_shell_GABA_ab[t_mask_GABA])
)
V_shell_noGABA_plt_ab = V_shell_noGABA_ab[t_mask_noGABA]

V_core_GABA_plt_cd = np.hstack(
    (V_core_noGABA_cd[t_mask_pre][:-1], V_core_GABA_cd[t_mask_GABA])
)
V_core_noGABA_plt_cd = V_core_noGABA_cd[t_mask_noGABA]
V_shell_GABA_plt_cd = np.hstack(
    (V_shell_noGABA_cd[t_mask_pre][:-1], V_shell_GABA_cd[t_mask_GABA])
)
V_shell_noGABA_plt_cd = V_shell_noGABA_cd[t_mask_noGABA]

fig_ab = plt.figure(figsize=(14, 6))
fig_cd = plt.figure(figsize=(14, 6))

peak_index_core_GABA_ab, _ = find_peaks(V_core_GABA_ab, prominence=4)
peak_interval_core_GABA_ab = np.diff(t_eval[peak_index_core_GABA_ab])
peak_index_core_noGABA_ab, _ = find_peaks(V_core_noGABA_ab)
peak_interval_core_noGABA_ab = np.diff(t_eval[peak_index_core_noGABA_ab])
print(
    np.mean(peak_interval_core_GABA_ab[-13:-3]),
    np.std(peak_interval_core_GABA_ab[-13:-3]),
)
print(
    np.mean(peak_interval_core_noGABA_ab[-13:-3]),
    np.std(peak_interval_core_noGABA_ab[-13:-3]),
)
period_GABA = round(np.mean(peak_interval_core_GABA_ab[-13:-3]), 2)
period_noGABA = round(np.mean(peak_interval_core_noGABA_ab[-13:-3]), 2)

peak_index_core_GABA_cd, _ = find_peaks(V_core_GABA_cd, prominence=4)
peak_interval_core_GABA_cd = np.diff(t_eval[peak_index_core_GABA_cd])
peak_index_core_noGABA_cd, _ = find_peaks(V_core_noGABA_cd)
peak_interval_core_noGABA_cd = np.diff(t_eval[peak_index_core_noGABA_cd])
print(
    np.mean(peak_interval_core_GABA_cd[-13:-3]),
    np.std(peak_interval_core_GABA_cd[-13:-3]),
)
print(
    np.mean(peak_interval_core_noGABA_cd[-13:-3]),
    np.std(peak_interval_core_noGABA_cd[-13:-3]),
)
period_GABA = round(np.mean(peak_interval_core_GABA_cd[-13:-3]), 2)
period_noGABA = round(np.mean(peak_interval_core_noGABA_cd[-13:-3]), 2)

ax1_ab = fig_ab.add_subplot(2, 1, 1)
ax1_cd = fig_cd.add_subplot(2, 1, 1)
V_core_noGABA_plt_list = [V_core_GABA_plt_ab, V_core_GABA_plt_cd]
V_shell_noGABA_plt_list = [V_shell_GABA_plt_ab, V_shell_GABA_plt_cd]
for i, ax1 in enumerate([ax1_ab, ax1_cd]):
    ax1.plot(t_eval_plt, V_core_noGABA_plt_list[i], c="#FF0000", label="Core")
    ax1.plot(
        t_eval_plt,
        V_shell_noGABA_plt_list[i],
        c="#0070C0",
        linestyle="--",
        label="Shell",
    )
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

ax2_ab = fig_ab.add_subplot(2, 1, 2)
ax2_cd = fig_cd.add_subplot(2, 1, 2)
for ax2, V_core_GABA_plt, V_shell_GABA_plt in zip(
    [ax2_ab, ax2_cd],
    [V_core_GABA_plt_ab, V_core_GABA_plt_cd],
    [V_shell_GABA_plt_ab, V_shell_GABA_plt_cd],
):
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

panel_labels = [f"({chr(97 + i)})" for i in range(4)]

for ax, label in zip([ax1_ab, ax2_ab, ax1_cd, ax2_cd], panel_labels):
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

fig_ab.legend(loc="upper right", fontsize=20, handlelength=4)
fig_cd.legend(loc="upper right", fontsize=20, handlelength=4)

os.makedirs("results", exist_ok=True)
fig_ab.savefig("./results/Figure3_10(a)(b).png")
fig_cd.savefig("./results/Figure3_10(c)(d).png")
plt.show()

# %%

# %%


# %%
seed = 221
np.random.seed(seed)
GABA_start_f = 300
GABA_start_h = 314
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

t_max = 4000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.01)
# %%
init1 = np.random.rand(11) * np.array([1.25, 40, 30, 700, 15, 1, 1.25, 40, 30, 700, 15])
LD_params_0 = {
    "LD_time": [0, t_max],
    "L_phase": 12,
    "D_phase": 12,
    "L_max": 0,
    "L_min": 0,
    "k_light": 12,
    "k_dark": 12,
}  # LD = 12
con_param_VIP = {
    "b_core": 50,  # VIP
    "b_shell": 50.0,
    "gsa": 0,
}

con_param_VIPGABA = {
    "b_core": 50,  # noVIP
    "b_shell": 50.0,
    "gsa": 0.1,
}

model_VIP = cellmodel.CellModel(LD_params_0, cell_params, connect_param=con_param_VIP)
model_VIPGABA = cellmodel.CellModel(
    LD_params_0, cell_params, connect_param=con_param_VIPGABA
)

sol_VIP = solve_ivp(
    fun=lambda t, y: model_VIP.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init1,
    t_eval=t_eval,
    method="LSODA",
)  # no light VIP
init_VIPGABA_f = sol_VIP.y[:, GABA_start_f * 100]
init_VIPGABA_h = sol_VIP.y[:, GABA_start_h * 100]

sol_VIPGABA_f = solve_ivp(
    fun=lambda t, y: model_VIPGABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_VIPGABA_f,
    t_eval=t_eval,
    method="LSODA",
)
sol_VIPGABA_h = solve_ivp(
    fun=lambda t, y: model_VIPGABA.Twocell_model(t, y, LD=True),
    t_span=t_span,
    y0=init_VIPGABA_h,
    t_eval=t_eval,
    method="LSODA",
)


n_conv = 100
window_conv = np.ones(n_conv) / n_conv

V_core_VIP = np.convolve(sol_VIP.y[4, :], window_conv, mode="same")
V_shell_VIP = np.convolve(sol_VIP.y[10, :], window_conv, mode="same")
V_core_VIPGABA_f = np.convolve(sol_VIPGABA_f.y[4, :], window_conv, mode="same")
V_shell_VIPGABA_f = np.convolve(sol_VIPGABA_f.y[10, :], window_conv, mode="same")
V_core_VIPGABA_h = np.convolve(sol_VIPGABA_h.y[4, :], window_conv, mode="same")
V_shell_VIPGABA_h = np.convolve(sol_VIPGABA_h.y[10, :], window_conv, mode="same")

# 時間軸の表示範囲
t_range_VIPGABA_f = (0, 250 - GABA_start_f + 300)
t_range_VIPGABA_h = (0, 250 - GABA_start_h + 300)
t_range_VIP = (250, 550)
t_range_pre_f = (250, GABA_start_f)
t_range_pre_h = (250, GABA_start_h)
t_range_plt = (0, 300)
t_mask_VIPGABA_f = (t_eval >= t_range_VIPGABA_f[0]) & (t_eval <= t_range_VIPGABA_f[1])
t_mask_VIPGABA_h = (t_eval >= t_range_VIPGABA_h[0]) & (t_eval <= t_range_VIPGABA_h[1])
t_mask_VIP = (t_eval >= t_range_VIP[0]) & (t_eval <= t_range_VIP[1])
t_mask_pre_f = (t_eval >= t_range_pre_f[0]) & (t_eval <= t_range_pre_f[1])
t_mask_pre_h = (t_eval >= t_range_pre_h[0]) & (t_eval <= t_range_pre_h[1])
t_mask_plt = (t_eval >= t_range_plt[0]) & (t_eval <= t_range_plt[1])
t_eval_plt = t_eval[t_mask_plt]
V_core_VIPGABA_plt_f = np.hstack(
    (V_core_VIP[t_mask_pre_f][:-1], V_core_VIPGABA_f[t_mask_VIPGABA_f])
)
V_core_VIPGABA_plt_h = np.hstack(
    (V_core_VIP[t_mask_pre_h][:-1], V_core_VIPGABA_h[t_mask_VIPGABA_h])
)
V_core_VIP_plt = V_core_VIP[t_mask_VIP]
V_shell_VIPGABA_plt_f = np.hstack(
    (V_shell_VIP[t_mask_pre_f][:-1], V_shell_VIPGABA_f[t_mask_VIPGABA_f])
)
V_shell_VIPGABA_plt_h = np.hstack(
    (V_shell_VIP[t_mask_pre_h][:-1], V_shell_VIPGABA_h[t_mask_VIPGABA_h])
)
V_shell_VIP_plt = V_shell_VIP[t_mask_VIP]

fig = plt.figure(figsize=(14, 9))
peak_index_core_VIP, _ = find_peaks(V_core_VIP)
peak_interval_core_VIP = np.diff(t_eval[peak_index_core_VIP])
peak_index_core_VIPGABA_f, _ = find_peaks(V_core_VIPGABA_f)
peak_index_core_VIPGABA_h, _ = find_peaks(V_core_VIPGABA_h)
peak_interval_core_VIPGABA_f = np.diff(t_eval[peak_index_core_VIPGABA_f])
peak_interval_core_VIPGABA_h = np.diff(t_eval[peak_index_core_VIPGABA_h])


period_VIP = round(np.mean(peak_interval_core_VIP[-13:-3]), 2)
period_VIPGABA_f = round(np.mean(peak_interval_core_VIPGABA_f[-13:-3]), 2)
period_VIPGABA_h = round(np.mean(peak_interval_core_VIPGABA_h[-13:-3]), 2)

ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(t_eval_plt, V_core_VIP_plt, c="#FF0000", label="Core")
ax1.plot(t_eval_plt, V_shell_VIP_plt, c="#0070C0", linestyle="--", label="Shell")
ax1.set_ylabel(r"$V$", fontsize=30)
ax1.axvline(GABA_start_f - 250, c="k", linestyle="-")
ax1.axvline(GABA_start_h - 250, c="k", linestyle="--")
ax1.text(
    0.44,
    0.8,
    "VIP + GABA -",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax1.text(
    0,
    1.02,
    "VIP + GABA -",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax1.text(
    0.02,
    0.8,
    rf"${period_VIP}~h$",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax1.text(
    0.8,
    0.8,
    rf"${period_VIP}~h$",
    transform=ax1.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

ax1.set_ylim((0, 8.5))

ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(t_eval_plt, V_core_VIPGABA_plt_f, c="#FF0000")
ax2.plot(t_eval_plt, V_shell_VIPGABA_plt_f, c="#0070C0", linestyle="--")
ax2.set_ylabel(r"$V$", fontsize=30)
ax2.text(
    0.44,
    0.8,
    "VIP + GABA +",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

ax2.axvline(GABA_start_f - 250, c="k", linestyle="-")
ax2.set_ylim((0, 8.5))

ax2.text(
    0.02,
    0.8,
    rf"${period_VIP}~h$",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax2.text(
    0.8,
    0.8,
    rf"${period_VIPGABA_f}~h$",
    transform=ax2.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(t_eval_plt, V_core_VIPGABA_plt_h, c="#FF0000")
ax3.plot(t_eval_plt, V_shell_VIPGABA_plt_h, c="#0070C0", linestyle="--")
ax3.set_ylabel(r"$V$", fontsize=30)
ax3.set_xlabel("Time(h)", fontsize=20)
ax3.text(
    0.44,
    0.8,
    "VIP + GABA +",
    transform=ax3.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax3.axvline(GABA_start_h - 250, c="k", linestyle="--")
ax3.set_ylim((0, 8.5))

ax3.text(
    0.02,
    0.8,
    rf"${period_VIP}~h$",
    transform=ax3.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)
ax3.text(
    0.8,
    0.8,
    rf"${period_VIPGABA_h}~h$",
    transform=ax3.transAxes,
    ha="left",
    va="bottom",
    fontsize=30,
)

panel_labels = [f"({chr(101 + i)})" for i in range(3)]  # (e)(f)(g)

for ax, label in zip([ax1, ax2, ax3], panel_labels):
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
os.makedirs("results", exist_ok=True)
fig.savefig("./results/Figure3_10(e)(f)(g).png")
plt.show()

# %%
