# Figure3.5の作成プログラム

# %%

from utils import cellmodel  # 自作関数の読み込み
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import os

np.random.seed(222)
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "lines.linewidth": 5,  # 線の太さ
    }
)
# %%
os.makedirs("results", exist_ok=True)
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

t_max = 2000
t_span = [0, t_max]
t_eval = np.arange(0, t_max, 0.001)

# %%
init1 = np.random.rand(5) * np.array([1.25, 40, 30, 700, 15])

# 光なし（定常状態の自己発振）
model = cellmodel.CellModel(LD_params={}, cell_params=cell_params)
sol = solve_ivp(
    lambda t, y: model.Onecell_model(t, y),
    (0, t_max),
    y0=init1,
    t_eval=t_eval,
    method="LSODA",
)
M_trace = sol.y[0]
V_trace = sol.y[4]
Ca_trace = sol.y[3]
peaks, _ = find_peaks(V_trace, prominence=0.5)
peaks_Ca, _ = find_peaks(Ca_trace, prominence=5)
t_peaks = sol.t[peaks]
t_peaks_Ca = sol.t[peaks_Ca]

# 周期の推定（最後のいくつかのピークを使う）
period = np.mean(np.diff(t_peaks[-10:-2]))
print(f"推定周期: {period:.2f} 時間")

# ピークを90°に対応させるために、最初のピーク時間を取得
t1 = t_peaks[-10]  # 最初のピークを位相0に対応させるため基準に
t1_Ca = t_peaks_Ca[-10]

# ------------------------------
# 光パルス条件の準備
# ------------------------------
pulse_width = 1
pulse_intensity = 5
base_pulse_param = {
    "L_max": pulse_intensity,
    "pulse_width": pulse_width,
    "pulse_start": None,
}

# ------------------------------
# PRCの計算（位相を90度基準に変換）
# ------------------------------
PRC_phases = []
PRC_shifts = []

# 波形保存用（deg=200°と280°付近）
highlight_phases = [330.6, 210.6]
highlight_traces = {}
highlight_times = {}
highlight_phaseshift = {}

for shift_hour in np.linspace(0, period - 1, 24):
    pulse_time = 1100 + shift_hour  # パルス開始時間

    # 相対位相計算（ピークt1を90°に対応させる）
    relative_phase = ((pulse_time - t1) % period) / period * 360
    shifted_phase = (relative_phase + 90) % 360
    PRC_phases.append(shifted_phase)

    # 光パルスありのモデル作成
    pulse_param = {**base_pulse_param, "pulse_start": pulse_time}
    model = cellmodel.CellModel(
        LD_params={}, cell_params=cell_params, pulse_param=pulse_param
    )

    sol_pulse = solve_ivp(
        lambda t, y: model.Onecell_model(t, y, light_pulse=True),
        (0, t_max),
        y0=init1,
        t_eval=t_eval,
        method="LSODA",
    )

    V_trace_pulse = sol_pulse.y[4]
    peaks_pulse, _ = find_peaks(V_trace_pulse, height=0.5)
    t_peaks_pulse = sol_pulse.t[peaks_pulse]

    # 基準と刺激ありの最後のピークの比較
    ref_peak = t_peaks[-1]
    test_peak = t_peaks_pulse[-1]

    dphi = ((test_peak - ref_peak) / period) * (-1) * 360  # 位相シフト [deg]
    PRC_shifts.append(dphi)

    # 特定位相付近の波形保存
    for hphase in highlight_phases:
        if abs(shifted_phase - hphase) < 0.5:
            highlight_traces[hphase] = V_trace_pulse
            highlight_times[hphase] = sol_pulse.t
            highlight_phaseshift[hphase] = (
                ((test_peak - ref_peak) / period) * (-1) * 360
            )


# ------------------------------
# PRCのプロット
# ------------------------------
fig = plt.figure(figsize=(10, 10))

ax1 = fig.add_subplot(312)
plot_min = t1 - period / 4
plot_max = t1 + period * 3 / 4
plot_onecycle_mask = (t_eval > plot_min) & (t_eval < plot_max)
plot_0 = plot_min
plot_90 = t1
plot_180 = t1 + period / 4
plot_270 = t1 + period / 2
plot_360 = plot_max

ax1.plot(t_eval[plot_onecycle_mask], M_trace[plot_onecycle_mask], c="r")
ax1.set_xticks([plot_0, plot_90, plot_180, plot_270, plot_360])
ax1.set_xticklabels(["0", "90", "180", "270", "360"])
ax1.set_xlabel(r"$V$" + "  " + "Phase (degree)", fontsize=20)
ax1.set_ylabel(r"$M$", fontsize=20)
ax1.set_ylim((0, 1))
ax1_1 = ax1.twinx()
ax1_1.plot(t_eval[plot_onecycle_mask], V_trace[plot_onecycle_mask], c="k")
ax1_1.set_yticks([0, 5, 10])
ax1_1.set_ylabel(r"$V$", fontsize=20, rotation=270, labelpad=20)

ax2 = fig.add_subplot(311)
ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax2.scatter(PRC_phases, PRC_shifts, s=50, c="k")
colors = ["#E97132", "#156082"]
for i, hphase in enumerate(highlight_phases):
    if hphase in highlight_traces:
        ax2.scatter(
            hphase,
            highlight_phaseshift[hphase],
            color=colors[i],
            s=200,
        )
ax2.set_xticks([0, 90, 180, 270, 360])
ax2.set_yticks([-100, -50, 0, 50, 100])
ax2.set_xlabel(r"$V$" + "  " + "Phase (degree)", fontsize=20)
ax2.set_ylabel("Phase shift (degree)", fontsize=20)
# ------------------------------
# 波形比較プロット（基準波形とdeg=200°, 280°の刺激波形）
# ------------------------------
ax3 = fig.add_subplot(313)
ax3.plot(sol.t, V_trace, label="No Pulse", color="black")
colors = ["#E97132", "#156082"]
for i, hphase in enumerate(highlight_phases):
    if hphase in highlight_traces:
        ax3.plot(
            highlight_times[hphase],
            highlight_traces[hphase],
            color=colors[i],
            alpha=0.7,
        )
ax3.set_xticks([1090, 1114, 1138, 1162, 1188, 1212])
ax3.set_yticks([0, 5, 10])
ax3.set_xlim((1080, 1220))
ax3.set_xlabel("Time (h)", fontsize=20)
ax3.set_ylabel(r"$V$", fontsize=20)
panel_labels = [f"({chr(97 + i)})" for i in range(3)]

for ax, label in zip([ax2, ax1, ax3], panel_labels):
    ax.text(
        -0.2,
        1.25,
        label,
        transform=ax.transAxes,
        fontsize=30,
        fontweight="bold",
        va="top",
        ha="left",
    )
plt.tight_layout()
fig.savefig("./results/Figure3_5.png", dpi=300)
plt.show()


# %%
