# %%
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from utils import neuron_model
from matplotlib.patches import Patch
from matplotlib.path import Path
import matplotlib.patches as mpatches

# --- スタイル設定 [5, 6] ---
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
    }
)

light_strangth = "0.2"


# 背景描画関数 [5, 7, 8]
def draw_daytime_background(
    ax, t_eval, LD_time, L_phase, D_phase, color="yellow", alpha=0.2
):
    period = L_phase + D_phase
    t_start = float(max(LD_time[0], t_eval[0]))
    t_end = float(min(LD_time[1], t_eval[-1]))
    day_start = D_phase / 2.0
    if day_start < t_start:
        k = np.floor((t_start - day_start) / period)
        day_start += k * period
    while day_start < t_end:
        day_end = day_start + L_phase
        s, e = max(day_start, t_start), min(day_end, t_end)
        if s < e:
            ax.axvspan(s, e, color=color, alpha=alpha)
        day_start += period


def draw_light_darktime_background(
    ax,
    t_eval,
    LD_time,
    L_phase,
    D_phase,
    light_color="yellow",
    dark_color="gray",
    alpha=0.2,
):
    period = L_phase + D_phase
    t_start = float(max(LD_time[0], t_eval[0]))
    t_end = float(min(LD_time[1], t_eval[-1]))
    light_start = D_phase / 2.0
    dark_start = D_phase / 2.0 + L_phase
    ax.axvspan(0, D_phase / 2.0, color=dark_color, alpha=alpha)
    if light_start < t_start:
        k = np.floor((t_start - light_start) / period)
        light_start += k * period
        dark_start += k * period
    while light_start < t_end:
        light_end = light_start + L_phase
        s, e = max(light_start, t_start), min(light_end, t_end)
        if s < e:
            ax.axvspan(s, e, color=light_color, alpha=alpha)
        light_start += period
    while dark_start < t_end:
        dark_end = dark_start + D_phase
        s, e = max(dark_start, t_start), min(dark_end, light_end)
        if s < e:
            ax.axvspan(s, e, color=dark_color, alpha=alpha)
        dark_start = dark_end + L_phase
    ax.axvspan(light_end, t_eval[-1], color=dark_color, alpha=alpha)


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


def evaluate_period(df, threshold):
    """閾値法で周期を判定する"""
    period_data = dict()
    for col in [c for c in df.columns if c.startswith("V")]:
        t = df["t"].values
        V = df[col].values

        crossings = detect_threshold_crossings(t, V, threshold)
        if len(crossings) < 5:
            period_data[col] = [False, 0]
            continue

        # 各交差間隔の平均を取る（＝推定周期）
        intervals = np.diff(crossings)
        mean_period = np.mean(intervals[-13:-3])
        std_period = np.std(intervals[-13:-3])

        # 周期が光周期に近ければ同調とみなす
        period_data[col] = [std_period < 0.1, mean_period]
    return period_data


def plot_aftereffect_shift_layout():
    vip_core, vip_shell = 50, 50
    vip_condition = f"{vip_core}_{vip_shell}"
    hist_gabasc = ["0.02", "0.075", "0.09", "0.1"]

    fig = plt.figure(figsize=(12, 17))

    fig2 = plt.figure(figsize=(12, 12))
    fig3 = plt.figure(figsize=(12, 12))

    gs = GridSpec(
        4,
        4,
        figure=fig,
        width_ratios=[0.6, 0.6, 0.5, 0.5],
        height_ratios=[1.6, 0.55, 0.55, 0.55],
        wspace=0.5,
        hspace=0.8,
    )

    # --- 1. 左上: 履歴効果散布図 (a) [9, 10] ---
    ax_main = fig.add_subplot(gs[0, :2])

    x, y = [], []
    x3, y3 = [], []

    for gaba_sc in [str(round(g / 200, 3)) for g in range(40)]:
        aftereffect_result = pd.read_csv(
            os.path.join(
                "VIP_GABA_5&5cell_aftereffect",
                light_strangth,
                vip_condition,
                f"evaluate_period_summary_{gaba_sc}.csv",
            )
        )

        entrain_result = pd.read_csv(
            os.path.join(
                "VIP_GABA_5&5cell_entrain",
                light_strangth,
                vip_condition,
                f"evaluate_period_summary_{gaba_sc}.csv",
            )
        )

        rand_result = pd.read_csv(
            os.path.join(
                "./VIP_GABA_5_5cell_randint",
                vip_condition,
                f"evaluate_period_summary_{gaba_sc}.csv",
            )
        )

        for L_phase in set(aftereffect_result["L_phase"]):
            df_period = aftereffect_result[aftereffect_result["L_phase"] == L_phase]
            df_entrain = entrain_result[entrain_result["L_phase"] == L_phase]

            if np.prod(df_entrain["is_sync"]) == 1:
                if np.prod(df_period["is_sync"]) == 1:
                    x.append(df_period["period"].iloc[0])
                    y.append(float(gaba_sc))

        for rand_seed in set(rand_result["rand_seed"]):
            df = rand_result[rand_result["rand_seed"] == rand_seed]

            if np.prod(df["is_sync"]) == 1:
                y3.append(df["gaba_sc"].iloc[0])
                x3.append(df["period"].iloc[0])
    ax_main.scatter(x3, y3, c="k", s=10, alpha=0.1)
    ax_main.scatter(x, y, c="r", s=10)

    ax_main.set_xlabel("Period (h)")
    ax_main.set_ylabel(r"$g_{SC}$")
    ax_main.set_xlim(20, 28)
    ax_main.set_ylim(0, 0.21)
    ax_main.text(
        0.02, 0.9, "(a)", transform=ax_main.transAxes, fontsize=24, fontweight="bold"
    )

    panel_labels = ["(b)", "(c)", "(d)", "(e)"]
    for idx, gabasc in enumerate(hist_gabasc):
        y_val = float(gabasc)
        # 水平線を引く
        ax_main.axhline(
            y=y_val, color="black", linestyle="--", linewidth=1.5, alpha=0.7
        )
        # 線の右端にラベルを表示
        ax_main.text(
            28.1,
            y_val,
            panel_labels[idx],
            color="black",
            va="center",
            fontsize=20,
            fontweight="bold",
        )

    # --- 2. 右上: ヒストグラム (b, c, d, e) ---
    hist_gs = gs[0, 2:].subgridspec(2, 2, wspace=0.3, hspace=0.4)
    labels_hist = ["(b)", "(c)", "(d)", "(e)"]
    for i, gabasc in enumerate(hist_gabasc):
        ax_h = fig.add_subplot(hist_gs[i // 2, i % 2])
        y_val = float(gabasc)
        path = os.path.join(
            "VIP_GABA_5&5cell_aftereffect",
            light_strangth,
            vip_condition,
            f"evaluate_period_summary_{gabasc}.csv",
        )
        entrain_path = os.path.join(
            "VIP_GABA_5&5cell_entrain",
            light_strangth,
            vip_condition,
            f"evaluate_period_summary_{gabasc}.csv",
        )
        if os.path.exists(path):
            res = pd.read_csv(path)
            df_entrain = pd.read_csv(entrain_path)

            x_below_12 = []
            x_above_12 = []
            for rs in set(res["L_phase"]):
                # 同期判定の条件
                is_sync_res = np.prod(res[res["L_phase"] == rs]["is_sync"]) == 1
                is_sync_ent = (
                    np.prod(df_entrain[df_entrain["L_phase"] == rs]["is_sync"]) == 1
                )

                if is_sync_res and is_sync_ent:
                    period_val = res[res["L_phase"] == rs]["period"].iloc[0]
                    if float(rs) < 12.05:
                        x_below_12.append(period_val)
                    else:
                        x_above_12.append(period_val)
            ax_h.hist(
                [x_below_12, x_above_12],
                bins=np.arange(22, 26.1, 0.1).tolist(),
                color=[
                    "#55A868",
                    "#DD8452",
                ],  # L_phase < 12 を skyblue、それ以外を gray
                edgecolor="black",
                stacked=True,
            )
            ax_h.text(
                0.02,
                0.85,
                labels_hist[i],
                transform=ax_h.transAxes,
                fontsize=24,
                fontweight="bold",
            )
            ax_h.set_xlim(22, 26)
            ax_h.set_ylim((0, 22))
            ax_h.set_title(r"$g_{{SC}}=$" + f"{y_val}", fontsize=15)
            if i >= 2:
                ax_h.set_xlabel("Period (h)", fontsize=15)

    hist_handles = [
        Patch(
            facecolor="#55A868", edgecolor="black", label="T cycle" + r"$\leqq$" + "24"
        ),
        Patch(facecolor="#DD8452", edgecolor="black", label="T cycle > 24"),
    ]
    leg1 = fig.legend(
        handles=hist_handles,
        loc="upper right",
        fontsize=15,
        bbox_to_anchor=(0.96, 0.96),
    )
    fig.add_artist(leg1)
    # --- 3. 下段: LDから暗状態へのシフト (f, g, h) [2-4, 13] ---
    with open("phase_initial_conditions.json", "r") as f:
        phase_init_data = json.load(f)

    # パラメータ設定 [14, 15]
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

    # シフトを比較するための異なるL_phase設定 [13]
    l_phases = [11.6, 12.0, 13.0]
    l_phases_2 = [11.7, 12.0, 12.3]
    labels_wave = ["(f)", "(g)", "(h)"]
    g_sc_sample = 0.18
    t_max = 4000  # シフトを見るために2000h前後を描画
    ld_end_time = 2000  # 2000時間で光を止める [1]
    wave_gs = gs[1:, :].subgridspec(
        3,
        1,
        hspace=0.3,  # ← wave 間を詰める
    )
    for idx, lp in enumerate(l_phases):
        ax_wave = fig.add_subplot(wave_gs[idx, 0])
        ax_wave_2 = fig2.add_subplot(3, 1, idx + 1)
        ax_wave_3 = fig3.add_subplot(3, 1, idx + 1)

        L_phase = round(lp, 2)
        D_phase = L_phase  # T-cycle = 2 * L_phase

        L_phase_2 = round(l_phases_2[idx], 2)
        D_phase_2 = L_phase_2  # T-cycle = 2 * L_phase
        ax_wave.text(
            0.5,
            1.03,
            r"$g_{SC}=0.09$",
            transform=ax_wave.transAxes,
            fontsize=20,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
        # ネットワーク構築
        cells_num = 10
        cell_types = ["core"] * 5 + ["shell"] * 5
        vip_weights = [vip_core if t == "core" else vip_shell for t in cell_types]
        gaba_mat = np.zeros((cells_num, cells_num))
        for r in range(10):
            for e in range(10):
                if cell_types[e] == "shell" and cell_types[r] == "core":
                    gaba_mat[r][e] = g_sc_sample / cells_num

        # 初期値 (Seedは固定)
        rng = np.random.default_rng(110)
        y0 = []
        for p in [rng.integers(360) for _ in range(10)]:
            y0.extend(
                [
                    phase_init_data[str(p)][k]
                    for k in ["M", "Pc", "Pn", "Ca", "V", "VIP"]
                ]
            )

        cells = [
            neuron_model.SingleCell(cell_params, i, cell_type=cell_types[i])
            for i in range(10)
        ]
        net = neuron_model.CellNetwork(cells, vip_weights, gaba_mat)
        net_3 = neuron_model.CellNetwork(
            cells, vip_weights, np.zeros((cells_num, cells_num))
        )

        # 光刺激設定: 0hからld_end_timeまでLDサイクル、以降は暗状態 [2, 3]
        light_args = (
            L_phase,
            D_phase,
            0.2,
            0,
            L_phase,
            D_phase,
            0,
            ld_end_time // (L_phase + D_phase) * (L_phase + D_phase),
        )

        light_args_2 = (
            L_phase_2,
            D_phase_2,
            0.2,
            0,
            L_phase_2,
            D_phase_2,
            0,
            ld_end_time // (L_phase_2 + D_phase_2) * (L_phase_2 + D_phase_2),
        )

        sol = solve_ivp(
            fun=lambda t, y: net.ode_all_neurons(t, y, light_func_args=light_args),
            t_span=[0, t_max],
            t_eval=np.arange(0, t_max, 0.01),
            y0=y0,
            method="LSODA",
        )

        sol_2 = solve_ivp(
            fun=lambda t, y: net.ode_all_neurons(t, y, light_func_args=light_args_2),
            t_span=[0, t_max],
            t_eval=np.arange(0, t_max, 0.01),
            y0=y0,
            method="LSODA",
        )

        sol_3 = solve_ivp(
            fun=lambda t, y: net_3.ode_all_neurons(t, y, light_func_args=light_args_2),
            t_span=[0, t_max],
            t_eval=np.arange(0, t_max, 0.01),
            y0=y0,
            method="LSODA",
        )

        # 波形描画 [4]
        core_plot, shell_plot = False, False
        for i in range(10):
            col = "r" if cell_types[i] == "core" else "b"
            ls = "-" if cell_types[i] == "core" else "--"
            if idx == 0:
                if not core_plot and cell_types[i] == "core":
                    label = "Core"
                    core_plot = True
                else:
                    if not shell_plot and cell_types[i] == "shell":
                        label = "Shell"
                        shell_plot = True
                    else:
                        label = None

            t1_start, t1_end = 1900, 2100
            t2_start, t2_end = 3900, 4000

            mask1 = (sol.t >= t1_start) & (sol.t <= t1_end)
            mask2 = (sol.t >= t2_start) & (sol.t <= t2_end)
            t_concat = np.concatenate(
                [sol.t[mask1], sol.t[mask2] - (t2_start - t1_end)]
            )

            V_concat = np.concatenate(
                [sol.y[i * 6 + 4, mask1], sol.y[i * 6 + 4, mask2]]
            )
            V_concat_2 = np.concatenate(
                [sol_2.y[i * 6 + 4, mask1], sol_2.y[i * 6 + 4, mask2]]
            )
            V_concat_3 = np.concatenate(
                [sol_3.y[i * 6 + 4, mask1], sol_3.y[i * 6 + 4, mask2]]
            )

            ax_wave.plot(
                t_concat, V_concat, c=col, linestyle=ls, alpha=0.7, label=label
            )
            ax_wave_2.plot(
                t_concat, V_concat_2, c=col, linestyle=ls, alpha=0.7, label=label
            )
            ax_wave_3.plot(
                t_concat, V_concat_3, c=col, linestyle=ls, alpha=0.7, label=label
            )
            xticks = [
                t1_start,
                t1_start + 50,
                t1_start + 100,
                t1_start + 150,
                t1_end,
                t1_end + 50,
                t1_end + 100,
            ]

            xticks2 = [
                t1_start,
                t1_start + 100,
                t1_end,
                t1_end + 100,
            ]
            xticklabels = [
                "1900",
                "1950",
                "2000",
                "2050",
                "2100/3900",
                "3950",
                "4000",
            ]

            xticklabels2 = [
                "1900",
                "2000",
                "2100/3900",
                "4000",
            ]

            ax_wave.set_xticks(xticks)
            ax_wave.set_xticklabels(xticklabels)
            ax_wave.set_xlim(1900, 2200)

            ax_wave.axvline(x=t1_end, color="k", linestyle="--", linewidth=1)

            ax_wave_2.set_xticks(xticks2)
            ax_wave_2.set_xticklabels(
                xticklabels2, fontsize=30, fontfamily="Arial", fontweight="bold"
            )
            ax_wave_2.set_xlim(1900, 2200)

            ax_wave_3.set_xticks(xticks)
            ax_wave_3.set_xticklabels(xticklabels)
            ax_wave_3.set_xlim(1900, 2200)

        ax_wave.text(
            t1_end, 7.5, "//", ha="center", va="top", fontsize=18, fontweight="bold"
        )
        ax_wave.text(
            t1_end, 1.5, "//", ha="center", va="top", fontsize=18, fontweight="bold"
        )

        x_axes = ax_wave_2.transAxes.inverted().transform(
            ax_wave_2.transData.transform((t1_end, 0))
        )[0]
        d1 = 0.02  # 軸からのはみ出し
        d2 = 0.01  # 波の振幅
        wn = 21  # 波数（奇数）

        pp = (0, d2, 0, -d2)

        # ---- 縦方向に展開（ここが回転の本体）----
        py = np.linspace(-d1, 1 + d1, wn)
        px = np.array([x_axes + pp[i % 4] for i in range(wn)])

        p = Path(list(zip(px, py)), [Path.MOVETO] + [Path.CURVE3] * (wn - 1))

        # ---- 外側：黒（太）----
        outer2 = mpatches.PathPatch(
            p,
            lw=6,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_2.transAxes,
            zorder=10,
            capstyle="round",
        )

        # ---- 内側：白（細）----
        inner2 = mpatches.PathPatch(
            p,
            lw=3,
            edgecolor="white",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_2.transAxes,
            zorder=11,
            capstyle="round",
        )
        # ---- 外側：黒（太）----
        outer3 = mpatches.PathPatch(
            p,
            lw=6,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_3.transAxes,
            zorder=10,
            capstyle="round",
        )

        # ---- 内側：白（細）----
        inner3 = mpatches.PathPatch(
            p,
            lw=3,
            edgecolor="white",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_3.transAxes,
            zorder=11,
            capstyle="round",
        )

        ax_wave_2.add_patch(outer2)
        ax_wave_2.add_patch(inner2)
        ax_wave_3.add_patch(outer3)
        ax_wave_3.add_patch(inner3)

        # 背景（LDサイクル）の描画 [4]
        draw_light_darktime_background(
            ax_wave,
            sol.t,
            [0, ld_end_time // (L_phase + D_phase) * (L_phase + D_phase)],
            L_phase,
            D_phase,
        )

        draw_light_darktime_background(
            ax_wave_2,
            sol_2.t,
            [0, ld_end_time // (L_phase_2 + D_phase_2) * (L_phase_2 + D_phase_2)],
            L_phase_2,
            D_phase_2,
        )

        draw_light_darktime_background(
            ax_wave_3,
            sol_3.t,
            [0, ld_end_time // (L_phase_2 + D_phase_2) * (L_phase_2 + D_phase_2)],
            L_phase_2,
            D_phase_2,
        )

        ax_wave.set_ylim(0, 8.5)
        ax_wave.set_ylabel("V (t)")
        ax_wave.text(
            0.02,
            0.85,
            f"{labels_wave[idx]}",
            transform=ax_wave.transAxes,
            fontsize=18,
            fontweight="bold",
        )
        if idx == 2:
            ax_wave.set_xlabel("Time (h)")

        ax_wave_2.set_ylim(0, 8.5)
        ax_wave_2.set_ylabel("V (t)")
        if idx == 2:
            ax_wave_2.set_xlabel("Time (h)")

        ax_wave_3.set_ylim(0, 8.5)
        ax_wave_3.set_ylabel("V (t)")
        if idx == 2:
            ax_wave_3.set_xlabel("Time (h)")

        V_chunk = np.array([sol.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk = pd.DataFrame(V_chunk.T, columns=[f"V{i}" for i in range(cells_num)])
        df_chunk.insert(0, "t", sol.t)
        period_data = evaluate_period(df=df_chunk, threshold=3)

        ax_wave.text(
            0.85,
            0.9,
            rf"${round(period_data['V0'][1], 2)}~h$",
            transform=ax_wave.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
        )

        ax_wave.text(
            0.1,
            0.9,
            rf"${round(L_phase * 2, 2)}~h$",
            transform=ax_wave.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="left",
        )

        V_chunk_2 = np.array([sol_2.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk_2 = pd.DataFrame(
            V_chunk_2.T, columns=[f"V{i}" for i in range(cells_num)]
        )
        df_chunk_2.insert(0, "t", sol.t)
        period_data_2 = evaluate_period(df=df_chunk_2, threshold=3)

        ax_wave_2.text(
            0.85,
            0.9,
            f"{round(period_data_2['V0'][1], 2)} h",
            transform=ax_wave_2.transAxes,
            fontsize=40,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily="Arial",
        )

        V_chunk_3 = np.array([sol_3.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk_3 = pd.DataFrame(
            V_chunk_3.T, columns=[f"V{i}" for i in range(cells_num)]
        )
        df_chunk_3.insert(0, "t", sol.t)
        period_data_3 = evaluate_period(df=df_chunk_3, threshold=3)

        ax_wave_3.text(
            0.85,
            0.9,
            f"{round(period_data_3['V0'][1], 2)} h",
            transform=ax_wave_3.transAxes,
            fontsize=40,
            fontweight="bold",
            va="top",
            ha="left",
            fontfamily="Arial",
        )

        ax_main.scatter(
            period_data["V0"][1], g_sc_sample / 2, marker="*", c="k", s=400, alpha=0.5
        )
        labelx = period_data["V0"][1] + 0.5
        labely = g_sc_sample / 2 - 0.01
        if idx == 0:
            ax_main.text(
                labelx - 2.5,
                labely,
                "(f)&(g)",
                fontsize=24,
                fontweight="bold",
            )
        if idx == 2:
            ax_main.text(
                labelx,
                labely,
                "(h)",
                fontsize=24,
                fontweight="bold",
            )

    fig.suptitle(
        "(q)" + "  " + rf"$w_C={vip_core}, w_S={vip_shell}, 10neurons$",
        fontsize=30,
        y=0.96,
    )

    fig.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.93, 0.6), handlelength=4
    )
    fig2.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.9, 0.95), handlelength=4
    )
    fig3.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.9, 0.95), handlelength=4
    )
    os.makedirs("results", exist_ok=True)
    fig.savefig("./results/Figure3_21.png", bbox_inches="tight", dpi=300)
    # fig2.savefig(
    #     f"./results/Aftereffect_Shift_{vip_condition}_0.08.png", bbox_inches="tight", dpi=300
    # )
    # fig3.savefig(
    #     f"./results/Aftereffect_Shift_{vip_condition}_0.png", bbox_inches="tight", dpi=300
    # )
    plt.show()


if __name__ == "__main__":
    plot_aftereffect_shift_layout()
# %%
