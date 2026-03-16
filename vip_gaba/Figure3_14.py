# %%
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import json
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
from utils import neuron_model  # 自作モジュール
from matplotlib.path import Path
import matplotlib.patches as mpatches

# --- スタイル設定 ---
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


def detect_threshold_crossings(t, V, threshold):
    crossings = []
    for i in range(1, len(V)):
        if V[i - 1] < threshold and V[i] >= threshold:
            frac = (threshold - V[i - 1]) / (V[i] - V[i - 1])
            t_cross = t[i - 1] + frac * (t[i] - t[i - 1])
            crossings.append(t_cross)
    return np.array(crossings)


def evaluate_period(df, threshold):
    period_data = dict()
    for col in [c for c in df.columns if c.startswith("V")]:
        t, V = df["t"].values, df[col].values
        crossings = detect_threshold_crossings(t, V, threshold)
        if len(crossings) < 5:
            period_data[col] = [False, 0]
            continue
        intervals = np.diff(crossings)
        mean_period = np.mean(intervals[-13:-3])
        std_period = np.std(intervals[-13:-3])
        period_data[col] = [std_period < 0.1, mean_period]
    return period_data


# --- メイン統合描画 ---
def plot_a4_layout():
    vip_core = 50
    vip_shell = 50
    vip_condition = f"{vip_core}_{vip_shell}"
    hist_gabasc = ["0.02", "0.075", "0.09", "0.11"]

    # A4縦横比 (210:297) に基づくサイズ設定
    fig = plt.figure(figsize=(12, 17))

    fig3 = plt.figure(figsize=(12, 12))  # (d)~(f)の波形を初期状態とともにplot
    fig4 = plt.figure(figsize=(12, 12))  # (d)~(f)のgabaなしの波形を初期状態とともにplot
    # 4行4列のグリッド
    gs = GridSpec(
        4,
        4,
        figure=fig,
        width_ratios=[0.6, 0.6, 0.5, 0.5],
        height_ratios=[1.6, 0.55, 0.55, 0.55],
        wspace=0.5,
        hspace=0.8,
    )

    # --- 1. 左上: 散布図 (a) ---
    ax_main = fig.add_subplot(gs[0, :2])
    x_all, y_all = [], []
    csv_files = sorted(
        [
            g
            for g in os.listdir(f"./VIP_GABA_5_5cell_randint/{vip_condition}")
            if not g.startswith(".")
        ],
    )
    for csv_file in csv_files:
        df_res = pd.read_csv(
            os.path.join(f"./VIP_GABA_5_5cell_randint/{vip_condition}", csv_file)
        )
        for rs in set(df_res["rand_seed"]):
            sub = df_res[df_res["rand_seed"] == rs]
            if np.prod(sub["is_sync"]) == 1:
                y_all.append(sub["gaba_sc"].iloc[0])
                x_all.append(sub["period"].iloc[0])

    ax_main.scatter(x_all, y_all, color="k", s=10)
    ax_main.set_xlabel("Period (h)")
    ax_main.set_ylabel(r"$g_{SC}$")
    ax_main.set_xlim(20, 28)
    ax_main.set_ylim(0, 0.21)
    ax_main.text(
        0.02, 0.9, "(a)", transform=ax_main.transAxes, fontsize=24, fontweight="bold"
    )

    panel_labels = ["(b)", "(c)", "(d)", "(e)"]
    for idx, gabasc in enumerate(hist_gabasc):
        y_val = float(gabasc) / 2
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
            "VIP_GABA_5_5cell_randint",
            vip_condition,
            f"evaluate_period_summary_{gabasc}.csv",
        )
        if os.path.exists(path):
            res = pd.read_csv(path)
            x_hist = [
                res[res["rand_seed"] == rs]["period"].iloc[0]
                for rs in set(res["rand_seed"])
                if np.prod(res[res["rand_seed"] == rs]["is_sync"]) == 1
            ]
            ax_h.hist(
                x_hist,
                bins=np.arange(22, 26.1, 0.1).tolist(),
                color="gray",
                edgecolor="black",
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
            ax_h.set_ylim((0, 210))
            ax_h.set_title(r"$g_{{SC}}=$" + f"{y_val}", fontsize=15)
            if i >= 2:
                ax_h.set_xlabel("Period (h)", fontsize=15)

    # --- 3. 下段: 波形図 (c, d, e) ---
    # パラメータとシミュレーション
    with open("./phase_initial_conditions.json", "r") as f:
        phase_init_data = json.load(f)

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

    l_g_sc = [0.18, 0.18, 0.18]
    seeds = [264120, 264745, 264410]
    labels_wave = ["(f)", "(g)", "(h)"]
    wave_gs = gs[1:, :].subgridspec(
        3,
        1,
        hspace=0.3,  # ← wave 間を詰める
    )
    for idx, seed in enumerate(seeds):
        ax_wave = fig.add_subplot(wave_gs[idx, 0])  # 2行目以降を1行ずつ使用
        ax_wave_3 = fig3.add_subplot(3, 1, idx + 1)
        ax_wave_4 = fig4.add_subplot(3, 1, idx + 1)
        g_sc = l_g_sc[idx]
        ax_wave.text(
            0.5,
            1.03,
            r"$g_{SC}=$" + str(g_sc / 2),
            transform=ax_wave.transAxes,
            fontsize=20,
            ha="center",
            va="bottom",
        )

        rng = np.random.default_rng(seed)

        cells_num = 10
        cell_types = ["core"] * 5 + ["shell"] * 5
        vip_weights = [50] * cells_num
        gaba_mat = np.zeros((10, 10))
        for r in range(10):
            for e in range(10):
                if cell_types[e] == "shell" and cell_types[r] == "core":
                    gaba_mat[r][e] = g_sc / cells_num

        y0 = []
        for p in [rng.integers(360) for _ in range(cells_num)]:
            ps = str(p)
            y0.extend(
                [phase_init_data[ps][k] for k in ["M", "Pc", "Pn", "Ca", "V", "VIP"]]
            )

        cells = [
            neuron_model.SingleCell(cell_params, i, cell_type=cell_types[i])
            for i in range(10)
        ]
        net = neuron_model.CellNetwork(cells, vip_weights, gaba_mat)
        sol = solve_ivp(
            fun=lambda t, y: net.ode_all_neurons(t, y, (12, 12, 0, 0, 12, 12, 0, 2000)),
            t_span=[0, 2000],
            t_eval=np.arange(0, 2000, 0.01),
            y0=y0,
            method="LSODA",
        )

        net_0 = neuron_model.CellNetwork(
            cells, vip_weights, np.zeros((cells_num, cells_num))
        )
        sol_0 = solve_ivp(
            fun=lambda t, y: net_0.ode_all_neurons(
                t, y, (12, 12, 0, 0, 12, 12, 0, 2000)
            ),
            t_span=[0, 2000],
            t_eval=np.arange(0, 2000, 0.01),
            y0=y0,
            method="LSODA",
        )

        core_plot, shell_plot = False, False
        for i in range(10):
            ls, col = ("-", "red") if cell_types[i] == "core" else ("--", "blue")
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
            else:
                label = None
            ax_wave.plot(
                sol.t, sol.y[i * 6 + 4, :], c=col, linestyle=ls, alpha=0.7, label=label
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

            t1_start, t1_end = 0, 50
            t2_start, t2_end = 1850, 2000

            mask1 = (sol.t >= t1_start) & (sol.t <= t1_end)
            mask2 = (sol.t >= t2_start) & (sol.t <= t2_end)
            t_concat = np.concatenate(
                [sol.t[mask1], sol.t[mask2] - (t2_start - t1_end)]
            )

            V_concat = np.concatenate(
                [sol.y[i * 6 + 4, mask1], sol.y[i * 6 + 4, mask2]]
            )
            V_concat_0 = np.concatenate(
                [sol_0.y[i * 6 + 4, mask1], sol_0.y[i * 6 + 4, mask2]]
            )

            ax_wave_3.plot(
                t_concat, V_concat, c=col, linestyle=ls, alpha=0.7, label=label
            )
            ax_wave_4.plot(
                t_concat, V_concat_0, c=col, linestyle=ls, alpha=0.7, label=label
            )
            xticks = [t1_start, t1_end, t1_end + 50, t1_end + 100, t1_end + 150]

            xticklabels = [
                "0",
                "50/1850",
                "1900",
                "1950",
                "2000",
            ]

            ax_wave_3.set_xticks(xticks)
            ax_wave_3.set_xticklabels(
                xticklabels, fontsize=30, fontfamily="Arial", fontweight="bold"
            )
            ax_wave_3.set_xlim(0, 200)

            # ax_wave_3.axvline(
            #     x=t1_end, color="k", linestyle="--", linewidth=1
            # )

            ax_wave_4.set_xticks(xticks)
            ax_wave_4.set_xticklabels(
                xticklabels, fontsize=30, fontfamily="Arial", fontweight="bold"
            )
            ax_wave_4.set_xlim(0, 200)

            # ax_wave_4.axvline(
            #     x=t1_end, color="k", linestyle="--", linewidth=1
            # )

        x_axes = ax_wave_4.transAxes.inverted().transform(
            ax_wave_4.transData.transform((t1_end, 0))
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
        # ---- 外側：黒（太）----
        outer4 = mpatches.PathPatch(
            p,
            lw=6,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_4.transAxes,
            zorder=10,
            capstyle="round",
        )

        # ---- 内側：白（細）----
        inner4 = mpatches.PathPatch(
            p,
            lw=3,
            edgecolor="white",
            facecolor="none",
            clip_on=False,
            transform=ax_wave_4.transAxes,
            zorder=11,
            capstyle="round",
        )

        ax_wave_3.add_patch(outer3)
        ax_wave_3.add_patch(inner3)
        ax_wave_4.add_patch(outer4)
        ax_wave_4.add_patch(inner4)

        V_chunk = np.array([sol.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk = pd.DataFrame(V_chunk.T, columns=[f"V{i}" for i in range(cells_num)])
        df_chunk.insert(0, "t", sol.t)
        period_data = evaluate_period(df=df_chunk, threshold=3)
        if period_data["V0"][0]:
            ax_wave.text(
                0.85,
                0.8,
                rf"${round(period_data['V0'][1], 2)}~h$",
                transform=ax_wave.transAxes,
                ha="left",
                va="bottom",
                fontsize=25,
            )
            ax_wave_3.text(
                0.8,
                0.8,
                f"{round(period_data['V0'][1], 2)} h",
                transform=ax_wave_3.transAxes,
                ha="left",
                va="bottom",
                fontsize=40,
                fontfamily="Arial",
            )
        ax_wave.set_ylabel("V (t)")
        ax_wave.set_ylim(0, 8.5)
        ax_wave.set_xlim(1900, 2000)
        if idx == 2:
            ax_wave.set_xlabel("Time (h)")

        ax_wave_3.set_ylabel("V (t)")
        ax_wave_3.set_ylim(0, 8.5)
        if idx == 2:
            ax_wave_3.set_xlabel(
                "Time (h)", fontsize=40, fontfamily="Arial", fontweight="bold"
            )

        V_chunk_0 = np.array([sol_0.y[i * 6 + 4, :] for i in range(cells_num)])
        df_chunk_0 = pd.DataFrame(
            V_chunk_0.T, columns=[f"V{i}" for i in range(cells_num)]
        )
        df_chunk_0.insert(0, "t", sol.t)
        period_data_0 = evaluate_period(df=df_chunk_0, threshold=3)
        if period_data_0["V0"][0]:
            ax_wave_4.text(
                0.8,
                0.8,
                f"{round(period_data_0['V0'][1], 2)} h",
                transform=ax_wave_4.transAxes,
                ha="left",
                va="bottom",
                fontsize=40,
                fontfamily="Arial",
            )
        ax_wave_4.set_ylabel("V (t)")
        ax_wave_4.set_ylim(0, 8.5)
        if idx == 2:
            ax_wave_4.set_xlabel(
                "Time (h)", fontsize=40, fontfamily="Arial", fontweight="bold"
            )

        ax_wave.text(
            0.02,
            0.85,
            labels_wave[idx],
            transform=ax_wave.transAxes,
            fontsize=24,
            fontweight="bold",
        )
        ax_wave_3.axvspan(0, 200, color="gray", alpha=0.2)
        ax_wave_4.axvspan(0, 200, color="gray", alpha=0.2)
        ax_main.scatter(
            period_data["V0"][1], g_sc / 2, marker="*", c="r", s=400, alpha=0.7
        )

        if idx == 2:
            labelx = period_data["V0"][1] + 0.5
            labely = g_sc / 2 - 0.01
        else:
            labelx = period_data["V0"][1] - 0.3
            labely = g_sc / 2 + 0.01
        ax_main.text(
            labelx,
            labely,
            labels_wave[idx],
            fontsize=24,
            fontweight="bold",
        )

    fig.suptitle(
        "(q)" + "  " + rf"$w_C={vip_core}, w_S={vip_shell}, 10neurons$",
        fontsize=30,
        y=0.95,
    )
    fig3.suptitle(
        r"$g_{SC}=$" + str(round(g_sc / 2, 2)),
        fontsize=30,
        y=0.95,
    )

    fig4.suptitle(
        r"$g_{SC}=$" + str(0),
        fontsize=30,
        y=0.95,
    )

    fig.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.93, 0.6), handlelength=4
    )
    fig3.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.9, 0.95), handlelength=4
    )
    fig4.legend(
        loc="upper right", fontsize=15, bbox_to_anchor=(0.9, 0.95), handlelength=4
    )

    os.makedirs("results", exist_ok=True)
    fig.savefig("./results/Figure3_14.png", bbox_inches="tight", dpi=300)
    # fig3.savefig(
    #     f"./results/Waveshape_{vip_core}_{vip_shell}_{g_sc / 2}.png",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # fig4.savefig(
    #     f"./results/Waveshape_{vip_core}_{vip_shell}_0.png",
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    plt.show()


if __name__ == "__main__":
    plot_a4_layout()

# %%
