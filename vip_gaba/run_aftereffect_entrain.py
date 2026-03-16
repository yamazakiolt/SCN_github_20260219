#!/usr/bin/env bashs
from itertools import product
import subprocess
from concurrent.futures import ThreadPoolExecutor

# ==============================
# パラメータ定義
# ==============================
# CELLS = [(1, 1), (5, 5)]
CELLS = [(5, 5)]
VIP_CORE = [25, 50, 75, 100, 125]
VIP_SHELL = [25, 50, 75, 100, 125]

MAX_WORKERS = 4


def run_simulation(args):
    cell_pair, core, shell, seed_base = args
    cell_a, cell_b = cell_pair

    cmd = [
        "python",
        "VIP_GABA_aftereffect_entrain.py",
        "--vip_core",
        str(core),
        "--vip_shell",
        str(shell),
        "--core_cells",
        str(cell_a),
        "--shell_cells",
        str(cell_b),
        "--seed_base",
        str(seed_base),
    ]

    print("=" * 60)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # 1. phase初期化（最初に一回だけ）
    print("Generating phase_init.json ...")
    subprocess.run(["python", "calc_phase.py"], check=True)

    vip_pairs = list(product(VIP_CORE, VIP_SHELL))
    # 2. 全組み合わせを順番に実行

    tasks = []
    for cell_pair in CELLS:
        for vip_index, (core, shell) in enumerate(vip_pairs):
            # VIP組み合わせ順でseedを決定
            seed_base = 110 + 41000 * vip_index

            tasks.append((cell_pair, core, shell, seed_base))
    print(f"Total jobs: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(run_simulation, tasks)
    print("All simulations finished.")
