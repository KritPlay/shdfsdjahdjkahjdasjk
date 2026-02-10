from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import plotly.graph_objects as go

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.scheduling import serial_sgs_selective, parallel_sgs_selective_greedy, parallel_sgs_selective
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.core.improvement import left_shift
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing

def fmt_obj(obj) -> str:
    return f"value={obj.value:_.2f} revenue={obj.revenue:_.2f} cost={obj.cost:_.2f} makespan={obj.makespan}"


def save_sa_history_html(best_hist: List[float], cur_hist: List[float], out_path: Path, title: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best_hist, mode="lines", name="best_value"))
    fig.add_trace(go.Scatter(y=cur_hist, mode="lines", name="current_value"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="iteration"),
        yaxis=dict(title="value"),
        showlegend=True,
    )
    fig.write_html(str(out_path))
    return out_path


def main():


    # ====== параметры ======
    instance_path = r"data\extended\j120.sm\j1201_2_with_metrics.sm"
    T = 80
    seed_order = 42
    seed_sa = 123

    # PSGS-greedy (если хочешь финально упаковать этим декодером)
    greedy_min_score = -1e18
    greedy_unlock_weight = 0.0

    # SA параметры
    sa_iters = 50_000          # 500k на j120 может быть очень долго
    sa_T0 = 2e4
    sa_alpha = 0.9998
    sa_Tmin = 1.0
    sa_neighbor = "insert"      # "swap" быстрее, "insert" сильнее
    sa_tries = 30

    # ====== декодеры ======

    # SSGS selective 
    def decode_ssgs(proj_, order_, T_):
        return serial_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
        ) 

    # PSGS selective
    def decode_psgs(proj_, order_, T_):
        return parallel_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
        )   

    # PSGS with greedy packing rule
    def decode_psgs_greedy(proj_, order_, T_):
        return parallel_sgs_selective_greedy(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    # ====== стартовый порядок ======
    order0 = random_topo_sort_fixed_ends(proj, seed=seed_order)
    print("[info] start order len =", len(order0), "first10 =", order0[:10], "last5 =", order0[-5:])



    out_dir = Path("data/experiments/viz/sa_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== загрузка проекта ======
    proj = PSPLibExtendedParser().parse(instance_path)
    print("[ok] parsed:", proj.name)
    print("[info] T =", T)



if __name__ == "__main__":
    main()