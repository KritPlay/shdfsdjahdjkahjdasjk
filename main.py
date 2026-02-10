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

    out_dir = Path("data/experiments/viz/sa_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== загрузка проекта ======
    proj = PSPLibExtendedParser().parse(instance_path)
    print("[ok] parsed:", proj.name)
    print("[info] T =", T)

    # ====== декодер для SA: SSGS selective ======
    def decode_ssgs(proj_, order_, T_):
        return serial_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
        )

    # (опционально) декодер для финальной упаковки/визуализации: PSGS greedy
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

    def decode_psgs(proj_, order_, T_):
        return parallel_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
        )
    # ====== стартовый порядок ======
    order0 = random_topo_sort_fixed_ends(proj, seed=seed_order)
    print("[info] start order len =", len(order0), "first10 =", order0[:10], "last5 =", order0[-5:])

    # ====== baseline (до SA) на SSGS ======
    res0 = decode_ssgs(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline SSGS]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    res0 = decode_psgs(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline PSGS]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    res0 = decode_psgs_greedy(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline PSGS with gready]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    fig0 = plot_schedule_gantt(

        proj, res0.schedule,
        selected=res0.selected,
        title=f"{proj.name} | BASELINE SSGS | T={T} | {fmt_obj(obj0)}",
        T=T,
    )
    p0 = save_schedule_html(fig0, out_dir / f"{proj.name}_T{T}_baseline_ssgs.html")
    print("[ok] saved:", p0)

    # ====== SA (оценка через SSGS) ======
    print("\n[sa] start (SSGS decode)...")
    t0 = perf_counter()
    sa = simulated_annealing(
        proj,
        T=T,
        start_order=order0,
        decode_fn=decode_psgs,
        seed=seed_sa,
        iters=sa_iters,
        T0=sa_T0,
        Tmin=sa_Tmin,
        alpha=sa_alpha,
        neighbor=sa_neighbor,
        tries_per_iter=sa_tries,
        keep_history=True,
    )
    dt_sa = perf_counter() - t0
    print("[sa] done in", f"{dt_sa:.3f}s",
          "| accepted=", sa.accepted,
          "| improved_best=", sa.improved_best,
          "| best_value=", f"{sa.best_value:_.2f}",
          "| last_value=", f"{sa.last_value:_.2f}")

    if sa.history_best is not None and sa.history_cur is not None:
        p_hist = save_sa_history_html(
            sa.history_best,
            sa.history_cur,
            out_dir / f"{proj.name}_T{T}_sa_history_ssgs.html",
            title=f"{proj.name} SA history (SSGS) | T={T} | neighbor={sa_neighbor} | alpha={sa_alpha}",
        )
        print("[ok] saved:", p_hist)

    # ====== decode best SA order (SSGS) ======
    res_best = decode_psgs(proj, sa.best_order, T)
    obj_best = evaluate_profit_over_horizon(proj, res_best.schedule, selected_jobs=res_best.selected, T=T)
    print("\n[best after SA | SSGS]", fmt_obj(obj_best), "selected=", len(res_best.selected), "skipped=", len(res_best.skipped))

    fig_best = plot_schedule_gantt(
        proj, res_best.schedule,
        selected=res_best.selected,
        title=f"{proj.name} | BEST after SA (SSGS) | T={T} | {fmt_obj(obj_best)}",
        T=T,
    )
    p_best = save_schedule_html(fig_best, out_dir / f"{proj.name}_T{T}_best_after_sa_ssgs.html")
    print("[ok] saved:", p_best)

    # ====== пост-улучшатель (left_shift) на SSGS-расписании ======
    impL = left_shift(proj, res_best.schedule, selected_jobs=res_best.selected, T=T, hide_dummies=True)
    objL = evaluate_profit_over_horizon(proj, impL.schedule, selected_jobs=res_best.selected, T=T)
    print("\n[post left_shift | SSGS] moved=", impL.moved, "|", fmt_obj(objL))

    figL = plot_schedule_gantt(
        proj, impL.schedule,
        selected=res_best.selected,
        title=f"{proj.name} | AFTER left_shift (SSGS) moved={impL.moved} | T={T} | {fmt_obj(objL)}",
        T=T,
    )
    pL = save_schedule_html(figL, out_dir / f"{proj.name}_T{T}_after_left_shift_ssgs.html")
    print("[ok] saved:", pL)

    # ====== (опционально) “перепаковать” лучший порядок PSGS-greedy и тоже сохранить ======
    try:
        res_g = decode_psgs_greedy(proj, sa.best_order, T)
        obj_g = evaluate_profit_over_horizon(proj, res_g.schedule, selected_jobs=res_g.selected, T=T)
        print("\n[best order repacked | PSGS greedy]", fmt_obj(obj_g), "selected=", len(res_g.selected), "skipped=", len(res_g.skipped))

        fig_g = plot_schedule_gantt(
            proj, res_g.schedule,
            selected=res_g.selected,
            title=f"{proj.name} | BEST order repacked (PSGS greedy) | T={T} | {fmt_obj(obj_g)}",
            T=T,
        )
        p_g = save_schedule_html(fig_g, out_dir / f"{proj.name}_T{T}_best_order_psgs_greedy.html")
        print("[ok] saved:", p_g)
    except Exception as e:
        print("[warn] PSGS greedy repack skipped:", type(e).__name__, e)

    # ====== выбрать лучший вариант по value ======
    candidates: List[Tuple[str, float]] = [
        ("baseline_ssgs", obj0.value),
        ("best_after_sa_ssgs", obj_best.value),
        ("after_left_shift_ssgs", objL.value),
    ]
    best_tag, best_val = max(candidates, key=lambda x: x[1])
    print("\n[best overall] =", best_tag, "value=", f"{best_val:_.2f}")

    print("\n[files]")
    print(" baseline        :", p0)
    print(" best_after_sa   :", p_best)
    print(" after_left      :", pL)
    print(" out_dir         :", out_dir)


if __name__ == "__main__":
    main()
