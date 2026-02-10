from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, perf_counter_ns
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from contextlib import contextmanager
from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, is_topological_order
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html

# --- Decoders
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.scheduling import parallel_sgs_selective, parallel_sgs_selective_greedy, parallel_sgs_selective_fast

# --- Searchers
from rcpsp_marketing.algorithms.local_search.hill_climb import hill_climb
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing


# ============================================================
# точечный профайлер (АГРЕГАЦИЯ ПО СЕКЦИЯМ)
# ============================================================

class Prof:
    """Точечный профайлер по секциям: total/mean/max/count/share."""
    def __init__(self):
        self._total_ns = defaultdict(int)
        self._count = defaultdict(int)
        self._max_ns = defaultdict(int)

    @contextmanager
    def sec(self, name: str):
        t0 = perf_counter_ns()
        try:
            yield
        finally:
            dt = perf_counter_ns() - t0
            self._total_ns[name] += dt
            self._count[name] += 1
            if dt > self._max_ns[name]:
                self._max_ns[name] = dt

    def to_rows(self) -> List[Dict[str, Any]]:
        total_all = sum(self._total_ns.values()) or 1
        rows: List[Dict[str, Any]] = []
        for name, tot in self._total_ns.items():
            cnt = self._count[name]
            mx = self._max_ns[name]
            rows.append({
                "section": name,
                "total_sec": tot / 1e9,
                "count": int(cnt),
                "mean_ms": (tot / max(1, cnt)) / 1e6,
                "max_ms": mx / 1e6,
                "share_%": (tot / total_all) * 100.0,
            })
        rows.sort(key=lambda r: r["total_sec"], reverse=True)
        return rows

    def print_top(self, n: int = 40) -> None:
        print("\n[profile] top sections:")
        for r in self.to_rows()[:n]:
            print(
                f"  - {r['section']:<45}"
                f" total={r['total_sec']:8.3f}s"
                f" mean={r['mean_ms']:8.3f}ms"
                f" max={r['max_ms']:8.3f}ms"
                f" cnt={r['count']:7d}"
                f" share={r['share_%']:6.2f}%"
            )


def timed_decode_wrapper(prof: Prof, dec_name: str, decode_fn):
    """Оборачиваем decode_fn, чтобы мерить ДЕКОД внутри HC/SA/RLS."""
    def _wrapped(proj, order, T):
        with prof.sec(f"decode.{dec_name}"):
            return decode_fn(proj, order, T)
    return _wrapped


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if reminding := m:
            return reminding.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def save_schedule_csv(proj: Any, sched: Any, selected: Iterable[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in selected:
        if j not in sched.start:
            continue
        rows.append({
            "job": int(j),
            "start": int(sched.start[j]),
            "finish": int(sched.finish[j]),
            "dur": int(sched.finish[j] - sched.start[j]),
            "type": getattr(proj.tasks[j], "job_type", "unknown") if hasattr(proj, "tasks") and j in proj.tasks else "unknown",
            "cost": float(getattr(proj.tasks[j], "total_cost", 0.0)) if hasattr(proj, "tasks") and j in proj.tasks else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(["start", "job"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_order_json(order: List[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def boxplot_values(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for (dec, sea), g in df.groupby(["decoder", "searcher"]):
        fig.add_trace(go.Box(
            y=g["value"],
            name=f"{dec}+{sea}",
            boxmean=True,
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="decoder + searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    return fig


# =========================
# Decoder registry
# =========================

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    decode_fn: Callable[[Any, List[int], int], Any]
    allow_search: bool = True


def make_decoders(*, greedy_min_score: float, greedy_unlock_weight: float) -> List[DecoderSpec]:
    def decode_ssgs(proj: Any, order: List[int], T: int):
        return serial_sgs_selective(proj, order, T=T, include_dummies=True)

    def decode_psgs(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_fast(proj, order, T=T, include_dummies=True, include_sink=False)

    def decode_psgs_greedy(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_greedy(
            proj, order,
            T=T,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return [
        DecoderSpec("SSGS", decode_ssgs, True),
        DecoderSpec("PSGS", decode_psgs, True),
        DecoderSpec("PSGS_greedy", decode_psgs_greedy, False),  # baseline only
    ]


# =========================
# Search registry
# =========================

@dataclass(frozen=True)
class SearchSpec:
    name: str
    run: Callable[..., Tuple[List[int], Dict[str, Any]]]


def rls_generic(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int,
    iters: int,
    tries_per_iter: int,
    max_profit_evals: Optional[int],
    prof: Optional[Prof] = None,   # <-- NEW
) -> Tuple[List[int], Dict[str, Any]]:
    rnd = __import__("random").Random(seed)

    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    movable_idx = [i for i, j in enumerate(start_order) if j != src and j != snk]
    if len(movable_idx) < 2:
        return list(start_order), {
            "accepted": 0,
            "profit_evals": 0,
            "iters_done": 0,
            "stopped_reason": "not_enough_movable",
        }

    profit_evals = 0
    accepted = 0
    iters_done = 0
    stopped_reason = ""

    def profit_eval(order: List[int]) -> float:
        nonlocal profit_evals
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")

        if prof is None:
            res = decode_fn(proj, order, T)
            obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)
        else:
            with prof.sec("rls.profit.decode"):
                res = decode_fn(proj, order, T)
            with prof.sec("rls.profit.objective"):
                obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

        profit_evals += 1
        return float(obj.value)

    cur = list(start_order)
    try:
        cur_val = profit_eval(cur)
    except StopIteration:
        return list(start_order), {
            "accepted": 0,
            "profit_evals": profit_evals,
            "iters_done": 0,
            "stopped_reason": "budget",
        }

    best = list(cur)
    best_val = cur_val

    for it in range(iters):
        iters_done = it + 1
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        for _ in range(tries_per_iter):
            if max_profit_evals is not None and profit_evals >= max_profit_evals:
                stopped_reason = "budget"
                break

            i, k = rnd.sample(movable_idx, 2)
            cand = list(cur)
            cand[i], cand[k] = cand[k], cand[i]

            if prof is None:
                if not is_topological_order(proj, cand):
                    continue
            else:
                with prof.sec("rls.topo_check"):
                    ok = is_topological_order(proj, cand)
                if not ok:
                    continue

            try:
                cand_val = profit_eval(cand)
            except StopIteration:
                stopped_reason = "budget"
                break

            if cand_val > cur_val:
                cur = cand
                cur_val = cand_val
                accepted += 1
                if cand_val > best_val:
                    best_val = cand_val
                    best = list(cand)
                break

        if stopped_reason == "budget":
            break

    if not stopped_reason:
        stopped_reason = "iters"

    return best, {
        "accepted": accepted,
        "profit_evals": profit_evals,
        "iters_done": iters_done,
        "stopped_reason": stopped_reason,
    }


def make_searchers(
    *,
    hc_iters: int,
    hc_tries: int,
    hc_neighbor: str,
    sa_iters: int,
    sa_tries: int,
    sa_neighbor: str,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,
    rls_iters: int,
    rls_tries: int,
    max_profit_evals: Optional[int],
) -> List[SearchSpec]:

    def run_none(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        return list(kwargs["start_order"]), {"stopped_reason": "baseline"}

    def run_hc(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        res = hill_climb(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=hc_iters,
            tries_per_iter=hc_tries,
            neighbor=hc_neighbor,
            max_profit_evals=max_profit_evals,
        )
        info = {
            "accepted": getattr(res, "accepted", 0),
            "profit_evals": getattr(res, "profit_evals", 0),
            "iters_done": getattr(res, "iters_done", 0),
            "stopped_reason": getattr(res, "stopped_reason", ""),
        }
        return list(res.best_order), info

    def run_sa(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        res = simulated_annealing(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=sa_iters,
            tries_per_iter=sa_tries,
            neighbor=sa_neighbor,
            T0=sa_T0,
            alpha=sa_alpha,
            Tmin=sa_Tmin,
            keep_history=False,
            max_profit_evals=max_profit_evals,
        )
        info = {
            "accepted": getattr(res, "accepted", 0),
            "profit_evals": getattr(res, "profit_evals", 0),
            "iters_done": getattr(res, "iters_done", 0),
            "stopped_reason": getattr(res, "stopped_reason", ""),
            "improved_best": getattr(res, "improved_best", 0),
        }
        return list(res.best_order), info

    # ВАЖНО: RLS теперь принимает prof (если передали)
    def run_rls(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        best, info = rls_generic(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=rls_iters,
            tries_per_iter=rls_tries,
            max_profit_evals=max_profit_evals,
            prof=kwargs.get("prof"),
        )
        return list(best), info

    return [
        SearchSpec("none", run_none),
        SearchSpec("HC", run_hc),
        SearchSpec("RLS", run_rls),
        SearchSpec("SA", run_sa),
    ]


# =========================
# Run one instance
# =========================

def run_instance(
    instance_path: Path,
    *,
    out_root: Path,
    T: int,
    seeds: List[int],
    greedy_min_score: float,
    greedy_unlock_weight: float,
    max_profit_evals: Optional[int],
    # HC
    hc_iters: int,
    hc_tries: int,
    hc_neighbor: str,
    # SA
    sa_iters: int,
    sa_tries: int,
    sa_neighbor: str,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,
    # RLS
    rls_iters: int,
    rls_tries: int,
    # saving policy
    save_all_runs: bool,
) -> None:
    prof = Prof()

    with prof.sec("parse_instance"):
        proj = PSPLibExtendedParser().parse(instance_path)

    with prof.sec("paths+mkdir"):
        cls = detect_class_from_path(instance_path)
        inst_name = safe_name(proj.name or instance_path.stem)

        inst_dir = out_root / cls / inst_name
        inst_dir.mkdir(parents=True, exist_ok=True)

    with prof.sec("make_decoders"):
        decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)

    with prof.sec("make_searchers"):
        searchers = make_searchers(
            hc_iters=hc_iters, hc_tries=hc_tries, hc_neighbor=hc_neighbor,
            sa_iters=sa_iters, sa_tries=sa_tries, sa_neighbor=sa_neighbor, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
            rls_iters=rls_iters, rls_tries=rls_tries,
            max_profit_evals=max_profit_evals,
        )

    # --- FAST lookup by name (micro-optimization and cleaner code) ---
    dec_by_name: Dict[str, DecoderSpec] = {d.name: d for d in decoders}
    sea_by_name: Dict[str, SearchSpec] = {s.name: s for s in searchers}

    starts_dir = inst_dir / "start_orders"
    starts_dir.mkdir(parents=True, exist_ok=True)

    start_orders: Dict[int, List[int]] = {}
    with prof.sec("start_orders.generate+save"):
        for s in seeds:
            order0 = random_topo_sort_fixed_ends(proj, seed=s)
            start_orders[s] = order0
            save_order_json(order0, starts_dir / f"start_order_seed{s}.json")

    rows: List[Dict[str, Any]] = []

    # ============================================================
    # OPTIMIZATION: store best_order (and obj) during the main loop
    # ============================================================
    # key -> dict with {"best_value", "best_row", "best_order"}
    best_per_combo: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for dec in decoders:
        decode_timed = timed_decode_wrapper(prof, dec.name, dec.decode_fn)

        for sea in searchers:
            if (not dec.allow_search) and sea.name != "none":
                continue

            combo_dir = inst_dir / f"decoder={dec.name}" / f"search={sea.name}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            for s in seeds:
                start_order = start_orders[s]
                seed_algo = 10_000 + s

                t0_ns = perf_counter_ns()

                with prof.sec(f"search.run.{dec.name}+{sea.name}"):
                    best_order, info = sea.run(
                        proj=proj,
                        T=T,
                        start_order=start_order,
                        decode_fn=decode_timed,   # <--- измеряем декод внутри поиска
                        seed_algo=seed_algo,
                        prof=prof,               # <--- RLS может использовать
                    )

                with prof.sec(f"decode.final.{dec.name}"):
                    res = dec.decode_fn(proj, best_order, T)

                with prof.sec("objective.final"):
                    obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

                dt_sec = (perf_counter_ns() - t0_ns) / 1e9

                run_tag = f"seed{s}"
                run_dir = combo_dir / run_tag
                html_path = run_dir / "schedule.html"

                if save_all_runs:
                    with prof.sec("save.mkdir"):
                        run_dir.mkdir(parents=True, exist_ok=True)

                    with prof.sec("save.plot_gantt"):
                        fig = plot_schedule_gantt(
                            proj, res.schedule,
                            selected=res.selected,
                            title=f"{proj.name} | {dec.name}+{sea.name} | seed={s} | T={T} | value={obj.value:_.2f}",
                            T=T,
                        )

                    with prof.sec("save.write_html"):
                        save_schedule_html(fig, html_path)

                    with prof.sec("save.write_csv"):
                        save_schedule_csv(proj, res.schedule, res.selected, run_dir / "schedule.csv")

                    with prof.sec("save.write_order"):
                        save_order_json(best_order, run_dir / "best_order.json")

                row = {
                    "class": cls,
                    "instance": inst_name,
                    "instance_path": str(instance_path),
                    "T": T,
                    "seed": s,
                    "seed_algo": seed_algo,
                    "decoder": dec.name,
                    "searcher": sea.name,

                    "value": float(obj.value),
                    "revenue": float(obj.revenue),
                    "cost": float(obj.cost),
                    "makespan": int(obj.makespan),
                    "selected": len(res.selected),
                    "skipped": len(getattr(res, "skipped", [])),
                    "time_sec": float(dt_sec),

                    "profit_evals": info.get("profit_evals"),
                    "accepted": info.get("accepted"),
                    "iters_done": info.get("iters_done"),
                    "stopped_reason": info.get("stopped_reason"),
                    "improved_best": info.get("improved_best"),

                    "saved_schedule_html": str(html_path) if save_all_runs else "",
                }
                rows.append(row)

                # --- OPTIMIZATION: store best_order so we DON'T rerun search later ---
                key = (dec.name, sea.name)
                cur = best_per_combo.get(key)
                if (cur is None) or (obj.value > cur["best_value"]):
                    best_per_combo[key] = {
                        "best_value": float(obj.value),
                        "best_row": row,            # keeps best seed, etc.
                        "best_order": best_order,   # <-- crucial
                    }

    with prof.sec("write_results_csv"):
        df = pd.DataFrame(rows).sort_values(["decoder", "searcher", "seed"])
        (inst_dir / "results.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    with prof.sec("write_summary_csv"):
        summary_rows = []
        for (dec_name, sea_name), best in best_per_combo.items():
            best_val = float(best["best_value"])
            best_row = best["best_row"]
            g = df[(df["decoder"] == dec_name) & (df["searcher"] == sea_name)]
            summary_rows.append({
                "decoder": dec_name,
                "searcher": sea_name,
                "best_value": best_val,
                "mean_value": float(g["value"].mean()),
                "std_value": float(g["value"].std(ddof=1)) if len(g) > 1 else 0.0,
                "best_seed": int(best_row["seed"]),
                "mean_time": float(g["time_sec"].mean()),
                "mean_selected": float(g["selected"].mean()),
            })
        summary = pd.DataFrame(summary_rows).sort_values(["decoder", "searcher"])
        (inst_dir / "summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")

    with prof.sec("write_boxplot_html"):
        fig_cmp = boxplot_values(df, title=f"{proj.name} | value distribution | T={T}")
        fig_cmp.write_html(str(inst_dir / "comparison_boxplot.html"))

    with prof.sec("write_meta_json"):
        meta = {
            "instance_name": proj.name,
            "instance_path": str(instance_path),
            "class": cls,
            "T": T,
            "seeds": list(seeds),
            "max_profit_evals": max_profit_evals,
            "HC": {"iters": hc_iters, "tries": hc_tries, "neighbor": hc_neighbor},
            "SA": {"iters": sa_iters, "tries": sa_tries, "neighbor": sa_neighbor, "T0": sa_T0, "alpha": sa_alpha, "Tmin": sa_Tmin},
            "RLS": {"iters": rls_iters, "tries": rls_tries},
            "greedy": {"min_score": greedy_min_score, "unlock_weight": greedy_unlock_weight},
            "save_all_runs": save_all_runs,
            "decoders": [d.name for d in decoders],
            "searchers": [s.name for s in searchers],
        }
        (inst_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ============================================================
    # REPLACE expensive recompute with cheap save-only-best
    # ============================================================
    if not save_all_runs:
        with prof.sec("save_best_without_recompute"):
            for (dec_name, sea_name), best in best_per_combo.items():
                best_row = best["best_row"]
                best_order: List[int] = best["best_order"]
                best_val = float(best["best_value"])
                s = int(best_row["seed"])

                dec2 = dec_by_name[dec_name]  # fast lookup; no next(...)
                # sea2 not needed anymore

                with prof.sec(f"best.decode.final.{dec_name}"):
                    res = dec2.decode_fn(proj, best_order, T)

                combo_dir = inst_dir / f"decoder={dec_name}" / f"search={sea_name}"
                run_dir = combo_dir / f"BEST_seed{s}"
                run_dir.mkdir(parents=True, exist_ok=True)

                with prof.sec("best.save.plot_gantt"):
                    fig = plot_schedule_gantt(
                        proj, res.schedule,
                        selected=res.selected,
                        title=f"{proj.name} | {dec_name}+{sea_name} | BEST seed={s} | T={T} | value={best_val:_.2f}",
                        T=T,
                    )

                with prof.sec("best.save.write_html"):
                    save_schedule_html(fig, run_dir / "schedule.html")

                with prof.sec("best.save.write_csv"):
                    save_schedule_csv(proj, res.schedule, res.selected, run_dir / "schedule.csv")

                with prof.sec("best.save.write_order"):
                    save_order_json(best_order, run_dir / "best_order.json")

    # печать и сохранение профиля
    prof.print_top(60)
    prof_df = pd.DataFrame(prof.to_rows())
    prof_df.to_csv(inst_dir / "timing_sections.csv", index=False, encoding="utf-8")
    print("[profile] saved:", inst_dir / "timing_sections.csv")

    print("[ok]", proj.name, "->", inst_dir)


def build_global_reports(out_root: Path) -> None:
    """
    Собирает глобальный отчёт по всем уже просчитанным инстансам в out_root:
      - all_results.csv  (все прогоны)
      - by_instance.csv  (агрегаты по каждому файлу/инстансу)
      - by_class.csv     (агрегаты по категориям j30/j60/j90/j120)
      - by_combo.csv     (агрегаты по комбинациям decoder+searcher)
      - by_combo_objective.csv (агрегаты по decoder+searcher+objective)
      - by_objective.csv (агрегаты только по objective)
      - overview_*.html  (boxplot + лидерборды)
      - report.md        (краткий текстовый отчёт)
    """

    out_root = Path(out_root)
    result_files = sorted(out_root.rglob("results.csv"))

    if not result_files:
        print("[report] no results.csv found in:", out_root)
        return

    # 1) Load all run-level results
    frames = []
    for rf in result_files:
        try:
            df = pd.read_csv(rf)
            df["results_csv_path"] = str(rf)
            frames.append(df)
        except Exception as e:
            print("[report][skip]", rf, type(e).__name__, e)

    if not frames:
        print("[report] failed to read any results.csv")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # ---- Backward compatibility: objective column may be missing in older runs
    if "objective" not in all_df.columns:
        all_df["objective"] = "unknown"
    else:
        all_df["objective"] = all_df["objective"].fillna("unknown").astype(str)
        all_df.loc[all_df["objective"].str.strip() == "", "objective"] = "unknown"

    global_dir = out_root / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # 2) Save all runs
    all_path = global_dir / "all_results.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8")
    print("[report] saved:", all_path)

    # Helpers
    def agg_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        g = df.groupby(group_cols, as_index=False)
        out = g.agg(
            runs=("value", "count"),
            mean_value=("value", "mean"),
            std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
            best_value=("value", "max"),
            mean_time=("time_sec", "mean"),
            mean_selected=("selected", "mean"),
            mean_skipped=("skipped", "mean"),
            mean_makespan=("makespan", "mean"),
        )
        idx = df.groupby(group_cols)["value"].idxmax()
        best_rows = df.loc[idx, group_cols + ["seed", "value"]].rename(
            columns={"seed": "best_seed", "value": "best_value_check"}
        )
        out = out.merge(best_rows, on=group_cols, how="left")
        out = out.drop(columns=["best_value_check"])
        return out

    # 3) Per-instance summary
    by_instance = agg_table(all_df, ["class", "instance", "decoder", "searcher"]).sort_values(
        ["class", "instance", "decoder", "searcher"]
    )
    p_inst = global_dir / "by_instance.csv"
    by_instance.to_csv(p_inst, index=False, encoding="utf-8")
    print("[report] saved:", p_inst)

    # 4) By-class summary
    by_class = agg_table(all_df, ["class", "decoder", "searcher"]).sort_values(["class", "decoder", "searcher"])
    p_cls = global_dir / "by_class.csv"
    by_class.to_csv(p_cls, index=False, encoding="utf-8")
    print("[report] saved:", p_cls)

    # 5) By-combo summary (old)
    by_combo = agg_table(all_df, ["decoder", "searcher"]).sort_values(["decoder", "searcher"])
    p_combo = global_dir / "by_combo.csv"
    by_combo.to_csv(p_combo, index=False, encoding="utf-8")
    print("[report] saved:", p_combo)

    # ===== NEW: Objective-aware aggregates =====
    by_objective = agg_table(all_df, ["objective"]).sort_values(["objective"])
    p_obj = global_dir / "by_objective.csv"
    by_objective.to_csv(p_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_obj)

    by_combo_obj = agg_table(all_df, ["decoder", "searcher", "objective"]).sort_values(
        ["decoder", "searcher", "objective"]
    )
    p_combo_obj = global_dir / "by_combo_objective.csv"
    by_combo_obj.to_csv(p_combo_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_combo_obj)

    by_class_obj = agg_table(all_df, ["class", "objective", "decoder", "searcher"]).sort_values(
        ["class", "objective", "decoder", "searcher"]
    )
    p_class_obj = global_dir / "by_class_objective.csv"
    by_class_obj.to_csv(p_class_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_class_obj)

    # 6) Visualizations
    fig_all = go.Figure()
    for (dec, sea), g in all_df.groupby(["decoder", "searcher"]):
        fig_all.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
    fig_all.update_layout(
        title="All instances: value distribution by decoder+searcher",
        xaxis=dict(title="decoder+searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    p_fig_all = global_dir / "overview_all_boxplot.html"
    fig_all.write_html(str(p_fig_all))

    objectives = sorted(all_df["objective"].dropna().unique().tolist())
    for obj in objectives:
        sub = all_df[all_df["objective"] == obj]
        if sub.empty:
            continue
        fig = go.Figure()
        for (dec, sea), g in sub.groupby(["decoder", "searcher"]):
            fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
        fig.update_layout(
            title=f"Objective={obj}: value distribution by decoder+searcher",
            xaxis=dict(title="decoder+searcher"),
            yaxis=dict(title="value"),
            showlegend=False,
        )
        fig.write_html(str(global_dir / f"overview_objective_{obj}_boxplot.html"))

    fig_objcmp = go.Figure()
    for (dec, sea, obj), g in all_df.groupby(["decoder", "searcher", "objective"]):
        fig_objcmp.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}+{obj}", boxmean=True))
    fig_objcmp.update_layout(
        title="Objective-aware: value distribution by decoder+searcher+objective",
        xaxis=dict(title="decoder+searcher+objective"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    fig_objcmp.write_html(str(global_dir / "overview_combo_objective_boxplot.html"))

    classes = sorted(all_df["class"].dropna().unique().tolist())
    for cls in classes:
        sub = all_df[all_df["class"] == cls]
        if sub.empty:
            continue
        fig = go.Figure()
        for (dec, sea), g in sub.groupby(["decoder", "searcher"]):
            fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
        fig.update_layout(
            title=f"{cls}: value distribution by decoder+searcher",
            xaxis=dict(title="decoder+searcher"),
            yaxis=dict(title="value"),
            showlegend=False,
        )
        fig.write_html(str(global_dir / f"overview_{cls}_boxplot.html"))

    top_mean = by_combo.sort_values("mean_value", ascending=False).head(15)
    top_best = by_combo.sort_values("best_value", ascending=False).head(15)

    top_mean_obj = by_combo_obj.sort_values("mean_value", ascending=False).head(20)
    top_best_obj = by_combo_obj.sort_values("best_value", ascending=False).head(20)

    def bar_leaderboard_combo(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
        x = [f"{d}+{s}" for d, s in zip(df["decoder"], df["searcher"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=df[metric]))
        fig.update_layout(title=title, xaxis=dict(title="combo"), yaxis=dict(title=metric))
        return fig

    def bar_leaderboard_combo_obj(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
        x = [f"{d}+{s}+{o}" for d, s, o in zip(df["decoder"], df["searcher"], df["objective"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=df[metric]))
        fig.update_layout(title=title, xaxis=dict(title="combo+objective"), yaxis=dict(title=metric))
        return fig

    bar_leaderboard_combo(top_mean, "mean_value", "Top combos by MEAN value").write_html(
        str(global_dir / "leaderboard_top_mean.html")
    )
    bar_leaderboard_combo(top_best, "best_value", "Top combos by BEST value").write_html(
        str(global_dir / "leaderboard_top_best.html")
    )

    bar_leaderboard_combo_obj(top_mean_obj, "mean_value", "Top combo+objective by MEAN value").write_html(
        str(global_dir / "leaderboard_top_mean_objective.html")
    )
    bar_leaderboard_combo_obj(top_best_obj, "best_value", "Top combo+objective by BEST value").write_html(
        str(global_dir / "leaderboard_top_best_objective.html")
    )

    # 7) Markdown short report
    n_instances = all_df[["class", "instance"]].drop_duplicates().shape[0]
    n_runs = len(all_df)
    combos = all_df[["decoder", "searcher"]].drop_duplicates().shape[0]
    obj_count = all_df[["objective"]].drop_duplicates().shape[0]

    lines = []
    lines.append("# Algorithm comparison report\n")
    lines.append(f"- Instances: **{n_instances}**")
    lines.append(f"- Runs total: **{n_runs}**")
    lines.append(f"- Unique combos (decoder+searcher): **{combos}**")
    lines.append(f"- Objectives present: **{obj_count}** ({', '.join(objectives)})")
    lines.append("")

    lines.append("## By objective (mean/best)")
    for _, r in by_objective.iterrows():
        lines.append(f"- objective={r['objective']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Top-10 combos by mean value (objective-agnostic)")
    for _, r in by_combo.sort_values("mean_value", ascending=False).head(10).iterrows():
        lines.append(f"- {r['decoder']}+{r['searcher']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Top-10 combo+objective by mean value")
    for _, r in by_combo_obj.sort_values("mean_value", ascending=False).head(10).iterrows():
        lines.append(f"- {r['decoder']}+{r['searcher']}+{r['objective']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Files generated")
    lines.append(f"- {all_path.name}")
    lines.append(f"- {p_inst.name}")
    lines.append(f"- {p_cls.name}")
    lines.append(f"- {p_combo.name}")
    lines.append(f"- {p_obj.name}")
    lines.append(f"- {p_combo_obj.name}")
    lines.append(f"- by_class_objective.csv")
    lines.append(f"- overview_all_boxplot.html")
    lines.append(f"- overview_objective_<objective>_boxplot.html")
    lines.append(f"- overview_combo_objective_boxplot.html")
    lines.append(f"- overview_<class>_boxplot.html")
    lines.append(f"- leaderboard_top_mean.html")
    lines.append(f"- leaderboard_top_best.html")
    lines.append(f"- leaderboard_top_mean_objective.html")
    lines.append(f"- leaderboard_top_best_objective.html")

    rep_path = global_dir / "report.md"
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print("[report] saved:", rep_path)
    print("[report] global dir:", global_dir)


# =========================
# MAIN (edit params here)
# =========================
def main():
    # ====== CONFIG: edit here ======
    MODE = "single"   # "single" | "dir"

    # --- single mode ---
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # --- dir mode ---
    IN_DIR = Path(r"data/extended/j60.sm")
    PATTERN = "*_with_metrics.sm"

    OUT_ROOT = Path(r"data/experiments/algorithm_comparison")

    T = 50
    SEEDS = list(range(1, 5))
    SAVE_ALL_RUNS = False

    # compute budget (calls to profit). None = no budget
    MAX_PROFIT_EVALS: Optional[int] = 10_000

    # PSGS greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0

    # HC params
    HC_ITERS = 10_000_000
    HC_TRIES = 50
    HC_NEIGHBOR = "insert"              # "swap" or "insert"

    # SA params
    SA_ITERS = 10_000_000
    SA_TRIES = 30
    SA_NEIGHBOR = "insert"
    SA_T0 = 2e5
    SA_ALPHA = 0.9998
    SA_TMIN = 1.0

    # RLS params
    RLS_ITERS = 10_000_000
    RLS_TRIES = 20

    # optional limit (only for dir mode; 0 = no limit)
    MAX_INSTANCES = 10
    # ====== END CONFIG ======

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)}")

    for p in files:
        if not p.exists() or not p.is_file():
            print("[skip] not found or not a file:", p)
            continue

        print("\n==============================================")
        print("[instance]", p)

        run_instance(
            p,
            out_root=OUT_ROOT,
            T=T,
            seeds=SEEDS,
            greedy_min_score=GREEDY_MIN_SCORE,
            greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
            max_profit_evals=MAX_PROFIT_EVALS,

            hc_iters=HC_ITERS,
            hc_tries=HC_TRIES,
            hc_neighbor=HC_NEIGHBOR,

            sa_iters=SA_ITERS,
            sa_tries=SA_TRIES,
            sa_neighbor=SA_NEIGHBOR,
            sa_T0=SA_T0,
            sa_alpha=SA_ALPHA,
            sa_Tmin=SA_TMIN,

            rls_iters=RLS_ITERS,
            rls_tries=RLS_TRIES,

            save_all_runs=SAVE_ALL_RUNS,
        )

    build_global_reports(OUT_ROOT)


if __name__ == "__main__":
    start_time = perf_counter()
    main()
    finish_time = perf_counter()
    elapsed = finish_time - start_time
    print(f"\n[total time] {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
