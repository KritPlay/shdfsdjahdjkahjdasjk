# tools/compare_end2end_ref_vs_fast.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends

# objectives
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

# decoders
from rcpsp_marketing.core.scheduling import (
    serial_sgs_selective,
    parallel_sgs_selective,
    parallel_sgs_selective_greedy,
)

# searchers (must be your UPDATED versions that accept objective/objective_fn)
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing
from rcpsp_marketing.algorithms.local_search.hill_climb import hill_climb
# if you updated your RLS to accept objective/objective_fn; otherwise set RUN_RLS=False
from rcpsp_marketing.algorithms.local_search.randomized_local_search import randomized_local_search


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def make_timed_decode(decode_fn: Callable[[Any, List[int], int], Any]) -> Tuple[Callable[[Any, List[int], int], Any], Dict[str, float]]:
    stats = {"calls": 0.0, "sec": 0.0}

    def wrapped(proj: Any, order: List[int], T: int) -> Any:
        t0 = perf_counter()
        out = decode_fn(proj, order, T)
        stats["sec"] += perf_counter() - t0
        stats["calls"] += 1.0
        return out

    return wrapped, stats


def make_timed_objective(obj_fn: Callable[..., Any]) -> Tuple[Callable[..., Any], Dict[str, float]]:
    stats = {"calls": 0.0, "sec": 0.0}

    def wrapped(proj: Any, schedule: Any, *, selected_jobs=None, T=None, include_dummy_costs: bool = False) -> Any:
        t0 = perf_counter()
        out = obj_fn(proj, schedule, selected_jobs=selected_jobs, T=T, include_dummy_costs=include_dummy_costs)
        stats["sec"] += perf_counter() - t0
        stats["calls"] += 1.0
        return out

    return wrapped, stats


# =========================
# Decoder registry
# =========================

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    decode_fn: Callable[[Any, List[int], int], Any]


def make_decoders(*, greedy_min_score: float, greedy_unlock_weight: float) -> Dict[str, DecoderSpec]:
    def decode_ssgs(proj: Any, order: List[int], T: int):
        return serial_sgs_selective(proj, order, T=T, include_dummies=True)

    def decode_psgs(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective(proj, order, T=T, include_dummies=True, include_sink=False)

    def decode_psgs_greedy(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_greedy(
            proj,
            order,
            T=T,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return {
        "SSGS": DecoderSpec("SSGS", decode_ssgs),
        "PSGS": DecoderSpec("PSGS", decode_psgs),
        "PSGS_greedy": DecoderSpec("PSGS_greedy", decode_psgs_greedy),
    }


# =========================
# End-to-end benchmark
# =========================

def run_one_searcher(
    *,
    searcher: str,  # "SA" | "HC" | "RLS"
    proj: Any,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed_algo: int,

    # objective impl
    obj_impl: str,  # "ref" | "fast"
    include_dummy_costs: bool,

    # budgets/params
    max_profit_evals: Optional[int],
    sa_iters: int,
    sa_neighbor: str,
    sa_tries: int,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,

    hc_iters: int,
    hc_neighbor: str,
    hc_tries: int,

    rls_iters: int,
    rls_tries: int,
) -> Dict[str, Any]:
    if obj_impl == "ref":
        base_obj_fn = evaluate_profit_over_horizon
    elif obj_impl == "fast":
        base_obj_fn = evaluate_profit_over_horizon_fast
    else:
        raise ValueError("obj_impl must be 'ref' or 'fast'")

    timed_decode, dec_stats = make_timed_decode(decode_fn)
    timed_obj, obj_stats = make_timed_objective(base_obj_fn)

    t0 = perf_counter()

    if searcher == "SA":
        res = simulated_annealing(
            proj,
            T=T,
            start_order=start_order,
            decode_fn=timed_decode,
            seed=seed_algo,
            iters=sa_iters,
            neighbor=sa_neighbor,
            tries_per_iter=sa_tries,
            T0=sa_T0,
            alpha=sa_alpha,
            Tmin=sa_Tmin,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key: custom objective
            include_dummy_costs=include_dummy_costs,
            keep_history=False,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = "budget" if res.stopped_by_budget else "iters_or_temp"

    elif searcher == "HC":
        res = hill_climb(
            proj,
            T=T,
            start_order=start_order,
            decode_fn=timed_decode,
            seed=seed_algo,
            iters=hc_iters,
            neighbor=hc_neighbor,
            tries_per_iter=hc_tries,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key
            include_dummy_costs=include_dummy_costs,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = str(res.stopped_reason)

    elif searcher == "RLS":
        res = randomized_local_search(
            proj,
            T=T,
            start_order=start_order,
            seed=seed_algo,
            iters=rls_iters,
            tries_per_iter=rls_tries,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key (your updated RLS)
            include_dummy_costs=include_dummy_costs,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = str(res.stopped_reason)

    else:
        raise ValueError("searcher must be 'SA' or 'HC' or 'RLS'")

    total_sec = perf_counter() - t0

    decode_sec = float(dec_stats["sec"])
    objective_sec = float(obj_stats["sec"])
    overhead_sec = max(0.0, total_sec - decode_sec - objective_sec)

    decode_calls = int(dec_stats["calls"])
    objective_calls = int(obj_stats["calls"])

    return {
        "best_value": best_value,
        "profit_evals": profit_evals,
        "accepted": accepted,
        "stopped_reason": stopped,

        "total_sec": total_sec,
        "decode_sec": decode_sec,
        "objective_sec": objective_sec,
        "overhead_sec": overhead_sec,

        "decode_calls": decode_calls,
        "objective_calls": objective_calls,
    }


def run_end2end(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],
    decoder_name: str,
    searchers: List[str],
    include_dummy_costs: bool,
    max_profit_evals: Optional[int],

    # params
    sa_iters: int,
    sa_neighbor: str,
    sa_tries: int,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,

    hc_iters: int,
    hc_neighbor: str,
    hc_tries: int,

    rls_iters: int,
    rls_tries: int,

    greedy_min_score: float,
    greedy_unlock_weight: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PSPLibExtendedParser()
    decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)
    if decoder_name not in decoders:
        raise ValueError(f"Unknown decoder '{decoder_name}', choose from {list(decoders.keys())}")
    decoder = decoders[decoder_name]

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)

        print(f"[instance] {cls} / {inst}")

        # paired start orders
        start_orders = {s: random_topo_sort_fixed_ends(proj, seed=s) for s in seeds}

        for sea in searchers:
            for s in seeds:
                start_order = start_orders[s]
                seed_algo = 10_000 + s  # keep deterministic

                # RUN ref
                ref = run_one_searcher(
                    searcher=sea,
                    proj=proj,
                    T=T,
                    start_order=start_order,
                    decode_fn=decoder.decode_fn,
                    seed_algo=seed_algo,
                    obj_impl="ref",
                    include_dummy_costs=include_dummy_costs,
                    max_profit_evals=max_profit_evals,
                    sa_iters=sa_iters, sa_neighbor=sa_neighbor, sa_tries=sa_tries, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
                    hc_iters=hc_iters, hc_neighbor=hc_neighbor, hc_tries=hc_tries,
                    rls_iters=rls_iters, rls_tries=rls_tries,
                )

                # RUN fast
                fast = run_one_searcher(
                    searcher=sea,
                    proj=proj,
                    T=T,
                    start_order=start_order,
                    decode_fn=decoder.decode_fn,
                    seed_algo=seed_algo,
                    obj_impl="fast",
                    include_dummy_costs=include_dummy_costs,
                    max_profit_evals=max_profit_evals,
                    sa_iters=sa_iters, sa_neighbor=sa_neighbor, sa_tries=sa_tries, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
                    hc_iters=hc_iters, hc_neighbor=hc_neighbor, hc_tries=hc_tries,
                    rls_iters=rls_iters, rls_tries=rls_tries,
                )

                # compare best_value equality (should match if objective impl is equivalent)
                dv = fast["best_value"] - ref["best_value"]
                ok_value = (dv == 0.0)

                # speedups
                speedup_total = (ref["total_sec"] / fast["total_sec"]) if fast["total_sec"] > 0 else float("inf")
                speedup_obj = (ref["objective_sec"] / fast["objective_sec"]) if fast["objective_sec"] > 0 else float("inf")

                rows.append({
                    "class": cls,
                    "instance": inst,
                    "instance_path": str(fp),
                    "decoder": decoder.name,
                    "searcher": sea,
                    "T": int(T),
                    "seed": int(s),
                    "seed_algo": int(seed_algo),
                    "include_dummy_costs": bool(include_dummy_costs),
                    "max_profit_evals": max_profit_evals if max_profit_evals is not None else "",

                    "ref_best_value": ref["best_value"],
                    "fast_best_value": fast["best_value"],
                    "diff_best_value": dv,
                    "ok_value_exact": bool(ok_value),

                    "ref_total_sec": ref["total_sec"],
                    "fast_total_sec": fast["total_sec"],
                    "speedup_total": speedup_total,

                    "ref_decode_sec": ref["decode_sec"],
                    "fast_decode_sec": fast["decode_sec"],
                    "ref_objective_sec": ref["objective_sec"],
                    "fast_objective_sec": fast["objective_sec"],
                    "speedup_objective": speedup_obj,

                    "ref_overhead_sec": ref["overhead_sec"],
                    "fast_overhead_sec": fast["overhead_sec"],

                    "ref_profit_evals": ref["profit_evals"],
                    "fast_profit_evals": fast["profit_evals"],
                    "ref_decode_calls": ref["decode_calls"],
                    "fast_decode_calls": fast["decode_calls"],
                    "ref_objective_calls": ref["objective_calls"],
                    "fast_objective_calls": fast["objective_calls"],

                    "ref_accepted": ref["accepted"],
                    "fast_accepted": fast["accepted"],
                    "ref_stopped_reason": ref["stopped_reason"],
                    "fast_stopped_reason": fast["stopped_reason"],

                    # shares (how much of total time is spent where)
                    "ref_share_objective": ref["objective_sec"] / ref["total_sec"] if ref["total_sec"] > 0 else 0.0,
                    "fast_share_objective": fast["objective_sec"] / fast["total_sec"] if fast["total_sec"] > 0 else 0.0,
                })

    df = pd.DataFrame(rows)
    runs_path = out_dir / "end2end_ref_vs_fast_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # summary
    def _std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    summary = (
        df.groupby(["class", "instance", "decoder", "searcher"], as_index=False)
          .agg(
              runs=("seed", "count"),
              ok_rate=("ok_value_exact", "mean"),
              mean_speedup_total=("speedup_total", "mean"),
              std_speedup_total=("speedup_total", _std),
              mean_speedup_objective=("speedup_objective", "mean"),
              std_speedup_objective=("speedup_objective", _std),
              mean_ref_total_sec=("ref_total_sec", "mean"),
              mean_fast_total_sec=("fast_total_sec", "mean"),
              mean_ref_share_objective=("ref_share_objective", "mean"),
              mean_fast_share_objective=("fast_share_objective", "mean"),
          )
          .sort_values(["class", "instance", "decoder", "searcher"])
    )
    sum_path = out_dir / "end2end_ref_vs_fast_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # short report
    total = len(df)
    ok = int(df["ok_value_exact"].sum()) if total else 0
    rep = []
    rep.append("End-to-end ref vs fast objective benchmark\n")
    rep.append(f"- runs: {total}")
    rep.append(f"- ok(best_value exact match): {ok} ({(ok/total*100 if total else 0):.1f}%)")
    rep.append("")
    rep.append("Mean speedups (overall):")
    for (sea,), g in df.groupby(["searcher"]):
        rep.append(f"- {sea}: total={g['speedup_total'].mean():.3f}x, objective={g['speedup_objective'].mean():.3f}x, "
                   f"ref_share_obj={g['ref_share_objective'].mean()*100:.1f}%")
    rep_path = out_dir / "end2end_ref_vs_fast_report.txt"
    rep_path.write_text("\n".join(rep), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    # ===== CONFIG =====
    MODE = "single"  # "single" | "dir"

    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    IN_DIR = Path(r"data/extended/j60.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 10  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/end2end_objective_compare")

    T = 50
    SEEDS = list(range(1, 11))

    DECODER = "SSGS"  # "SSGS" | "PSGS" | "PSGS_greedy"
    SEARCHERS = ["SA", "HC"]     # add "RLS" if your updated RLS is available/stable
    RUN_RLS = False             # safety switch if your RLS import/name differs

    INCLUDE_DUMMY_COSTS = False

    # budget (keeps runtimes sane)
    MAX_PROFIT_EVALS: Optional[int] = 10_000

    # SA params
    SA_ITERS = 200_000
    SA_TRIES = 30
    SA_NEIGHBOR = "insert"
    SA_T0 = 2e5
    SA_ALPHA = 0.9998
    SA_TMIN = 1.0

    # HC params
    HC_ITERS = 200_000
    HC_TRIES = 50
    HC_NEIGHBOR = "insert"

    # RLS params
    RLS_ITERS = 200_000
    RLS_TRIES = 20

    # PSGS_greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0
    # ==================

    if RUN_RLS and "RLS" not in SEARCHERS:
        SEARCHERS = SEARCHERS + ["RLS"]

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)} decoder={DECODER} searchers={SEARCHERS} T={T}")

    run_end2end(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,
        decoder_name=DECODER,
        searchers=SEARCHERS,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,
        max_profit_evals=MAX_PROFIT_EVALS,

        sa_iters=SA_ITERS,
        sa_neighbor=SA_NEIGHBOR,
        sa_tries=SA_TRIES,
        sa_T0=SA_T0,
        sa_alpha=SA_ALPHA,
        sa_Tmin=SA_TMIN,

        hc_iters=HC_ITERS,
        hc_neighbor=HC_NEIGHBOR,
        hc_tries=HC_TRIES,

        rls_iters=RLS_ITERS,
        rls_tries=RLS_TRIES,

        greedy_min_score=GREEDY_MIN_SCORE,
        greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")
