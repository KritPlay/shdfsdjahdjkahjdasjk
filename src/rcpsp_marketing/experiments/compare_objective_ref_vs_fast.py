from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)
from rcpsp_marketing.core.scheduling import (
    serial_sgs_selective,
    parallel_sgs_selective,
    parallel_sgs_selective_greedy,
)


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


def _time_objective(
    fn: Callable[..., Any],
    *,
    proj: Any,
    schedule: Any,
    selected: Any,
    T: int,
    include_dummy_costs: bool,
    repeats: int,
) -> Tuple[Any, float]:
    """
    Возвращает (obj, avg_ms_per_call) для fn на фиксированном (schedule, selected).
    """
    # warmup (JIT нет, но прогреваем кэш/страницы/ветвления)
    obj = fn(proj, schedule, selected_jobs=selected, T=T, include_dummy_costs=include_dummy_costs)

    t0 = perf_counter()
    for _ in range(repeats):
        obj = fn(proj, schedule, selected_jobs=selected, T=T, include_dummy_costs=include_dummy_costs)
    dt = perf_counter() - t0
    avg_ms = (dt / max(1, repeats)) * 1000.0
    return obj, avg_ms


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

    decs = {
        "SSGS": DecoderSpec("SSGS", decode_ssgs),
        "PSGS": DecoderSpec("PSGS", decode_psgs),
        "PSGS_greedy": DecoderSpec("PSGS_greedy", decode_psgs_greedy),
    }
    return decs


# =========================
# Experiment
# =========================

def run_experiment(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],
    decoder_name: str,
    repeats: int,
    include_dummy_costs: bool,
    greedy_min_score: float,
    greedy_unlock_weight: float,
    abs_tol: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)
    if decoder_name not in decoders:
        raise ValueError(f"Unknown decoder_name='{decoder_name}'. Choose one of: {list(decoders.keys())}")
    decoder = decoders[decoder_name]

    parser = PSPLibExtendedParser()

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)

        print(f"[instance] {cls} / {inst}")

        for s in seeds:
            order = random_topo_sort_fixed_ends(proj, seed=s)

            # Decode ONCE
            dec_t0 = perf_counter()
            res = decoder.decode_fn(proj, order, T)
            dec_dt = perf_counter() - dec_t0

            # Evaluate both on the SAME (schedule, selected)
            obj_ref, ms_ref = _time_objective(
                evaluate_profit_over_horizon,
                proj=proj,
                schedule=res.schedule,
                selected=res.selected,
                T=T,
                include_dummy_costs=include_dummy_costs,
                repeats=repeats,
            )
            obj_fast, ms_fast = _time_objective(
                evaluate_profit_over_horizon_fast,
                proj=proj,
                schedule=res.schedule,
                selected=res.selected,
                T=T,
                include_dummy_costs=include_dummy_costs,
                repeats=repeats,
            )

            v_ref = float(obj_ref.value)
            v_fast = float(obj_fast.value)
            r_ref = float(obj_ref.revenue)
            r_fast = float(obj_fast.revenue)
            c_ref = float(obj_ref.cost)
            c_fast = float(obj_fast.cost)

            dv = v_fast - v_ref
            dr = r_fast - r_ref
            dc = c_fast - c_ref

            rel = dv / (abs(v_ref) + 1e-12)

            ok = (abs(dv) <= abs_tol) and (abs(dr) <= abs_tol) and (abs(dc) <= abs_tol)

            speedup = (ms_ref / ms_fast) if ms_fast > 0 else float("inf")

            rows.append({
                "class": cls,
                "instance": inst,
                "instance_path": str(fp),
                "T": int(T),
                "seed": int(s),
                "decoder": decoder.name,

                "decode_time_ms": dec_dt * 1000.0,

                "ref_value": v_ref,
                "fast_value": v_fast,
                "diff_value": dv,
                "rel_diff_value": rel,

                "ref_revenue": r_ref,
                "fast_revenue": r_fast,
                "diff_revenue": dr,

                "ref_cost": c_ref,
                "fast_cost": c_fast,
                "diff_cost": dc,

                "ref_ms_per_call": ms_ref,
                "fast_ms_per_call": ms_fast,
                "speedup": speedup,

                "repeats": int(repeats),
                "include_dummy_costs": bool(include_dummy_costs),
                "abs_tol": float(abs_tol),
                "ok": bool(ok),
            })

    df = pd.DataFrame(rows)

    # --- save runs
    runs_path = out_dir / "objective_compare_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # --- summary
    def _std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    summary = (
        df.groupby(["class", "instance", "decoder"], as_index=False)
          .agg(
              runs=("seed", "count"),
              ok_rate=("ok", "mean"),
              max_abs_diff_value=("diff_value", lambda x: float(x.abs().max())),
              max_abs_diff_revenue=("diff_revenue", lambda x: float(x.abs().max())),
              max_abs_diff_cost=("diff_cost", lambda x: float(x.abs().max())),
              mean_ref_ms=("ref_ms_per_call", "mean"),
              mean_fast_ms=("fast_ms_per_call", "mean"),
              mean_speedup=("speedup", "mean"),
              std_speedup=("speedup", _std),
              mean_decode_ms=("decode_time_ms", "mean"),
          )
          .sort_values(["class", "instance", "decoder"])
    )

    sum_path = out_dir / "objective_compare_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # --- short report
    total = len(df)
    ok_cnt = int(df["ok"].sum()) if total else 0
    ok_rate = (ok_cnt / total) if total else 0.0

    overall = {
        "total_runs": total,
        "ok_runs": ok_cnt,
        "ok_rate": ok_rate,
        "max_abs_diff_value": float(df["diff_value"].abs().max()) if total else 0.0,
        "max_abs_diff_revenue": float(df["diff_revenue"].abs().max()) if total else 0.0,
        "max_abs_diff_cost": float(df["diff_cost"].abs().max()) if total else 0.0,
        "mean_ref_ms": float(df["ref_ms_per_call"].mean()) if total else 0.0,
        "mean_fast_ms": float(df["fast_ms_per_call"].mean()) if total else 0.0,
        "mean_speedup": float(df["speedup"].mean()) if total else 0.0,
    }

    rep_lines = []
    rep_lines.append("Objective ref vs fast comparison\n")
    rep_lines.append(f"- runs: {overall['total_runs']}")
    rep_lines.append(f"- ok:   {overall['ok_runs']} ({overall['ok_rate']*100:.1f}%)  tol={abs_tol}")
    rep_lines.append("")
    rep_lines.append("Max absolute diffs:")
    rep_lines.append(f"- value:   {overall['max_abs_diff_value']:.6g}")
    rep_lines.append(f"- revenue: {overall['max_abs_diff_revenue']:.6g}")
    rep_lines.append(f"- cost:    {overall['max_abs_diff_cost']:.6g}")
    rep_lines.append("")
    rep_lines.append("Timing (ms per call, averaged):")
    rep_lines.append(f"- ref:  {overall['mean_ref_ms']:.4f} ms")
    rep_lines.append(f"- fast: {overall['mean_fast_ms']:.4f} ms")
    rep_lines.append(f"- speedup: {overall['mean_speedup']:.2f}x")
    rep_lines.append("")
    rep_lines.append("Files:")
    rep_lines.append(f"- {runs_path.name}")
    rep_lines.append(f"- {sum_path.name}")

    rep_path = out_dir / "objective_compare_report.txt"
    rep_path.write_text("\n".join(rep_lines), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    # ===== CONFIG =====
    MODE = "dir"  # "single" | "dir"

    # single
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # dir
    IN_DIR = Path(r"data/extended/j120.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 30  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/objective_compare")

    # horizon and seeds
    T = 50
    SEEDS = list(range(1, 11))

    # decoder used to generate schedules (objective is measured on those schedules)
    DECODER = "SSGS"  # "SSGS" | "PSGS" | "PSGS_greedy"

    # timing: objective will be called N times on same schedule (average per call)
    REPEATS = 200

    # objective options
    INCLUDE_DUMMY_COSTS = False

    # PSGS_greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0

    # accuracy check tolerance
    ABS_TOL = 1e-9
    # ==================

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)} decoder={DECODER} repeats={REPEATS} T={T}")

    run_experiment(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,
        decoder_name=DECODER,
        repeats=REPEATS,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,
        greedy_min_score=GREEDY_MIN_SCORE,
        greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
        abs_tol=ABS_TOL,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")