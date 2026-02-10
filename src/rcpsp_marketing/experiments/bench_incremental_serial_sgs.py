from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, is_topological_order
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.scheduling_incremental import SerialSSGSIncremental


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


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _insert(order: List[int], i: int, k: int) -> None:
    if i == k:
        return
    x = order.pop(i)
    if k > i:
        k -= 1
    order.insert(k, x)


def _pick_neighbor(
    proj: Any,
    rnd: random.Random,
    cur_order: List[int],
    *,
    movable_idx: List[int],
    neighbor: str,
    tries: int,
    insert_max_shift: Optional[int],
) -> Optional[Tuple[List[int], int]]:
    """
    Возвращает (cand_order, changed_from_idx) или None, если не нашли валидного.
    changed_from_idx = min(i,k) — корректная граница пересчёта префикса для swap/insert.
    """
    n = len(cur_order)

    for _ in range(tries):
        cand = list(cur_order)

        if neighbor == "swap":
            i, k = rnd.sample(movable_idx, 2)
            _swap(cand, i, k)
            changed = min(i, k)

        elif neighbor == "insert":
            i = rnd.choice(movable_idx)
            if insert_max_shift is None:
                k = rnd.choice([x for x in movable_idx if x != i])
            else:
                lo = max(0, i - insert_max_shift)
                hi = min(n - 1, i + insert_max_shift)
                window = [x for x in movable_idx if x != i and lo <= x <= hi]
                k = rnd.choice(window) if window else rnd.choice([x for x in movable_idx if x != i])

            if i == k:
                continue
            _insert(cand, i, k)
            changed = min(i, k)

        else:
            raise ValueError("neighbor must be 'swap' or 'insert'")

        if is_topological_order(proj, cand):
            return cand, changed

    return None


# =========================
# Experiment
# =========================

@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    fn: Callable[..., Any]


def run_experiment(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],

    # neighbor generation
    neighbor: str,                 # "swap" | "insert"
    steps: int,                    # how many neighbor evaluations per (instance, seed)
    tries_per_step: int,           # attempts to find topo-valid neighbor
    insert_max_shift: Optional[int],

    # incremental
    checkpoint_every: int,         # e.g. 10

    # objective selection
    objective: str,                # "ref" | "fast"
    include_dummy_costs: bool,

    # correctness
    abs_tol: float,

    # simulate real search (commit accepted moves)
    commit_prob: float,            # 0.0 => never commit (pure eval benchmark), e.g. 0.15 to simulate SA/HC
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PSPLibExtendedParser()

    # pick objective
    if objective == "ref":
        obj_spec = ObjectiveSpec("ref", evaluate_profit_over_horizon)
    elif objective == "fast":
        obj_spec = ObjectiveSpec("fast", evaluate_profit_over_horizon_fast)
    else:
        raise ValueError("objective must be 'ref' or 'fast'")

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)
        print(f"[instance] {cls} / {inst}")

        for s in seeds:
            rnd = random.Random(s)

            cur_order = random_topo_sort_fixed_ends(proj, seed=s)
            movable_idx = _make_movable_indices(proj, cur_order)

            inc = SerialSSGSIncremental(proj, T=T, include_dummies=True, checkpoint_every=checkpoint_every)
            inc.set_current(cur_order)

            ok_cnt = 0
            bad_cnt = 0
            no_nb = 0

            # separate timing buckets
            full_decode_time = 0.0
            full_obj_time = 0.0
            incr_decode_time = 0.0
            incr_obj_time = 0.0

            evals = 0

            for _ in range(steps):
                got = _pick_neighbor(
                    proj, rnd, cur_order,
                    movable_idx=movable_idx,
                    neighbor=neighbor,
                    tries=tries_per_step,
                    insert_max_shift=insert_max_shift,
                )
                if got is None:
                    no_nb += 1
                    continue

                cand_order, changed_from = got

                # --- FULL: decode + objective (split) ---
                t0 = perf_counter()
                full_res = serial_sgs_selective(proj, cand_order, T=T, include_dummies=True)
                t1 = perf_counter()
                full_obj = obj_spec.fn(
                    proj,
                    full_res.schedule,
                    selected_jobs=full_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t2 = perf_counter()

                full_decode_time += (t1 - t0)
                full_obj_time += (t2 - t1)

                # --- INCR: resume + objective (split) ---
                t3 = perf_counter()
                inc_res = inc.decode_neighbor(cand_order, changed_from_order_idx=changed_from)
                t4 = perf_counter()
                inc_obj = obj_spec.fn(
                    proj,
                    inc_res.schedule,
                    selected_jobs=inc_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t5 = perf_counter()

                incr_decode_time += (t4 - t3)
                incr_obj_time += (t5 - t4)

                evals += 1

                # correctness
                same_sched = (full_res.schedule.start == inc_res.schedule.start) and (full_res.schedule.finish == inc_res.schedule.finish)
                same_lists = (full_res.selected == inc_res.selected) and (full_res.skipped == inc_res.skipped)
                dv = float(inc_obj.value) - float(full_obj.value)
                ok = same_sched and same_lists and (abs(dv) <= abs_tol)

                if ok:
                    ok_cnt += 1
                else:
                    bad_cnt += 1

                # simulate “accepted” moves (optional)
                if commit_prob > 0.0 and rnd.random() < commit_prob:
                    cur_order = cand_order
                    movable_idx = _make_movable_indices(proj, cur_order)
                    inc.set_current(cur_order)

            full_time = full_decode_time + full_obj_time
            incr_time = incr_decode_time + incr_obj_time

            full_ms = (full_time / max(1, evals)) * 1000.0
            incr_ms = (incr_time / max(1, evals)) * 1000.0
            speedup = (full_ms / incr_ms) if incr_ms > 0 else float("inf")

            full_decode_ms = (full_decode_time / max(1, evals)) * 1000.0
            full_obj_ms = (full_obj_time / max(1, evals)) * 1000.0
            incr_decode_ms = (incr_decode_time / max(1, evals)) * 1000.0
            incr_obj_ms = (incr_obj_time / max(1, evals)) * 1000.0

            rows.append({
                "class": cls,
                "instance": inst,
                "instance_path": str(fp),
                "T": int(T),
                "seed": int(s),

                "neighbor": neighbor,
                "steps": int(steps),
                "tries_per_step": int(tries_per_step),
                "insert_max_shift": (None if insert_max_shift is None else int(insert_max_shift)),

                "checkpoint_every": int(checkpoint_every),

                "objective": obj_spec.name,
                "include_dummy_costs": bool(include_dummy_costs),
                "commit_prob": float(commit_prob),

                "evals": int(evals),
                "no_neighbor": int(no_nb),
                "ok": int(ok_cnt),
                "bad": int(bad_cnt),

                # end-to-end
                "full_ms_per_eval": float(full_ms),
                "incr_ms_per_eval": float(incr_ms),
                "speedup": float(speedup),

                # breakdown
                "full_decode_ms_per_eval": float(full_decode_ms),
                "full_obj_ms_per_eval": float(full_obj_ms),
                "incr_decode_ms_per_eval": float(incr_decode_ms),
                "incr_obj_ms_per_eval": float(incr_obj_ms),
            })

    df = pd.DataFrame(rows)

    # --- save runs
    runs_path = out_dir / "decode_incremental_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # --- summary (simple)
    summary = (
        df.groupby(["class", "neighbor", "checkpoint_every", "objective"], as_index=False)
          .agg(
              runs=("seed", "count"),
              evals=("evals", "sum"),
              ok=("ok", "sum"),
              bad=("bad", "sum"),
              mean_full_ms=("full_ms_per_eval", "mean"),
              mean_incr_ms=("incr_ms_per_eval", "mean"),
              mean_speedup=("speedup", "mean"),
          )
          .sort_values(["class", "neighbor", "checkpoint_every", "objective"])
    )

    sum_path = out_dir / "decode_incremental_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # --- weighted overall report (like your style) ---
    if len(df) > 0:
        df["full_total_ms"] = df["evals"] * df["full_ms_per_eval"]
        df["incr_total_ms"] = df["evals"] * df["incr_ms_per_eval"]

        df["full_decode_total_ms"] = df["evals"] * df["full_decode_ms_per_eval"]
        df["full_obj_total_ms"] = df["evals"] * df["full_obj_ms_per_eval"]
        df["incr_decode_total_ms"] = df["evals"] * df["incr_decode_ms_per_eval"]
        df["incr_obj_total_ms"] = df["evals"] * df["incr_obj_ms_per_eval"]

        total_evals = int(df["evals"].sum())

        full_total = float(df["full_total_ms"].sum())
        incr_total = float(df["incr_total_ms"].sum())
        speedup_total = (full_total / incr_total) if incr_total > 0 else float("inf")

        full_decode_avg = float(df["full_decode_total_ms"].sum()) / max(1, total_evals)
        full_obj_avg = float(df["full_obj_total_ms"].sum()) / max(1, total_evals)
        incr_decode_avg = float(df["incr_decode_total_ms"].sum()) / max(1, total_evals)
        incr_obj_avg = float(df["incr_obj_total_ms"].sum()) / max(1, total_evals)

        decode_speedup = (full_decode_avg / incr_decode_avg) if incr_decode_avg > 0 else float("inf")
        obj_speedup = (full_obj_avg / incr_obj_avg) if incr_obj_avg > 0 else float("inf")

        full_share_decode = full_decode_avg / max(1e-12, (full_decode_avg + full_obj_avg))
        incr_share_decode = incr_decode_avg / max(1e-12, (incr_decode_avg + incr_obj_avg))

        ok_total = int(df["ok"].sum())
        bad_total = int(df["bad"].sum())
        ok_rate = ok_total / max(1, (ok_total + bad_total))

        full_total_avg = full_total / max(1, total_evals)
        incr_total_avg = incr_total / max(1, total_evals)

    else:
        total_evals = 0
        full_total = incr_total = 0.0
        speedup_total = 0.0
        full_decode_avg = full_obj_avg = incr_decode_avg = incr_obj_avg = 0.0
        decode_speedup = obj_speedup = 0.0
        full_share_decode = incr_share_decode = 0.0
        ok_total = bad_total = 0
        ok_rate = 0.0
        full_total_avg = incr_total_avg = 0.0

    rep_lines: List[str] = []
    rep_lines.append("End-to-end full vs incremental decode benchmark\n")
    rep_lines.append(f"- runs: {len(df)}")
    rep_lines.append(f"- ok(exact schedule+lists+value): {ok_total} ({ok_rate*100:.1f}%)")
    rep_lines.append(f"- bad: {bad_total}")
    rep_lines.append(f"- evals(total): {total_evals}")
    rep_lines.append("")

    rep_lines.append("Mean speedups (overall, weighted by evals):")
    rep_lines.append(
        f"- total={speedup_total:.3f}x, decode={decode_speedup:.3f}x, objective={obj_speedup:.3f}x, "
        f"full_share_decode={full_share_decode*100:.1f}%, incr_share_decode={incr_share_decode*100:.1f}%"
    )
    rep_lines.append("")
    rep_lines.append("Timing (ms per eval, weighted):")
    rep_lines.append(f"- full: total={full_total_avg:.4f}, decode={full_decode_avg:.4f}, obj={full_obj_avg:.4f}")
    rep_lines.append(f"- incr: total={incr_total_avg:.4f}, decode={incr_decode_avg:.4f}, obj={incr_obj_avg:.4f}")
    rep_lines.append("")
    rep_lines.append("Files:")
    rep_lines.append(f"- {runs_path.name}")
    rep_lines.append(f"- {sum_path.name}")

    rep_path = out_dir / "decode_incremental_report.txt"
    rep_path.write_text("\n".join(rep_lines), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    MODE = "dir"  # "single" | "dir"

    # single
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # dir
    IN_DIR = Path(r"data/extended/j30.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 30  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/decode_incremental_compare")

    # horizon and seeds
    T = 50
    SEEDS = list(range(1, 11))

    # neighbor benchmark config
    NEIGHBOR = "swap"          # "swap" | "insert"
    STEPS = 2000               # neighbor evals per (instance, seed)
    TRIES_PER_STEP = 30        # try to find topo-valid neighbor
    INSERT_MAX_SHIFT = 10      # None = global insert

    # incremental config
    CHECKPOINT_EVERY = 10      # 5..20 обычно нормально

    # objective
    OBJECTIVE = "fast"         # "ref" | "fast"
    INCLUDE_DUMMY_COSTS = False

    # correctness
    ABS_TOL = 1e-9

    # simulate real search acceptance (0.0 = pure eval benchmark)
    COMMIT_PROB = 0.15

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(
        f"[info] mode={MODE} instances={len(files)} neighbor={NEIGHBOR} steps={STEPS} "
        f"T={T} objective={OBJECTIVE} checkpoint_every={CHECKPOINT_EVERY} commit_prob={COMMIT_PROB}"
    )

    run_experiment(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,

        neighbor=NEIGHBOR,
        steps=STEPS,
        tries_per_step=TRIES_PER_STEP,
        insert_max_shift=INSERT_MAX_SHIFT,

        checkpoint_every=CHECKPOINT_EVERY,

        objective=OBJECTIVE,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,

        abs_tol=ABS_TOL,
        commit_prob=COMMIT_PROB,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")
