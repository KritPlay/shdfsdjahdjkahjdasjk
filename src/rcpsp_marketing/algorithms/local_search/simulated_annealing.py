from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Literal

from rcpsp_marketing.core.precedence import is_topological_order

# оба варианта одной и той же цели
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert", "mixed"]


@dataclass(slots=True)
class SAResult:
    best_order: List[int]
    best_value: float
    best_obj: object

    iters: int
    accepted: int
    improved_best: int
    last_value: float

    profit_evals: int
    stopped_by_budget: bool

    # histories (downsampled by log_every)
    history_it: Optional[List[int]] = None
    history_temp: Optional[List[float]] = None
    history_best: Optional[List[float]] = None
    history_cur: Optional[List[float]] = None
    history_accept: Optional[List[int]] = None          # 1 accepted else 0
    history_delta: Optional[List[float]] = None         # cand-cur (0 if no cand)
    history_selected: Optional[List[int]] = None        # len(selected)
    history_makespan: Optional[List[int]] = None        # schedule.makespan
    history_no_neighbor: Optional[List[int]] = None     # 1 if no valid neighbor found
    history_eval_ms: Optional[List[float]] = None       # время оценки цели (ms), если включишь


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def simulated_annealing(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int = 0,
    iters: int = 50_000,

    # temperature schedule
    T0: float = 1.0,
    Tmin: float = 1e-6,
    alpha: float = 0.9995,

    # neighborhood
    neighbor: NeighborType = "swap",          # "swap" | "insert" | "mixed"
    tries_per_iter: int = 20,                 # attempts to find valid neighbor
    p_swap: float = 0.8,                      # only for mixed: P(use swap)
    insert_max_shift: Optional[int] = None,   # local insert window (e.g. 10)

    # compute budget (limit number of objective evaluations)
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",       # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,  # если хочешь подать свою функцию напрямую
    include_dummy_costs: bool = False,

    # logging
    keep_history: bool = False,
    log_every: int = 200,         # store one point each N iterations
    log_eval_time: bool = False,  # писать время оценки (чтобы сравнить ref vs fast)
) -> SAResult:
    """
    SA for priority list.

    decode_fn(proj, order, T) -> object with fields:
      - schedule
      - selected (iterable job ids)

    objective:
      - "ref": evaluate_profit_over_horizon
      - "fast": evaluate_profit_over_horizon_fast
      - or pass objective_fn=...

    Budget:
      profit evaluation == one call to objective_fn(...)
    """
    import time  # локально, чтобы не тащить всегда

    rnd = random.Random(seed)

    # choose objective
    if objective_fn is None:
        if objective == "ref":
            objective_fn = evaluate_profit_over_horizon
        elif objective == "fast":
            objective_fn = evaluate_profit_over_horizon_fast
        else:
            raise ValueError(f"Unknown objective='{objective}'. Use 'ref' or 'fast' or pass objective_fn=...")

    if neighbor == "mixed" and not (0.0 <= p_swap <= 1.0):
        raise ValueError("p_swap must be in [0, 1] for neighbor='mixed'")

    cur_order = list(start_order)

    # histories (downsampled)
    hist_it = [] if keep_history else None
    hist_temp = [] if keep_history else None
    hist_best = [] if keep_history else None
    hist_cur = [] if keep_history else None
    hist_accept = [] if keep_history else None
    hist_delta = [] if keep_history else None
    hist_selected = [] if keep_history else None
    hist_makespan = [] if keep_history else None
    hist_no_neighbor = [] if keep_history else None
    hist_eval_ms = [] if (keep_history and log_eval_time) else None

    def log_point(
        it: int, *,
        temp_: float, best_: float, cur_: float,
        acc: int, delta_: float,
        sel: int, ms: int, no_nb: int,
        eval_ms: float = 0.0,
    ) -> None:
        if not keep_history:
            return
        if log_every <= 1 or it % log_every == 0:
            hist_it.append(it)
            hist_temp.append(float(temp_))
            hist_best.append(float(best_))
            hist_cur.append(float(cur_))
            hist_accept.append(int(acc))
            hist_delta.append(float(delta_))
            hist_selected.append(int(sel))
            hist_makespan.append(int(ms))
            hist_no_neighbor.append(int(no_nb))
            if hist_eval_ms is not None:
                hist_eval_ms.append(float(eval_ms))

    def propose_neighbor(order: List[int]) -> Optional[List[int]]:
        src = getattr(proj, "source_id", None)
        snk = getattr(proj, "sink_id", None)

        def movable_indices(cur: List[int]) -> List[int]:
            # пересчитываем каждый раз от текущего порядка (важно для insert)
            return [idx for idx, j in enumerate(cur) if j != src and j != snk]

        def do_swap(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False
            i, k = rnd.sample(mov, 2)
            _swap(cand, i, k)
            return True

        def do_insert(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False

            i = rnd.choice(mov)

            if insert_max_shift is None:
                k = rnd.choice([x for x in mov if x != i])
            else:
                lo = max(0, i - insert_max_shift)
                hi = min(len(cand) - 1, i + insert_max_shift)
                window = [x for x in mov if x != i and lo <= x <= hi]
                if not window:
                    k = rnd.choice([x for x in mov if x != i])
                else:
                    k = rnd.choice(window)

            # move i -> k
            job = cand.pop(i)
            if k > i:
                k -= 1
            cand.insert(k, job)
            return True

        for _ in range(tries_per_iter):
            cand = list(order)
            mov = movable_indices(cand)
            if len(mov) < 2:
                return None

            # выбираем тип шага
            if neighbor == "swap":
                ok = do_swap(cand, mov)
            elif neighbor == "insert":
                ok = do_insert(cand, mov)
            elif neighbor == "mixed":
                step = "swap" if rnd.random() < p_swap else "insert"
                ok = do_swap(cand, mov) if step == "swap" else do_insert(cand, mov)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if not ok:
                continue

            if is_topological_order(proj, cand):
                return cand

        return None

    # baseline eval
    cur_res = decode_fn(proj, cur_order, T)

    t0 = time.perf_counter()
    cur_obj = objective_fn(
        proj,
        cur_res.schedule,
        selected_jobs=cur_res.selected,
        T=T,
        include_dummy_costs=include_dummy_costs,
    )
    eval_ms0 = (time.perf_counter() - t0) * 1000.0

    profit_evals = 1
    cur_val = float(cur_obj.value)

    best_order = list(cur_order)
    best_obj = cur_obj
    best_val = cur_val

    accepted = 0
    improved_best = 0
    stopped_by_budget = False

    temp = float(T0)

    # initial log
    log_point(
        0,
        temp_=temp,
        best_=best_val,
        cur_=cur_val,
        acc=1,
        delta_=0.0,
        sel=len(getattr(cur_res, "selected", [])),
        ms=int(getattr(cur_res, "schedule").makespan),
        no_nb=0,
        eval_ms=eval_ms0,
    )

    for it in range(1, iters + 1):
        if temp < Tmin:
            break

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_by_budget = True
            break

        cand_order = propose_neighbor(cur_order)

        if cand_order is None:
            temp *= alpha
            log_point(
                it,
                temp_=temp,
                best_=best_val,
                cur_=cur_val,
                acc=0,
                delta_=0.0,
                sel=len(getattr(cur_res, "selected", [])),
                ms=int(getattr(cur_res, "schedule").makespan),
                no_nb=1,
                eval_ms=0.0,
            )
            continue

        cand_res = decode_fn(proj, cand_order, T)

        t1 = time.perf_counter()
        cand_obj = objective_fn(
            proj,
            cand_res.schedule,
            selected_jobs=cand_res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        eval_ms = (time.perf_counter() - t1) * 1000.0

        profit_evals += 1
        cand_val = float(cand_obj.value)
        delta = cand_val - cur_val

        # accept rule
        if delta >= 0.0:
            accept = True
        else:
            p = math.exp(delta / max(1e-12, temp))
            accept = (rnd.random() < p)

        acc_flag = 1 if accept else 0

        if accept:
            cur_order = cand_order
            cur_res = cand_res
            cur_obj = cand_obj
            cur_val = cand_val
            accepted += 1

            if cand_val > best_val:
                best_val = cand_val
                best_obj = cand_obj
                best_order = list(cand_order)
                improved_best += 1

        temp *= alpha

        log_point(
            it,
            temp_=temp,
            best_=best_val,
            cur_=cur_val,
            acc=acc_flag,
            delta_=delta,
            sel=len(getattr(cur_res, "selected", [])),
            ms=int(getattr(cur_res, "schedule").makespan),
            no_nb=0,
            eval_ms=eval_ms,
        )

    return SAResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        accepted=accepted,
        improved_best=improved_best,
        last_value=cur_val,
        profit_evals=profit_evals,
        stopped_by_budget=stopped_by_budget,
        history_it=hist_it,
        history_temp=hist_temp,
        history_best=hist_best,
        history_cur=hist_cur,
        history_accept=hist_accept,
        history_delta=hist_delta,
        history_selected=hist_selected,
        history_makespan=hist_makespan,
        history_no_neighbor=hist_no_neighbor,
        history_eval_ms=hist_eval_ms,
    )
