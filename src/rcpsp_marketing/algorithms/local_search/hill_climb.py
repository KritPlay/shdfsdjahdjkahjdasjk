from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Any, Literal

from rcpsp_marketing.core.precedence import is_topological_order
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert", "mixed"]


@dataclass(slots=True)
class HCResult:
    best_order: List[int]
    best_value: float
    best_obj: object
    iters: int                 # requested iters
    iters_done: int            # executed iters
    accepted: int              # how many improving moves accepted
    profit_evals: int          # calls to objective function
    stopped_reason: str        # "budget" | "iters" | "no_improve"


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def hill_climb(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int = 0,
    iters: int = 50_000,

    # neighborhood
    neighbor: NeighborType = "swap",          # "swap" | "insert" | "mixed"
    tries_per_iter: int = 50,                 # attempts to find a valid neighbor per iteration
    p_swap: float = 0.8,                      # only for mixed
    insert_max_shift: Optional[int] = None,   # local insert window (e.g. 10)

    # budget
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",      # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,
    include_dummy_costs: bool = False,
) -> HCResult:
    """
    Hill Climbing (first-improvement) on priority list.

    - Only accepts strictly improving moves.
    - Stops when:
        * budget reached (max_profit_evals),
        * iters reached,
        * no improving neighbor found in this iteration (local optimum).

    Budget counts ONLY calls to objective_fn (includes initial evaluation).
    """

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

    rnd = random.Random(seed)

    profit_evals = 0

    def _profit_eval(order: List[int]) -> tuple[float, object]:
        nonlocal profit_evals
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")

        res = decode_fn(proj, order, T)
        obj = objective_fn(
            proj,
            res.schedule,
            selected_jobs=res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        profit_evals += 1
        return float(obj.value), obj

    def propose_neighbor(order: List[int]) -> Optional[List[int]]:
        """
        Try up to tries_per_iter to find a topologically valid neighbor.
        """
        src = getattr(proj, "source_id", None)
        snk = getattr(proj, "sink_id", None)

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
                k = rnd.choice(window) if window else rnd.choice([x for x in mov if x != i])

            if i == k:
                return False

            job = cand.pop(i)
            if k > i:
                k -= 1
            cand.insert(k, job)
            return True

        for _ in range(tries_per_iter):
            cand = list(order)
            mov = _make_movable_indices(proj, cand)
            if len(mov) < 2:
                return None

            if neighbor == "swap":
                ok = do_swap(cand, mov)
            elif neighbor == "insert":
                ok = do_insert(cand, mov)
            elif neighbor == "mixed":
                step = "swap" if rnd.random() < p_swap else "insert"
                ok = do_swap(cand, mov) if step == "swap" else do_insert(cand, mov)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if ok and is_topological_order(proj, cand):
                return cand

        return None

    # initial evaluation
    cur_order = list(start_order)
    try:
        cur_val, cur_obj = _profit_eval(cur_order)
    except StopIteration:
        return HCResult(
            best_order=list(cur_order),
            best_value=float("-inf"),
            best_obj=None,
            iters=iters,
            iters_done=0,
            accepted=0,
            profit_evals=profit_evals,
            stopped_reason="budget",
        )

    best_order = list(cur_order)
    best_val = cur_val
    best_obj = cur_obj

    accepted = 0
    iters_done = 0
    stopped_reason = ""

    for it in range(iters):
        iters_done = it + 1

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        cand_order = propose_neighbor(cur_order)
        if cand_order is None:
            stopped_reason = "no_improve"
            break

        try:
            cand_val, cand_obj = _profit_eval(cand_order)
        except StopIteration:
            stopped_reason = "budget"
            break

        if cand_val > cur_val:
            # accept (first improvement)
            cur_order = cand_order
            cur_val = cand_val
            cur_obj = cand_obj
            accepted += 1

            if cand_val > best_val:
                best_val = cand_val
                best_obj = cand_obj
                best_order = list(cand_order)
        else:
            # для HC "плохие" соседи не принимаем; но продолжаем итерации
            # (если хочешь "до первого улучшения" внутри итерации — тогда возвращаемся к старой схеме)
            pass

    if not stopped_reason:
        stopped_reason = "iters"

    return HCResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        iters_done=iters_done,
        accepted=accepted,
        profit_evals=profit_evals,
        stopped_reason=stopped_reason,
    )
