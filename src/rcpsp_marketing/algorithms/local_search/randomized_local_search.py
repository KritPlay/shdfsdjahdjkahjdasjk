from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Any, Callable, Literal

from rcpsp_marketing.core.precedence import is_topological_order
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert"]


@dataclass(slots=True)
class RLSResult:
    best_order: List[int]
    best_value: float
    best_obj: object
    iters: int
    accepted: int

    profit_evals: int = 0
    iters_done: int = 0
    stopped_reason: str = ""


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _insert(order: List[int], i: int, k: int) -> None:
    """
    Move element at index i to index k (Python-style insert after pop).
    If i < k, after pop indices shift left, so target becomes k-1.
    """
    if i == k:
        return
    x = order.pop(i)
    if i < k:
        k -= 1
    order.insert(k, x)


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def randomized_local_search(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    seed: int = 0,
    iters: int = 2000,
    tries_per_iter: int = 1,

    # neighbor type
    neighbor: NeighborType = "swap",
    insert_max_shift: Optional[int] = None,  # например 10, чтобы insert был локальным

    # compute budget
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",  # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,
    include_dummy_costs: bool = False,
) -> RLSResult:
    """
    Randomized Local Search (first improvement):
    - сосед: swap или insert (move)
    - принимаем, если стало лучше
    """

    # choose objective
    if objective_fn is None:
        if objective == "ref":
            objective_fn = evaluate_profit_over_horizon
        elif objective == "fast":
            objective_fn = evaluate_profit_over_horizon_fast
        else:
            raise ValueError(f"Unknown objective='{objective}'. Use 'ref' or 'fast' or pass objective_fn=...")

    rnd = random.Random(seed)

    movable_idx = _make_movable_indices(proj, start_order)
    if len(movable_idx) < 2:
        raise ValueError("Not enough movable jobs to perform moves (need >=2).")

    cur_order = list(start_order)

    profit_evals = 0
    accepted = 0
    iters_done = 0
    stopped_reason = ""

    def _profit_eval(order: List[int]) -> tuple[float, object]:
        nonlocal profit_evals

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")

        res = serial_sgs_selective(proj, order, T=T, include_dummies=True)
        obj = objective_fn(
            proj,
            res.schedule,
            selected_jobs=res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        profit_evals += 1
        return float(obj.value), obj

    def _sample_two_indices() -> tuple[int, int]:
        """
        Берём два индекса для соседа.
        Для insert можно ограничить дальность сдвига insert_max_shift (локальный insert).
        """
        if neighbor == "swap" or insert_max_shift is None:
            return tuple(rnd.sample(movable_idx, 2))  # type: ignore

        # local insert: выбираем i, потом k близко к i
        i = rnd.choice(movable_idx)
        # кандидаты k в окне
        lo = max(0, i - insert_max_shift)
        hi = min(len(cur_order) - 1, i + insert_max_shift)
        candidates = [k for k in range(lo, hi + 1) if k != i and cur_order[k] not in (getattr(proj, "source_id", None), getattr(proj, "sink_id", None))]
        if not candidates:
            # fallback
            k = rnd.choice([x for x in movable_idx if x != i])
            return i, k
        return i, rnd.choice(candidates)

    # --- initial evaluation
    try:
        cur_val, cur_obj = _profit_eval(cur_order)
    except StopIteration:
        return RLSResult(
            best_order=list(cur_order),
            best_value=float("-inf"),
            best_obj=None,
            iters=iters,
            accepted=0,
            profit_evals=profit_evals,
            iters_done=0,
            stopped_reason="budget",
        )

    best_order = list(cur_order)
    best_obj = cur_obj
    best_val = cur_val

    # --- main loop
    for it in range(iters):
        iters_done = it + 1

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        improved = False

        for _ in range(tries_per_iter):
            if max_profit_evals is not None and profit_evals >= max_profit_evals:
                stopped_reason = "budget"
                break

            i, k = _sample_two_indices()
            cand = list(cur_order)

            if neighbor == "swap":
                _swap(cand, i, k)
            elif neighbor == "insert":
                _insert(cand, i, k)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if not is_topological_order(proj, cand):
                continue

            try:
                cand_val, cand_obj = _profit_eval(cand)
            except StopIteration:
                stopped_reason = "budget"
                break

            if cand_val > cur_val:
                cur_order = cand
                cur_val = cand_val
                cur_obj = cand_obj
                accepted += 1
                improved = True

                if cand_val > best_val:
                    best_val = cand_val
                    best_obj = cand_obj
                    best_order = list(cand)

                break  # first improvement

        if stopped_reason == "budget":
            break

        # если нет улучшения — остаёмся на месте
        if not improved:
            pass

    if not stopped_reason:
        stopped_reason = "iters"

    return RLSResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        accepted=accepted,
        profit_evals=profit_evals,
        iters_done=iters_done,
        stopped_reason=stopped_reason,
    )
