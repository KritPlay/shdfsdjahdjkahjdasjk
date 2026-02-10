from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, topo_sort
from rcpsp_marketing.core.scheduling import serial_sgs_selective

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective
except Exception:
    parallel_sgs_selective = None

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective_greedy
except Exception:
    parallel_sgs_selective_greedy = None


DecoderFn = Callable[..., Any]


def _run_one(
    name: str,
    decoder_fn: DecoderFn,
    proj: Any,
    order: list[int],
    T: int,
    **kwargs,
) -> dict[str, Any]:
    t0 = perf_counter()
    res = decoder_fn(proj, order, T=T, include_dummies=True, **kwargs)
    dt = perf_counter() - t0

    obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

    return {
        "decoder": name,
        "value": obj.value,
        "revenue": obj.revenue,
        "cost": obj.cost,
        "selected": len(res.selected),
        "skipped": len(res.skipped),
        "makespan": res.schedule.makespan,
        "time_sec": dt,
    }


def compare_decoders(
    proj: Any,
    *,
    T: int,
    order: list[int] | None = None,
    order_mode: str = "random_fixed_ends",   # "random_fixed_ends" | "topo_smallest"
    seed: int = 42,
    greedy_min_score: float = -1e18,
    greedy_unlock_weight: float = 0.0,
    save_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Сравнивает декодеры на одном и том же priority list (без поисковых методов/улучшателей).
    Возвращает DataFrame, можно сохранить в CSV.
    """

    if order is None:
        if order_mode == "random_fixed_ends":
            order = random_topo_sort_fixed_ends(proj, seed=seed)
        elif order_mode == "topo_smallest":
            order = topo_sort(proj, prefer="smallest")
        else:
            raise ValueError(f"Unknown order_mode={order_mode}")

    rows: list[dict[str, Any]] = []
    rows.append(_run_one("SSGS selective", serial_sgs_selective, proj, order, T))

    if parallel_sgs_selective is not None:
        rows.append(_run_one("PSGS selective", parallel_sgs_selective, proj, order, T, include_sink=False))

    if parallel_sgs_selective_greedy is not None:
        rows.append(_run_one(
            "PSGS greedy-score",
            parallel_sgs_selective_greedy,
            proj, order, T,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        ))

    df = pd.DataFrame(rows).sort_values("value", ascending=False)

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    return df
