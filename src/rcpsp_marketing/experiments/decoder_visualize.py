from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html
from rcpsp_marketing.core.scheduling import serial_sgs_selective

# optional decoders
try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective
except Exception:
    parallel_sgs_selective = None

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective_greedy
except Exception:
    parallel_sgs_selective_greedy = None


DecoderFn = Callable[..., Any]


def _run_and_save(
    *,
    name: str,
    decoder_fn: DecoderFn,
    proj: Any,
    order: list[int],
    T: int,
    out_dir: Path,
    **kwargs,
) -> Path:
    t0 = perf_counter()
    res = decoder_fn(proj, order, T=T, include_dummies=True, **kwargs)
    dt = perf_counter() - t0

    obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

    title = (
        f"{proj.name} | {name} | T={T} | "
        f"value={obj.value:_.2f} | sel={len(res.selected)} skip={len(res.skipped)} | "
        f"time={dt:.4f}s"
    )
    fig = plot_schedule_gantt(proj, res.schedule, selected=res.selected, title=title, T=T)

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )
    out_path = out_dir / f"{proj.name}_T{T}_{safe_name}.html"
    return save_schedule_html(fig, out_path)


def visualize_decoder_packings(
    proj: Any,
    *,
    order: list[int],
    T: int,
    out_dir: str | Path = "data/experiments/viz",
    greedy_min_score: float = -1e18,
    greedy_unlock_weight: float = 0.0,
) -> dict[str, Path]:
    """
    Сохраняет Gantt-упаковку (schedule packing) для каждого доступного декодера.
    Возвращает: {decoder_name: html_path}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # 1) SSGS
    saved["SSGS selective"] = _run_and_save(
        name="SSGS selective",
        decoder_fn=serial_sgs_selective,
        proj=proj,
        order=order,
        T=T,
        out_dir=out_dir,
    )

    # 2) PSGS (если есть)
    if parallel_sgs_selective is not None:
        saved["PSGS selective"] = _run_and_save(
            name="PSGS selective",
            decoder_fn=parallel_sgs_selective,
            proj=proj,
            order=order,
            T=T,
            out_dir=out_dir,
            include_sink=False,
        )

    # 3) PSGS greedy-score (если есть)
    if parallel_sgs_selective_greedy is not None:
        saved["PSGS greedy-score"] = _run_and_save(
            name="PSGS greedy-score",
            decoder_fn=parallel_sgs_selective_greedy,
            proj=proj,
            order=order,
            T=T,
            out_dir=out_dir,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return saved
