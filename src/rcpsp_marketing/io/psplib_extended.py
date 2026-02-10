"""
Парсер расширенного формата с маркетинговыми метриками.

План:
- на базе результата `load_psplib_base` добавлять поля из *_with_metrics.sm;
- мэппить их в модели из `rcpsp_marketing.data.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import re

from rcpsp_marketing.data.models import Project, ServiceMetrics, MetricEffectsPct
from rcpsp_marketing.io.psplib_base import PSPLibBaseParser, _find_block, _clean_line, _re_int, _re_float


def parse_service_metrics(content: str) -> Optional[ServiceMetrics]:
    block = _find_block(
        content,
        r"SERVICE\s+METRICS\s*\(INITIAL\)",
        r"(RESOURCE\s+COSTS|TASK\s+COSTS|METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    if not block:
        return None

    def find_float(name: str, default: float) -> float:
        m = re.search(rf"\b{name}\s*=\s*({_re_float.pattern})", block, flags=re.I)
        return float(m.group(1)) if m else default

    def find_int(name: str, default: int) -> int:
        m = re.search(rf"\b{name}\s*=\s*({_re_int.pattern})", block, flags=re.I)
        return int(m.group(1)) if m else default

    return ServiceMetrics(
        LT_0=find_float("LT_0", 0.0),
        AC_0=find_int("AC_0", 0),
        CPC_0=find_float("CPC_0", 0.0),
        LCR_0=find_float("LCR_0", 0.0),
        PCR_0=find_float("PCR_0", 0.0),
        CAC_0=find_float("CAC_0", 0.0),
        MARGIN0=find_float("MARGIN0", 0.0),
    )


def parse_resource_costs(content: str) -> Dict[int, float]:
    block = _find_block(
        content,
        r"RESOURCE\s+COSTS\s*\(per\s+unit\s+per\s+time\)",
        r"(TASK\s+COSTS|METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    costs: Dict[int, float] = {}
    if not block:
        return costs
    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln:
            continue
        m = re.search(r"R(\d+)\s*=\s*(" + _re_float.pattern + r")", ln, flags=re.I)
        if m:
            costs[int(m.group(1))] = float(m.group(2))
    return costs


def parse_task_costs(content: str) -> Dict[int, float]:
    block = _find_block(
        content,
        r"TASK\s+COSTS\s*\(total\)",
        r"(METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    costs: Dict[int, float] = {}
    if not block:
        return costs
    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln:
            continue
        m = re.search(r"^\s*(\d+)\s*:\s*(" + _re_float.pattern + r")", ln)
        if m:
            costs[int(m.group(1))] = float(m.group(2))
    return costs


def parse_metric_changes(content: str) -> Dict[int, tuple[str, MetricEffectsPct]]:
    block = _find_block(
        content,
        r"METRIC\s+CHANGES\s*\(per\s+job.*?\)",
        r"(END\s+OF\s+FILE|^\*{3,})"
    )
    out: Dict[int, tuple[str, MetricEffectsPct]] = {}
    if not block:
        return out

    # формат генератора: "job : type  AC% LT% CPC% LCR% PCR%" :contentReference[oaicite:6]{index=6}
    line_re = re.compile(
        r"^\s*(\d+)\s*:\s*([A-Za-z_]+)\s*"
        r"([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)",
        flags=re.I | re.M
    )

    for m in line_re.finditer(block):
        j = int(m.group(1))
        jt = m.group(2).lower()
        eff = MetricEffectsPct(
            AC=float(m.group(3)),
            LT=float(m.group(4)),
            CPC=float(m.group(5)),
            LCR=float(m.group(6)),
            PCR=float(m.group(7)),
        )
        out[j] = (jt, eff)

    return out


@dataclass(slots=True)
class PSPLibExtendedParser:
    """
    Парсер extended-файлов: raw PSPLIB + блоки SERVICE/RESOURCE COSTS/TASK COSTS/METRIC CHANGES.
    """

    base: PSPLibBaseParser = field(default_factory=PSPLibBaseParser)

    def parse(self, path: str | Path, *, name: str = "") -> Project:
        path = Path(path)

        # 1) базовый проект
        proj = self.base.parse(path, name=name or path.stem)

        # 2) читаем всё содержимое для расширенных блоков
        content = path.read_text(encoding="utf-8", errors="replace")

        proj.metrics0 = parse_service_metrics(content)
        proj.resource_costs = parse_resource_costs(content)

        task_costs = parse_task_costs(content)
        changes = parse_metric_changes(content)

        # 3) приклеиваем к задачам
        for j, t in proj.tasks.items():
            if j in task_costs:
                t.total_cost = float(task_costs[j])
            if j in changes:
                jt, eff = changes[j]
                t.job_type = jt
                t.effects_pct = eff

        return proj
