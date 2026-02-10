"""
Парсер базового формата PSPLib (без маркетинговых расширений).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

from rcpsp_marketing.data.models import Project, Task


_re_int = re.compile(r"-?\d+")
_re_float = re.compile(r"-?\d+(?:\.\d+)?")


def _clean_line(line: str) -> str:
    return line.split("#")[0].strip()


def _find_block(content: str, header_regex: str, stop_regex: str) -> str:
    """Текст после заголовка до стоп-маркера или конца."""
    s = re.search(header_regex, content, flags=re.I)
    if not s:
        return ""
    tail = content[s.end():]
    stop = re.search(stop_regex, tail, flags=re.I | re.M)
    return tail[:stop.start()] if stop else tail


def parse_jobs_count(content: str) -> int:
    for pat in [r"jobs\s*\(.*?\)\s*:\s*(\d+)", r"jobs\s*:\s*(\d+)", r"number\s+of\s+activities.*?:\s*(\d+)"]:
        m = re.search(pat, content, flags=re.I)
        if m:
            return int(m.group(1))
    # fallback: обычно 32 для j30 (30 задач + source/sink)
    ints = [int(x) for x in _re_int.findall(content)]
    return max(ints) if ints else 0


def parse_num_renewable_resources(content: str) -> int:
    # формат может быть:
    # "renewable resources :  4"  или "- renewable :  4   R"
    m = re.search(r"renewable\s*resources.*?:\s*(\d+)", content, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r"-\s*renewable\s*:?\s*(\d+)", content, flags=re.I)
    if m2:
        return int(m2.group(1))
    # fallback
    return 4


def parse_project_info(content: str) -> Dict[str, int]:
    """
    Вытащим полезное (если есть):
    horizon, rel.date, duedate, tardcost, MPM-Time.
    Не критично для SSGS, но полезно для штрафов/экспорта.
    """
    meta: Dict[str, int] = {}

    mh = re.search(r"horizon\s*:\s*(\d+)", content, flags=re.I)
    if mh:
        meta["horizon"] = int(mh.group(1))

    block = _find_block(
        content,
        r"PROJECT\s+INFORMATION",
        r"(PRECEDENCE\s+RELATIONS|REQUESTS/DURATIONS|RESOURCEAVAILABILITIES|^\*{3,})"
    )
    if block:
        # обычно там строка вида: "1  30  0  38  26  38"
        #                    pronr #jobs rel duedate tardcost mpm-time
        lines = [ln for ln in map(_clean_line, block.splitlines()) if _re_int.search(ln)]
        for ln in lines:
            ints = [int(x) for x in _re_int.findall(ln)]
            if len(ints) >= 6:
                meta["rel_date"] = ints[2]
                meta["duedate"] = ints[3]
                meta["tardcost"] = ints[4]
                meta["mpm_time"] = ints[5]
                break
    return meta


def parse_precedence_arcs(content: str) -> List[Tuple[int, int]]:
    """
    PRECEDENCE RELATIONS:
    jobnr  #modes  #successors   successors...
    Дуги j -> succ_k.
    """
    block = _find_block(
        content,
        r"PRECEDENCE\s+RELATIONS",
        r"(REQUESTS/DURATIONS|RESOURCEAVAILABILITIES|PROJECT|^\*{3,}|SERVICE\s+METRICS|RESOURCE\s+COSTS|TASK\s+COSTS|METRIC\s+CHANGES)"
    )
    arcs: List[Tuple[int, int]] = []
    if not block:
        return arcs

    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln or not _re_int.search(ln):
            continue
        # пропускаем заголовки
        low = ln.lower()
        if "jobnr" in low or "successors" in low:
            continue

        ints = [int(x) for x in _re_int.findall(ln)]
        if len(ints) >= 3:
            j = ints[0]
            k = ints[2]
            succs = ints[3:3 + k] if len(ints) >= 3 + k else []
            for s in succs:
                arcs.append((j, s))

    return arcs


def parse_requests_durations(content: str, nR: int) -> Dict[int, Task]:
    """
    REQUESTS/DURATIONS:
    jobnr mode duration R1..Rn
    duration = 3-е число. :contentReference[oaicite:1]{index=1}
    """
    block = _find_block(
        content,
        r"REQUESTS/DURATIONS",
        r"(PRECEDENCE|RESOURCEAVAILABILITIES|PROJECT|^\*{3,}|SERVICE\s+METRICS|RESOURCE\s+COSTS|TASK\s+COSTS|METRIC\s+CHANGES)"
    )
    tasks: Dict[int, Task] = {}

    if not block:
        return tasks

    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln:
            continue

        low = ln.lower()
        if "jobnr" in low or "mode" in low or "duration" in low:
            continue
        if set(ln) <= set("- "):
            continue
        if not _re_int.search(ln):
            continue

        ints = [int(x) for x in _re_int.findall(ln)]
        if len(ints) >= 3 + nR:
            job = ints[0]
            dur = ints[2]
            reqs = ints[3:3 + nR]
            tasks[job] = Task(
                id=job,
                duration=dur,
                req={i + 1: reqs[i] for i in range(nR)},
            )

    return tasks


def parse_resource_availabilities(content: str, nR: int) -> Dict[int, int]:
    """
    RESOURCEAVAILABILITIES:
      R 1  R 2 ...
       12  13 ...
    Берём последние nR чисел из блока (устойчиво). :contentReference[oaicite:2]{index=2}
    """
    block = _find_block(
        content,
        r"RESOURCEAVAILABILITIES",
        r"(PRECEDENCE|REQUESTS/DURATIONS|PROJECT|^\*{3,}|SERVICE\s+METRICS|RESOURCE\s+COSTS|TASK\s+COSTS|METRIC\s+CHANGES)"
    )
    if not block:
        return {}

    all_ints = [int(x) for x in _re_int.findall(block)]
    if len(all_ints) >= nR:
        vals = all_ints[-nR:]
        return {i + 1: vals[i] for i in range(nR)}
    return {}


def _build_adj(n_jobs: int, arcs: List[Tuple[int, int]]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    succ: Dict[int, List[int]] = {i: [] for i in range(1, n_jobs + 1)}
    pred: Dict[int, List[int]] = {i: [] for i in range(1, n_jobs + 1)}
    for u, v in arcs:
        succ.setdefault(u, []).append(v)
        pred.setdefault(v, []).append(u)
    for i in succ:
        succ[i].sort()
    for i in pred:
        pred[i].sort()
    return succ, pred


@dataclass(slots=True)
class PSPLibBaseParser:
    """
    Парсер raw PSPLIB single-mode.
    Возвращает Project без маркетинговых расширений.
    """

    def parse(self, path: str | Path, *, name: str = "") -> Project:
        path = Path(path)
        content = path.read_text(encoding="utf-8", errors="replace")

        n_jobs = parse_jobs_count(content)
        nR = parse_num_renewable_resources(content)

        tasks = parse_requests_durations(content, nR)
        arcs = parse_precedence_arcs(content)
        avail = parse_resource_availabilities(content, nR)
        meta = parse_project_info(content)

        # если tasks пустой (редко, но вдруг) — создадим пустые задачи, чтобы project был целостный
        if not tasks and n_jobs > 0:
            for j in range(1, n_jobs + 1):
                tasks[j] = Task(id=j, duration=0, req={})

        succ, pred = _build_adj(n_jobs, arcs)

        proj = Project(
            name=name or path.stem,
            tasks=tasks,
            renewable_avail=avail,
            successors=succ,
            predecessors=pred,
            metrics0=None,
            resource_costs={},
            meta=meta,
            source_id=1,
            sink_id=n_jobs if n_jobs > 0 else 0,
        )
        # полезно сразу ловить ошибки парсинга
        proj.validate(check_dag=True)
        return proj
