from __future__ import annotations
import random

from collections import deque
from typing import Dict, List, Iterable, Optional


class CycleError(ValueError):
    """Граф содержит цикл, топологическая сортировка невозможна."""

def topo_sort(project, *, prefer: str = "smallest") -> List[int]:
    """
    Топологическая сортировка (Kahn).
    prefer:
      - "smallest": брать узел с минимальным id (стабильнее для тестов)
      - "fifo": обычная очередь (быстрее)
    """
    # indegree
    indeg: Dict[int, int] = {j: len(project.predecessors.get(j, [])) for j in project.tasks.keys()}

    if prefer == "smallest":
        # держим список кандидатов, каждый раз выбираем минимальный
        ready = sorted([j for j, d in indeg.items() if d == 0])
        order: List[int] = []

        while ready:
            v = ready.pop(0)  # smallest
            order.append(v)

            for u in project.successors.get(v, []):
                indeg[u] -= 1
                if indeg[u] == 0:
                    # вставить так, чтобы список оставался отсортированным
                    # (без bisect тоже ок для j30/j60/j120)
                    ready.append(u)
                    ready.sort()

        if len(order) != len(indeg):
            raise CycleError("Precedence graph has a cycle (topo_sort failed).")

        return order

    elif prefer == "fifo":
        q = deque([j for j, d in indeg.items() if d == 0])
        order: List[int] = []

        while q:
            v = q.popleft()
            order.append(v)

            for u in project.successors.get(v, []):
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)

        if len(order) != len(indeg):
            raise CycleError("Precedence graph has a cycle (topo_sort failed).")

        return order

    else:
        raise ValueError(f"Unknown prefer='{prefer}'")

def is_topological_order(project, order: Iterable[int]) -> bool:
    """Быстрая проверка, что порядок не нарушает precedence."""
    pos = {}
    for i, j in enumerate(order):
        pos[j] = i

    # все ли задачи присутствуют
    if set(pos.keys()) != set(project.tasks.keys()):
        return False

    for u, succs in project.successors.items():
        pu = pos[u]
        for v in succs:
            if pu > pos[v]:
                return False
    return True

def order_without_dummies(project, order: List[int]) -> List[int]:
    """Убрать supersource/supersink, если они у тебя есть."""
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)
    out = []
    for j in order:
        if src is not None and j == src:
            continue
        if snk is not None and j == snk:
            continue
        out.append(j)
    return out

def random_topo_sort(project, *, seed: int | None = None) -> List[int]: 
    """
    Случайный топологический порядок (Kahn + случайный выбор из ready).
    seed — для воспроизводимости.
    """
    rnd = random.Random(seed)

    indeg: Dict[int, int] = {j: len(project.predecessors.get(j, [])) for j in project.tasks.keys()}
    ready = [j for j, d in indeg.items() if d == 0]
    order: List[int] = []

    while ready:
        v = rnd.choice(ready)
        ready.remove(v)
        order.append(v)

        for u in project.successors.get(v, []):
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)

    if len(order) != len(indeg):
        raise CycleError("Precedence graph has a cycle (random_topo_sort failed).")

    return order

def random_topo_sort_fixed_ends(project, *, seed: int | None = None) -> List[int]:
    """
    Случайная топологическая сортировка, но:
    - source_id (если есть) ставим первым
    - sink_id (если есть) стараемся держать последним
    """
    rnd = random.Random(seed)

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    nodes = list(project.tasks.keys())
    indeg: Dict[int, int] = {j: len(project.predecessors.get(j, [])) for j in nodes}
    ready = [j for j, d in indeg.items() if d == 0]

    order: List[int] = []

    # 1) принудительно ставим source первым, если он есть и готов
    if src is not None and src in indeg:
        if indeg[src] != 0:
            raise CycleError(f"source_id={src} has indegree={indeg[src]} (unexpected)")
        if src in ready:
            ready.remove(src)
        order.append(src)
        for u in project.successors.get(src, []):
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)

    # 2) основной цикл: выбираем случайно из ready, но sink держим на конец
    while ready:
        candidates = ready

        # если sink среди готовых и есть другие варианты — не берём sink сейчас
        if snk is not None and snk in ready and len(ready) > 1:
            candidates = [x for x in ready if x != snk]

        v = rnd.choice(candidates)
        ready.remove(v)
        order.append(v)

        for u in project.successors.get(v, []):
            indeg[u] -= 1
            if indeg[u] == 0:
                ready.append(u)

    if len(order) != len(nodes):
        raise CycleError("Precedence graph has a cycle (random_topo_sort_fixed_ends failed).")

    # если по какой-то причине sink не оказался последним — можно проверить
    if snk is not None and snk in order and order[-1] != snk:
        # Это возможно, если sink становится готовым слишком рано и остаётся единственным кандидатом.
        # В большинстве PSPLIB он всё равно будет последним.
        pass

    return order