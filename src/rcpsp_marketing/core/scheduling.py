from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import heapq
import numpy as np

from rcpsp_marketing.data.models import Project, Schedule
from rcpsp_marketing.core.objective import MetricState 

@dataclass(slots=True)
class SGSResult:
    schedule: Schedule
    selected: List[int]     # реально запланированные задачи (включая source/sink если ты их оставишь)
    skipped: List[int]      # пропущенные


def serial_sgs_selective(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
) -> SGSResult:
    """
    Селективный Serial SSGS:
    - строит параллельное расписание
    - разрешает пропуск задач
    - гарантирует makespan <= T для выбранных задач
    """
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    # какие задачи вообще рассматриваем
    jobs = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        jobs.append(j)

    # дискретный профиль ресурсов 0..T-1
    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    selected: List[int] = []
    skipped: List[int] = []

    done: Set[int] = set()  # выполненные (запланированные)

    def earliest_by_preds(j: int) -> int:
        preds = project.predecessors.get(j, [])
        if not preds:
            return 0
        if any(p not in done for p in preds):
            return -1  # пока недоступна (есть невыполненные предшественники)
        return max(finish[p] for p in preds)

    def fits_resources(j: int, s: int) -> bool:
        d = project.tasks[j].duration
        if s + d > T:
            return False
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            c = cap[r]
            for t in range(s, s + d):
                if prof[t] + need > c:
                    return False
        return True

    def apply_resources(j: int, s: int) -> None:
        d = project.tasks[j].duration
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            for t in range(s, s + d):
                prof[t] += need

    # Основной проход по priority list
    for j in jobs:
        # если не хотим планировать фиктивные — они уже исключены
        est = earliest_by_preds(j)
        if est < 0:
            # недоступна из-за предшественников → пропускаем
            skipped.append(j)
            continue

        d = project.tasks[j].duration
        # если duration=0 — ставим мгновенно (если влезаем)
        if d == 0:
            start[j] = min(est, T)
            finish[j] = min(est, T)
            selected.append(j)
            done.add(j)
            continue

        # ищем самое раннее время старта, где хватает ресурсов
        s = est
        placed = False
        while s + d <= T:
            if fits_resources(j, s):
                start[j] = s
                finish[j] = s + d
                apply_resources(j, s)
                selected.append(j)
                done.add(j)
                placed = True
                break
            s += 1

        if not placed:
            skipped.append(j)

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped)

def parallel_sgs_selective(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
    include_sink: bool = False,
) -> SGSResult:
    """
    Selective Parallel SGS (PSGS), event-driven:
    - идём по времени t (перескакиваем по событиям завершения)
    - на каждом t запускаем максимально возможное число готовых задач (по priority_list)
    - задачи, которые уже не успевают стартовать (t > T-d), пропускаем
    - source (если есть) ставим в t=0 как dur=0
    - sink по умолчанию НЕ пытаемся планировать, т.к. при selective-отборе он часто не станет ready
    """

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    # порядок -> позиция (чтобы сортировать "как в priority list")
    pos = {j: i for i, j in enumerate(priority_list)}

    # какие задачи рассматриваем
    jobs: List[int] = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        if not include_sink and (snk is not None and j == snk):
            continue
        jobs.append(j)

    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    selected: List[int] = []
    skipped: List[int] = []

    done: Set[int] = set()        # завершённые
    scheduled: Set[int] = set()   # уже поставленные (в работе или завершённые)
    remaining: Set[int] = set(jobs)

    # --- helpers ---
    def preds_done(j: int) -> bool:
        return all(p in done for p in project.predecessors.get(j, []))

    def fits_resources(j: int, s: int) -> bool:
        d = project.tasks[j].duration
        if d <= 0:
            return True
        if s + d > T:
            return False
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            c = cap[r]
            for t in range(s, s + d):
                if prof[t] + need > c:
                    return False
        return True

    def apply_resources(j: int, s: int) -> None:
        d = project.tasks[j].duration
        if d <= 0:
            return
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            for t in range(s, s + d):
                prof[t] += need

    # --- initialize time ---
    t = 0

    # source как мгновенная задача (если включены dummy)
    if include_dummies and src is not None and src in project.tasks:
        start[src] = 0
        finish[src] = 0
        selected.append(src)
        scheduled.add(src)
        done.add(src)
        remaining.discard(src)

    # список выполняющихся: (finish_time, job)
    running: List[Tuple[int, int]] = []

    while t < T:
        # 1) завершаем всё, что закончилось к моменту t
        if running:
            still_running = []
            for fj, j in running:
                if fj <= t:
                    done.add(j)
                else:
                    still_running.append((fj, j))
            running = still_running

        # 2) если есть задачи, которые уже невозможно стартовать — сразу skip
        # (в selective это важно, чтобы не зацикливаться)
        for j in list(remaining):
            d = project.tasks[j].duration
            if d > 0 and t > T - d:
                remaining.remove(j)
                skipped.append(j)

        # 3) пытаемся запускать ready задачи в момент t
        changed = True
        while changed:
            changed = False

            eligible = [j for j in remaining if preds_done(j)]
            if not eligible:
                break

            # порядок запуска: по priority list
            eligible.sort(key=lambda j: pos.get(j, 10**9))

            for j in eligible:
                d = project.tasks[j].duration

                # dur=0 — ставим мгновенно и сразу "done" (может открыть других)
                if d == 0:
                    start[j] = t
                    finish[j] = t
                    selected.append(j)
                    scheduled.add(j)
                    done.add(j)
                    remaining.remove(j)
                    changed = True
                    break

                # обычные задачи
                if fits_resources(j, t):
                    start[j] = t
                    finish[j] = t + d
                    apply_resources(j, t)
                    selected.append(j)
                    scheduled.add(j)
                    remaining.remove(j)
                    running.append((t + d, j))
                    changed = True
                    # продолжаем на том же t — вдруг ещё что-то влезет
                    continue

            # если прошли по eligible и ничего не запустили — выходим
            # (до следующего события ресурсы не изменятся)
            # changed останется False

        # 4) переход ко времени следующего события
        if running:
            t_next = min(fj for fj, _ in running)
            # защита от залипания
            if t_next <= t:
                t += 1
            else:
                t = t_next
        else:
            # никто не выполняется -> ничего больше не изменится
            # всё оставшееся считаем пропущенным
            for j in sorted(remaining, key=lambda x: pos.get(x, 10**9)):
                skipped.append(j)
            break

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped)


def parallel_sgs_selective_fast(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
    include_sink: bool = False,
) -> SGSResult:
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    # pos: приоритет (меньше = раньше)
    pos = {j: i for i, j in enumerate(priority_list)}

    # jobs
    jobs: List[int] = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        if not include_sink and (snk is not None and j == snk):
            continue
        jobs.append(j)

    tasks = project.tasks
    preds_map = project.predecessors
    succ_map = project.successors

    # ---- ресурсы -> индексы ----
    cap_dict = project.renewable_avail
    res_ids = list(cap_dict.keys())
    r2i = {r: i for i, r in enumerate(res_ids)}
    cap = np.array([cap_dict[r] for r in res_ids], dtype=np.int32)

    R = len(res_ids)
    usage = np.zeros((R, T), dtype=np.int32)

    # ---- precompute duration + req_list (ri, need) ----
    dur: Dict[int, int] = {}
    req_list: Dict[int, List[Tuple[int, int]]] = {}
    for j in jobs:
        tj = tasks[j]
        d = int(tj.duration)
        dur[j] = d
        lst: List[Tuple[int, int]] = []
        for r, need in tj.req.items():
            if need <= 0:
                continue
            ri = r2i.get(r)
            if ri is None:
                continue
            lst.append((ri, int(need)))
        req_list[j] = lst

    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    selected: List[int] = []
    skipped: List[int] = []

    remaining = set(jobs)
    done = set()

    # ---- indegree по предшественникам (как в твоём preds_done) ----
    indeg: Dict[int, int] = {}
    for j in jobs:
        indeg[j] = len(preds_map.get(j, []))

    # ready heap by pos
    ready: List[Tuple[int, int]] = []
    heapq.heapify(ready)

    def push_ready(j: int) -> None:
        heapq.heappush(ready, (pos.get(j, 10**9), j))

    def mark_done(j: int) -> None:
        done.add(j)
        # уменьшаем indeg у потомков
        for s in succ_map.get(j, []):
            if s in remaining:
                indeg[s] = indeg.get(s, 0) - 1
                if indeg[s] == 0:
                    push_ready(s)

    # source
    if include_dummies and src is not None and src in tasks:
        start[src] = 0
        finish[src] = 0
        selected.append(src)
        if src in remaining:
            remaining.remove(src)
        # source считается done (как у тебя)
        mark_done(src)

    # начальные ready (indeg==0)
    for j in list(remaining):
        if indeg.get(j, 0) == 0:
            push_ready(j)

    def try_place(j: int, s: int) -> bool:
        d = dur[j]
        if d <= 0:
            return True
        if s + d > T:
            return False

        reqs = req_list[j]
        # check
        e = s + d
        for ri, need in reqs:
            seg = usage[ri, s:e]
            if int(seg.max()) + need > int(cap[ri]):
                return False
        # apply
        for ri, need in reqs:
            usage[ri, s:e] += need
        return True

    # running heap: (finish_time, job)
    running: List[Tuple[int, int]] = []
    heapq.heapify(running)

    t = 0
    while t < T:
        # 1) finish tasks up to t
        while running and running[0][0] <= t:
            fj, j = heapq.heappop(running)
            mark_done(j)

        # 2) выкинуть задачи, которые уже не могут стартовать
        # (делаем лениво: когда достаём из ready, проверяем t > T-d)
        # поэтому тут ничего не нужно

        # 3) пытаться запускать на t
        progressed = True
        deferred: List[Tuple[int, int]] = []

        while progressed:
            progressed = False

            # достаём всех ready “на сейчас”
            while ready:
                p, j = heapq.heappop(ready)
                if j not in remaining:
                    continue  # stale
                d = dur[j]
                # если уже поздно стартовать
                if d > 0 and t > T - d:
                    remaining.remove(j)
                    skipped.append(j)
                    continue

                # dur=0 -> мгновенно done, может открыть других (продолжим на том же t)
                if d == 0:
                    start[j] = t
                    finish[j] = t
                    selected.append(j)
                    remaining.remove(j)
                    mark_done(j)
                    progressed = True
                    # после открытия новых ready — продолжим цикл
                    break

                # обычная задача
                if try_place(j, t):
                    start[j] = t
                    finish[j] = t + d
                    selected.append(j)
                    remaining.remove(j)
                    heapq.heappush(running, (t + d, j))
                    progressed = True
                    break
                else:
                    # не влезла по ресурсам — отложим до следующего события
                    deferred.append((p, j))

            # если мы “прервались” из-за прогресса — отложенные вернём и продолжим
            if progressed:
                for item in deferred:
                    heapq.heappush(ready, item)
                deferred.clear()

        # вернуть отложенные (если остались)
        for item in deferred:
            heapq.heappush(ready, item)

        # 4) перейти к следующему событию
        if running:
            t_next = running[0][0]
            if t_next <= t:
                t += 1
            else:
                t = t_next
        else:
            # никто не работает -> всё оставшееся недостижимо
            for j in sorted(remaining, key=lambda x: pos.get(x, 10**9)):
                skipped.append(j)
            break

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped)


def parallel_sgs_selective_greedy(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
    include_sink: bool = False,
    min_score: float = -1e18,
    unlock_weight: float = 0.0,
) -> SGSResult:
    """
    Selective Parallel SGS (PSGS) + greedy packing (score-based):
    - идём по времени t (event-driven)
    - на каждом t берём ready задачи и жадно запускаем по убыванию score
    - score ≈ (remaining_time * Δprofit_rate - cost) / resource_time
    - эффекты применяются по завершению задач (как в objective)
    - sink по умолчанию не планируем (в selective он часто недостижим)
    """

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    # позиция в priority list (как tie-break)
    pos = {j: i for i, j in enumerate(priority_list)}

    # какие задачи рассматриваем
    jobs: List[int] = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        if not include_sink and (snk is not None and j == snk):
            continue
        jobs.append(j)

    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    selected: List[int] = []
    skipped: List[int] = []

    done: Set[int] = set()
    remaining: Set[int] = set(jobs)
    running: List[Tuple[int, int]] = []  # (finish_time, job)

    # состояние метрик на текущем времени t (эффекты по completion)
    if project.metrics0 is None:
        raise ValueError("metrics0 is None: greedy PSGS требует extended-инстанс с SERVICE METRICS (INITIAL).")
    st = MetricState.from_metrics0(project.metrics0)

    def preds_done(j: int) -> bool:
        return all(p in done for p in project.predecessors.get(j, []))

    def fits_resources(j: int, s: int) -> bool:
        d = project.tasks[j].duration
        if d <= 0:
            return True
        if s + d > T:
            return False
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            c = cap[r]
            for tt in range(s, s + d):
                if prof[tt] + need > c:
                    return False
        return True

    def apply_resources(j: int, s: int) -> None:
        d = project.tasks[j].duration
        if d <= 0:
            return
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            for tt in range(s, s + d):
                prof[tt] += need

    def resource_time(j: int) -> float:
        """Нормированная 'трудоёмкость' по ресурсам и длительности."""
        d = project.tasks[j].duration
        if d <= 0:
            return 0.0
        rt = 0.0
        for r, need in project.tasks[j].req.items():
            if need <= 0:
                continue
            c = float(cap.get(r, 1))
            rt += (float(need) / max(1.0, c))
        return rt * float(d)

    def score_job(j: int, t_now: int) -> float:
        """Оценка выгодности запуска j в момент t_now."""
        task = project.tasks[j]
        d = task.duration
        # не успеет завершиться — сразу плохо
        if d > 0 and t_now + d > T:
            return -1e30

        pr0 = st.profit_rate()

        # применим эффекты "мысленно"
        tmp = MetricState(LT=st.LT, AC=st.AC, CPC=st.CPC, LCR=st.LCR, PCR=st.PCR)
        tmp.recompute()
        tmp.apply_effects_pct(task.effects_pct)
        pr1 = tmp.profit_rate()

        delta = pr1 - pr0

        remaining_after_finish = max(0, T - (t_now + d))
        benefit = float(remaining_after_finish) * float(delta)

        # cost платим всегда, если задачу берём
        benefit -= float(task.total_cost)

        # бонус "за открытие" потомков (опционально)
        if unlock_weight > 0.0:
            benefit += unlock_weight * float(len(project.successors.get(j, [])))

        rt = resource_time(j)
        if rt <= 1e-12:
            # dur=0 или нет ресурсов -> просто по benefit
            return benefit

        return benefit / rt

    # source как мгновенная задача
    if include_dummies and src is not None and src in project.tasks:
        start[src] = 0
        finish[src] = 0
        selected.append(src)
        done.add(src)
        remaining.discard(src)

    t = 0
    while t < T:
        # 1) завершить всё, что закончилось к t, и применить эффекты
        if running:
            new_running: List[Tuple[int, int]] = []
            for fj, j in running:
                if fj <= t:
                    done.add(j)
                    st.apply_effects_pct(project.tasks[j].effects_pct)
                else:
                    new_running.append((fj, j))
            running = new_running

        # 2) пропустить то, что уже не стартанёт так, чтобы успеть завершиться
        for j in list(remaining):
            d = project.tasks[j].duration
            if d > 0 and t > T - d:
                remaining.remove(j)
                skipped.append(j)

        # 3) “в этом t” сначала закрываем все dur=0 (они сразу done и меняют st)
        changed_zero = True
        while changed_zero:
            changed_zero = False
            ready0 = [j for j in remaining if preds_done(j) and project.tasks[j].duration == 0]
            if not ready0:
                break
            # сортируем по score (и tie-break по priority_list)
            ready0.sort(key=lambda j: (-score_job(j, t), pos.get(j, 10**9)))
            for j in ready0:
                sc = score_job(j, t)
                if sc < min_score:
                    remaining.remove(j)
                    skipped.append(j)
                    continue
                start[j] = t
                finish[j] = t
                selected.append(j)
                done.add(j)
                remaining.remove(j)
                st.apply_effects_pct(project.tasks[j].effects_pct)  # мгновенно
                changed_zero = True
                break  # обновили done -> может открыть новые dur=0

        # 4) greedy packing для dur>0
        ready = [j for j in remaining if preds_done(j) and project.tasks[j].duration > 0]
        if ready:
            # сортируем по убыванию score, tie-break по priority list
            ready.sort(key=lambda j: (-score_job(j, t), pos.get(j, 10**9)))

            for j in ready:
                sc = score_job(j, t)
                if sc < min_score:
                    # можем просто пропустить (не брать сейчас вообще)
                    # но чтобы не зависнуть — считаем “skipped”
                    remaining.remove(j)
                    skipped.append(j)
                    continue

                if fits_resources(j, t):
                    d = project.tasks[j].duration
                    start[j] = t
                    finish[j] = t + d
                    apply_resources(j, t)
                    selected.append(j)
                    remaining.remove(j)
                    running.append((t + d, j))

        # 5) перейти к следующему событию
        if running:
            t_next = min(fj for fj, _ in running)
            if t_next <= t:
                t += 1
            else:
                t = t_next
        else:
            # ничего не идёт — остальное недостижимо в рамках greedy
            for j in sorted(remaining, key=lambda x: pos.get(x, 10**9)):
                skipped.append(j)
            break

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped)
