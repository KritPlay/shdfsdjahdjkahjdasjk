from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rcpsp_marketing.data.models import Project, Schedule


@dataclass(slots=True)
class ImproveResult:
    schedule: Schedule
    moved: int
    notes: str = ""


def _scheduled_jobs(schedule: Schedule, selected: Optional[Iterable[int]] = None) -> List[int]:
    if selected is None:
        return sorted(schedule.start.keys())
    sel = [int(j) for j in selected if int(j) in schedule.start]
    return sorted(sel)


def _build_usage_profile(
    project: Project,
    schedule: Schedule,
    *,
    jobs: List[int],
    T: int,
) -> Dict[int, List[int]]:
    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    for j in jobs:
        s = schedule.start[j]
        f = schedule.finish[j]
        if f <= s:
            continue
        req = project.tasks[j].req
        for r, need in req.items():
            if need <= 0:
                continue
            prof = usage[r]
            for t in range(s, f):
                prof[t] += need
    return usage


def _fits(
    project: Project,
    usage: Dict[int, List[int]],
    *,
    j: int,
    s: int,
    T: int,
) -> bool:
    d = project.tasks[j].duration
    if d <= 0:
        return True
    if s < 0 or s + d > T:
        return False

    req = project.tasks[j].req
    for r, need in req.items():
        if need <= 0:
            continue
        prof = usage[r]
        cap = project.renewable_avail.get(r, 0)
        for t in range(s, s + d):
            if prof[t] + need > cap:
                return False
    return True


def _remove_from_profile(project: Project, usage: Dict[int, List[int]], *, j: int, s: int) -> None:
    d = project.tasks[j].duration
    if d <= 0:
        return
    req = project.tasks[j].req
    for r, need in req.items():
        if need <= 0:
            continue
        prof = usage[r]
        for t in range(s, s + d):
            prof[t] -= need


def _add_to_profile(project: Project, usage: Dict[int, List[int]], *, j: int, s: int) -> None:
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


def left_shift(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    hide_dummies: bool = True,
) -> ImproveResult:
    """
    Левое уплотнение:
    - идём по задачам в порядке их текущего старта
    - каждую задачу пытаемся сдвинуть максимально влево (раньше),
      сохраняя precedence и ресурсы
    """
    if T is None:
        T = max(schedule.makespan, 0)
    T = int(T)

    jobs = _scheduled_jobs(schedule, selected_jobs)

    # при желании убираем фиктивные
    if hide_dummies:
        jobs = [j for j in jobs if j not in (project.source_id, project.sink_id)]

    # сортируем по текущему старту
    jobs.sort(key=lambda j: (schedule.start[j], j))

    # копия расписания (не портим оригинал)
    new_start = dict(schedule.start)
    new_finish = dict(schedule.finish)
    new_sched = Schedule(start=new_start, finish=new_finish, feasible=schedule.feasible, notes=schedule.notes)

    usage = _build_usage_profile(project, new_sched, jobs=_scheduled_jobs(new_sched, selected_jobs), T=T)

    moved = 0

    for j in jobs:
        s0 = new_sched.start[j]
        d = project.tasks[j].duration

        # нижняя граница по предшественникам (только тем, кто реально в расписании)
        preds = project.predecessors.get(j, [])
        est = 0
        for p in preds:
            if p in new_sched.finish:
                est = max(est, int(new_sched.finish[p]))

        # dur=0: можем поставить в est (ресурсы не трогаем)
        if d <= 0:
            if est < s0:
                new_sched.start[j] = est
                new_sched.finish[j] = est
                moved += 1
            continue

        # временно вынимаем задачу из профиля и ищем более ранний старт
        _remove_from_profile(project, usage, j=j, s=s0)

        best_s = s0
        for s in range(est, s0):
            if _fits(project, usage, j=j, s=s, T=T):
                best_s = s
                break

        if best_s != s0:
            new_sched.start[j] = best_s
            new_sched.finish[j] = best_s + d
            moved += 1

        # вставляем обратно по новому месту
        _add_to_profile(project, usage, j=j, s=new_sched.start[j])

    return ImproveResult(schedule=new_sched, moved=moved, notes="left_shift")


def right_shift(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    hide_dummies: bool = True,
) -> ImproveResult:
    """
    Правое выравнивание (right-justification):
    - идём по задачам в порядке убывания finish
    - каждую задачу пытаемся сдвинуть максимально вправо,
      не выходя за T и не нарушая precedence и ресурсы
    """
    if T is None:
        T = max(schedule.makespan, 0)
    T = int(T)

    jobs_all = _scheduled_jobs(schedule, selected_jobs)
    jobs = jobs_all

    if hide_dummies:
        jobs = [j for j in jobs if j not in (project.source_id, project.sink_id)]

    # по убыванию finish
    jobs.sort(key=lambda j: (schedule.finish[j], j), reverse=True)

    new_start = dict(schedule.start)
    new_finish = dict(schedule.finish)
    new_sched = Schedule(start=new_start, finish=new_finish, feasible=schedule.feasible, notes=schedule.notes)

    usage = _build_usage_profile(project, new_sched, jobs=jobs_all, T=T)

    moved = 0

    # помогаем себе: учитывать только succ, которые реально присутствуют
    scheduled_set: Set[int] = set(jobs_all)

    for j in jobs:
        s0 = new_sched.start[j]
        d = project.tasks[j].duration

        # нижняя граница (нельзя раньше предшественников) — хотя мы двигаем вправо,
        # всё равно нельзя заехать "на предшественников"
        preds = project.predecessors.get(j, [])
        lb = 0
        for p in preds:
            if p in new_sched.finish:
                lb = max(lb, int(new_sched.finish[p]))

        # верхняя граница по потомкам и горизонту
        ub = T - d
        succs = project.successors.get(j, [])
        for u in succs:
            if u in scheduled_set:
                ub = min(ub, int(new_sched.start[u]) - d)

        if ub <= s0:
            continue  # сдвинуть вправо некуда

        if d <= 0:
            # dur=0 можно сдвинуть до ub
            new_sched.start[j] = ub
            new_sched.finish[j] = ub
            moved += 1
            continue

        _remove_from_profile(project, usage, j=j, s=s0)

        best_s = s0
        for s in range(ub, lb - 1, -1):
            if _fits(project, usage, j=j, s=s, T=T):
                best_s = s
                break

        if best_s != s0:
            new_sched.start[j] = best_s
            new_sched.finish[j] = best_s + d
            moved += 1

        _add_to_profile(project, usage, j=j, s=new_sched.start[j])

    return ImproveResult(schedule=new_sched, moved=moved, notes="right_shift")
