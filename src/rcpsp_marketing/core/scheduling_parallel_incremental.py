from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from rcpsp_marketing.data.models import Project, Schedule


@dataclass(slots=True)
class SGSResult:
    schedule: Schedule
    selected: List[int]
    skipped: List[int]


@dataclass(slots=True)
class ParallelSnapshot:
    """
    Состояние PSGS на момент времени t0, если мы "фиксируем" все задачи со start < t0.
    """
    t0: int
    usage: Dict[int, List[int]]          # профиль ресурсов 0..T-1
    start: Dict[int, int]                # только зафиксированные задачи (start < t0)
    finish: Dict[int, int]
    selected_prefix: List[int]           # зафиксированные задачи
    done: Set[int]                       # finish <= t0
    running: List[Tuple[int, int]]       # (finish_time, job) для start < t0 < finish
    remaining: Set[int]                  # все jobs минус зафиксированные (start < t0)


def _build_jobs(project: Project, priority_list: List[int], include_dummies: bool, include_sink: bool) -> List[int]:
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    jobs: List[int] = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        if not include_sink and (snk is not None and j == snk):
            continue
        jobs.append(j)
    return jobs


def _apply_resources_profile(
    project: Project,
    *,
    cap: Dict[int, int],
    usage: Dict[int, List[int]],
    job: int,
    s: int,
    T: int,
) -> None:
    d = project.tasks[job].duration
    if d <= 0:
        return
    req = project.tasks[job].req
    for r, need in req.items():
        if need <= 0:
            continue
        prof = usage[r]
        for tt in range(s, min(T, s + d)):
            prof[tt] += need


def snapshot_parallel_prefix_by_time(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool,
    include_sink: bool,
    base_res: SGSResult,
    t0: int,
) -> ParallelSnapshot:
    """
    Строим snapshot в момент t0, фиксируя все задачи, которые в base_res стартовали раньше t0.
    """
    jobs = _build_jobs(project, priority_list, include_dummies, include_sink)
    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    start_fix: Dict[int, int] = {}
    finish_fix: Dict[int, int] = {}

    # зафиксированные = start < t0
    fixed: Set[int] = set()
    for j, sj in base_res.schedule.start.items():
        if sj < t0:
            fixed.add(j)

    # start/finish только для fixed
    for j in fixed:
        start_fix[j] = int(base_res.schedule.start[j])
        finish_fix[j] = int(base_res.schedule.finish[j])

    # usage набираем по fixed (включая тех, кто "в работе" после t0)
    for j in fixed:
        _apply_resources_profile(project, cap=cap, usage=usage, job=j, s=start_fix[j], T=T)

    done: Set[int] = set()
    running: List[Tuple[int, int]] = []
    for j in fixed:
        fj = finish_fix[j]
        if fj <= t0:
            done.add(j)
        else:
            # dur=0 сюда не попадёт (fj==sj < t0 => fj<=t0)
            running.append((fj, j))

    # remaining = jobs - fixed
    remaining: Set[int] = set(jobs) - fixed

    # selected_prefix: сохраним в том же порядке, как в base_res.selected, но только fixed
    fixed_list = [j for j in base_res.selected if j in fixed]

    return ParallelSnapshot(
        t0=int(t0),
        usage=usage,
        start=start_fix,
        finish=finish_fix,
        selected_prefix=fixed_list,
        done=done,
        running=running,
        remaining=remaining,
    )


def parallel_sgs_selective_resume_from_snapshot(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
    include_sink: bool = False,
    snap: ParallelSnapshot,
) -> SGSResult:
    """
    PSGS selective, но стартуем с состояния snap (время = snap.t0).
    """
    src = getattr(project, "source_id", None)

    # порядок -> позиция (tie-break)
    pos = {j: i for i, j in enumerate(priority_list)}

    jobs = _build_jobs(project, priority_list, include_dummies, include_sink)

    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: prof[:] for r, prof in snap.usage.items()}

    start: Dict[int, int] = dict(snap.start)
    finish: Dict[int, int] = dict(snap.finish)
    selected: List[int] = list(snap.selected_prefix)
    skipped: List[int] = []

    done: Set[int] = set(snap.done)
    remaining: Set[int] = set(snap.remaining)
    running: List[Tuple[int, int]] = list(snap.running)

    # source если он вообще есть и был зафиксирован — уже в start/finish/selected/done.
    # если source не зафиксирован (t0==0), то он будет обработан как обычная dur=0 задача,
    # но в исходном PSGS вы явно добавляли source в 0. Оставим такое же поведение:
    t = int(snap.t0)
    if t == 0 and include_dummies and src is not None and src in project.tasks:
        # только если ещё не зафиксирован
        if src not in start:
            start[src] = 0
            finish[src] = 0
            selected.append(src)
            done.add(src)
        remaining.discard(src)

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

    while t < T:
        # 1) завершить всё, что закончилось к t
        if running:
            still_running = []
            for fj, j in running:
                if fj <= t:
                    done.add(j)
                else:
                    still_running.append((fj, j))
            running = still_running

        # 2) если задачи уже невозможно стартовать -> skip
        for j in list(remaining):
            d = project.tasks[j].duration
            if d > 0 and t > T - d:
                remaining.remove(j)
                skipped.append(j)

        # 3) запуск ready задач в момент t
        changed = True
        while changed:
            changed = False

            eligible = [j for j in remaining if preds_done(j)]
            if not eligible:
                break

            eligible.sort(key=lambda j: pos.get(j, 10**9))

            for j in eligible:
                d = project.tasks[j].duration

                if d == 0:
                    start[j] = t
                    finish[j] = t
                    selected.append(j)
                    done.add(j)
                    remaining.remove(j)
                    changed = True
                    break

                if fits_resources(j, t):
                    start[j] = t
                    finish[j] = t + d
                    apply_resources(j, t)
                    selected.append(j)
                    remaining.remove(j)
                    running.append((t + d, j))
                    changed = True
                    continue

        # 4) jump to next event
        if running:
            t_next = min(fj for fj, _ in running)
            if t_next <= t:
                t += 1
            else:
                t = t_next
        else:
            # никто не выполняется -> остальное недостижимо
            for j in sorted(remaining, key=lambda x: pos.get(x, 10**9)):
                skipped.append(j)
            break

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped)
