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
class SGSCheckpoint:
    # состояние после обработки первых idx jobs (в jobs после фильтрации dummy)
    idx: int
    usage: Dict[int, List[int]]
    start: Dict[int, int]
    finish: Dict[int, int]
    done: Set[int]
    selected: List[int]
    skipped: List[int]


def _deepcopy_usage(usage: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {r: prof[:] for r, prof in usage.items()}


def _build_jobs(project: Project, priority_list: List[int], include_dummies: bool) -> List[int]:
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)
    jobs: List[int] = []
    for j in priority_list:
        if not include_dummies and (j == src or j == snk):
            continue
        jobs.append(j)
    return jobs


def _order_index_to_jobs_index(
    project: Project, priority_list: List[int], include_dummies: bool, order_idx: int
) -> int:
    # если dummy включены — индексация совпадает
    if include_dummies:
        return max(0, min(order_idx, len(priority_list)))

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)
    cnt = 0
    for i in range(min(order_idx, len(priority_list))):
        j = priority_list[i]
        if j == src or j == snk:
            continue
        cnt += 1
    return cnt


def serial_sgs_selective_with_checkpoints(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool = True,
    checkpoint_every: int = 10,
) -> Tuple[SGSResult, List[SGSCheckpoint]]:
    """
    Selective Serial SGS + чекпоинты состояния каждые checkpoint_every jobs.
    """
    jobs = _build_jobs(project, priority_list, include_dummies)

    cap = project.renewable_avail
    usage: Dict[int, List[int]] = {r: [0] * T for r in cap}

    start: Dict[int, int] = {}
    finish: Dict[int, int] = {}
    selected: List[int] = []
    skipped: List[int] = []
    done: Set[int] = set()

    def earliest_by_preds(j: int) -> int:
        preds = project.predecessors.get(j, [])
        if not preds:
            return 0
        if any(p not in done for p in preds):
            return -1
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

    cps: List[SGSCheckpoint] = []
    cps.append(
        SGSCheckpoint(
            idx=0,
            usage=_deepcopy_usage(usage),
            start=dict(start),
            finish=dict(finish),
            done=set(done),
            selected=list(selected),
            skipped=list(skipped),
        )
    )

    for idx, j in enumerate(jobs, start=1):
        est = earliest_by_preds(j)
        if est < 0:
            skipped.append(j)
        else:
            d = project.tasks[j].duration
            if d == 0:
                start[j] = min(est, T)
                finish[j] = min(est, T)
                selected.append(j)
                done.add(j)
            else:
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

        if checkpoint_every > 0 and (idx % checkpoint_every == 0):
            cps.append(
                SGSCheckpoint(
                    idx=idx,
                    usage=_deepcopy_usage(usage),
                    start=dict(start),
                    finish=dict(finish),
                    done=set(done),
                    selected=list(selected),
                    skipped=list(skipped),
                )
            )

    if cps and cps[-1].idx != len(jobs):
        cps.append(
            SGSCheckpoint(
                idx=len(jobs),
                usage=_deepcopy_usage(usage),
                start=dict(start),
                finish=dict(finish),
                done=set(done),
                selected=list(selected),
                skipped=list(skipped),
            )
        )

    sched = Schedule(start=start, finish=finish, feasible=True)
    return SGSResult(schedule=sched, selected=selected, skipped=skipped), cps


def serial_sgs_selective_resume(
    project: Project,
    priority_list: List[int],
    *,
    T: int,
    include_dummies: bool,
    checkpoint: SGSCheckpoint,
) -> SGSResult:
    """
    Пересчёт ТОЛЬКО хвоста начиная с checkpoint.idx (в jobs).
    """
    jobs = _build_jobs(project, priority_list, include_dummies)

    cap = project.renewable_avail
    usage = _deepcopy_usage(checkpoint.usage)
    start = dict(checkpoint.start)
    finish = dict(checkpoint.finish)
    selected = list(checkpoint.selected)
    skipped = list(checkpoint.skipped)
    done = set(checkpoint.done)

    def earliest_by_preds(j: int) -> int:
        preds = project.predecessors.get(j, [])
        if not preds:
            return 0
        if any(p not in done for p in preds):
            return -1
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

    for j in jobs[checkpoint.idx:]:
        est = earliest_by_preds(j)
        if est < 0:
            skipped.append(j)
            continue

        d = project.tasks[j].duration
        if d == 0:
            start[j] = min(est, T)
            finish[j] = min(est, T)
            selected.append(j)
            done.add(j)
            continue

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


def _pick_checkpoint(cps: List[SGSCheckpoint], jobs_k: int) -> SGSCheckpoint:
    best = cps[0]
    for cp in cps:
        if cp.idx <= jobs_k:
            best = cp
        else:
            break
    return best


@dataclass(slots=True)
class SerialSSGSIncremental:
    """
    Кэш для selective serial SGS: базовое решение + чекпоинты.
    decode_neighbor() пересчитывает только хвост.
    """
    project: Project
    T: int
    include_dummies: bool = True
    checkpoint_every: int = 10

    _cur_order: Optional[List[int]] = None
    _cur_res: Optional[SGSResult] = None
    _cur_cps: Optional[List[SGSCheckpoint]] = None

    def set_current(self, order: List[int]) -> SGSResult:
        res, cps = serial_sgs_selective_with_checkpoints(
            self.project,
            order,
            T=self.T,
            include_dummies=self.include_dummies,
            checkpoint_every=self.checkpoint_every,
        )
        self._cur_order = list(order)
        self._cur_res = res
        self._cur_cps = cps
        return res

    def decode_neighbor(self, cand_order: List[int], *, changed_from_order_idx: int) -> SGSResult:
        if self._cur_cps is None:
            res, _ = serial_sgs_selective_with_checkpoints(
                self.project, cand_order, T=self.T, include_dummies=self.include_dummies, checkpoint_every=0
            )
            return res

        jobs_k = _order_index_to_jobs_index(
            self.project, cand_order, self.include_dummies, changed_from_order_idx
        )
        cp = _pick_checkpoint(self._cur_cps, jobs_k)
        return serial_sgs_selective_resume(
            self.project,
            cand_order,
            T=self.T,
            include_dummies=self.include_dummies,
            checkpoint=cp,
        )
