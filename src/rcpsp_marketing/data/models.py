# src/rcpsp_marketing/data/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set


# -----------------------------
# Метрики и эффекты
# -----------------------------

@dataclass(frozen=True, slots=True)
class MetricEffectsPct:
    """Мультипликативные эффекты задачи в процентах.
    Пример: +5.0 означает * (1 + 0.05); -3.0 означает * (1 - 0.03)
    """
    AC: float = 0.0
    LT: float = 0.0
    CPC: float = 0.0
    LCR: float = 0.0
    PCR: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {"AC": self.AC, "LT": self.LT, "CPC": self.CPC, "LCR": self.LCR, "PCR": self.PCR}

    def __str__(self) -> str:
        return f"Δ% AC={self.AC:+.2f} LT={self.LT:+.2f} CPC={self.CPC:+.2f} LCR={self.LCR:+.2f} PCR={self.PCR:+.2f}"


@dataclass(frozen=True, slots=True)
class ServiceMetrics:
    """Начальные значения метрик (как в блоке SERVICE METRICS)."""
    LT_0: float
    AC_0: float
    CPC_0: float
    LCR_0: float
    PCR_0: float
    CAC_0: float
    MARGIN0: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "LT_0": self.LT_0,
            "AC_0": self.AC_0,
            "CPC_0": self.CPC_0,
            "LCR_0": self.LCR_0,
            "PCR_0": self.PCR_0,
            "CAC_0": self.CAC_0,
            "MARGIN0": self.MARGIN0,
        }

    def __str__(self) -> str:
        return (
            "ServiceMetrics("
            f"LT_0={self.LT_0}, AC_0={self.AC_0}, CPC_0={self.CPC_0}, "
            f"LCR_0={self.LCR_0}, PCR_0={self.PCR_0}, CAC_0={self.CAC_0}, MARGIN0={self.MARGIN0})"
        )


# -----------------------------
# RCPSP: задачи, проект
# -----------------------------

@dataclass(slots=True)
class Task:    
    id: int
    duration: int
    req: Dict[int, int] = field(default_factory=dict)  # renewable resource usage: r -> units
    total_cost: float = 0.0                            # из TASK COSTS (total)
    job_type: str = "unknown"                          # из METRIC CHANGES (type)
    effects_pct: MetricEffectsPct = field(default_factory=MetricEffectsPct)

    def validate(self, *, num_resources: Optional[int] = None) -> None:
        if self.id <= 0:
            raise ValueError(f"Task.id must be positive, got {self.id}")
        if self.duration < 0:
            raise ValueError(f"Task#{self.id}: duration must be >= 0, got {self.duration}")
        for r, u in self.req.items():
            if r <= 0:
                raise ValueError(f"Task#{self.id}: resource index must be >= 1, got R{r}")
            if u < 0:
                raise ValueError(f"Task#{self.id}: resource usage must be >= 0, got R{r}={u}")
            if num_resources is not None and r > num_resources:
                raise ValueError(f"Task#{self.id}: uses R{r} but project has only {num_resources} renewable resources")

        if self.total_cost < 0:
            raise ValueError(f"Task#{self.id}: total_cost must be >= 0, got {self.total_cost}")

    def summary(self) -> str:
        req_str = ", ".join(f"R{r}={u}" for r, u in sorted(self.req.items())) or "none"
        return (
            f"Task#{self.id} dur={self.duration} cost={self.total_cost:.2f} "
            f"type={self.job_type} req=[{req_str}] effects=({self.effects_pct})"
        )

    def __str__(self) -> str:
        return self.summary()


@dataclass(slots=True)
class Project:
    """Полное описание проекта RCPSP + маркетинговые расширения."""
    tasks: Dict[int, Task]
    renewable_avail: Dict[int, int]                 # r -> capacity per time
    successors: Dict[int, List[int]]                # i -> [j...]
    predecessors: Dict[int, List[int]]              # j -> [i...]
    metrics0: Optional[ServiceMetrics] = None
    resource_costs: Dict[int, float] = field(default_factory=dict)  # r -> cost per unit per time

    source_id: int = 1
    sink_id: int = 0

    name: str = ""

    meta: dict[str, object] = field(default_factory=dict)

    @property
    def n_jobs(self) -> int:
        return len(self.tasks)

    @property
    def n_resources(self) -> int:
        return len(self.renewable_avail)

    def task_ids(self) -> List[int]:
        return sorted(self.tasks.keys())

    def validate(self, *, check_dag: bool = True) -> None:
        if not self.tasks:
            raise ValueError("Project.tasks is empty")

        # capacities
        for r, cap in self.renewable_avail.items():
            if r <= 0:
                raise ValueError(f"Resource index must be >= 1, got R{r}")
            if cap < 0:
                raise ValueError(f"Resource capacity must be >= 0, got R{r}={cap}")

        # tasks
        for t in self.tasks.values():
            t.validate(num_resources=self.n_resources)

        # graph consistency
        all_ids = set(self.tasks.keys())
        for i, succs in self.successors.items():
            if i not in all_ids:
                raise ValueError(f"successors has unknown task id {i}")
            for j in succs:
                if j not in all_ids:
                    raise ValueError(f"successors[{i}] contains unknown task id {j}")
        for j, preds in self.predecessors.items():
            if j not in all_ids:
                raise ValueError(f"predecessors has unknown task id {j}")
            for i in preds:
                if i not in all_ids:
                    raise ValueError(f"predecessors[{j}] contains unknown task id {i}")

        # inverse check (weak but полезно)
        for i, succs in self.successors.items():
            for j in succs:
                if i not in self.predecessors.get(j, []):
                    raise ValueError(f"Graph mismatch: {i}->{j} in successors but not in predecessors[{j}]")
        for j, preds in self.predecessors.items():
            for i in preds:
                if j not in self.successors.get(i, []):
                    raise ValueError(f"Graph mismatch: {i}->{j} in predecessors but not in successors[{i}]")

        if check_dag:
            self._validate_dag()

    def _validate_dag(self) -> None:
        """Проверка отсутствия циклов (Kahn)."""
        indeg = {i: len(self.predecessors.get(i, [])) for i in self.tasks}
        q = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while q:
            v = q.pop()
            seen += 1
            for u in self.successors.get(v, []):
                indeg[u] -= 1
                if indeg[u] == 0:
                    q.append(u)
        if seen != len(self.tasks):
            raise ValueError("Precedence graph has a cycle (not a DAG)")

    def check_topological_order(self, order: Iterable[int]) -> None:
        """Проверка, что порядок не нарушает precedence."""
        pos: Dict[int, int] = {}
        for k, job in enumerate(order):
            pos[job] = k

        missing = set(self.tasks.keys()) - set(pos.keys())
        extra = set(pos.keys()) - set(self.tasks.keys())
        if missing:
            raise ValueError(f"Topological order is missing tasks: {sorted(missing)[:10]}")
        if extra:
            raise ValueError(f"Topological order has unknown tasks: {sorted(extra)[:10]}")

        for i, succs in self.successors.items():
            pi = pos[i]
            for j in succs:
                if pi > pos[j]:
                    raise ValueError(f"Order violates precedence: {i} must be before {j}")

    def summary(self, *, max_tasks: int = 5) -> str:
        ids = self.task_ids()
        head = ids[:max_tasks]
        tail = ids[-max_tasks:] if len(ids) > max_tasks else []
        tasks_preview = ", ".join(str(i) for i in head) + (" ... " + ", ".join(str(i) for i in tail) if tail else "")
        return (
            f"Project(name='{self.name}', jobs={self.n_jobs}, R={self.n_resources})\n"
            f"  tasks: {tasks_preview}\n"
            f"  capacities: " + ", ".join(f"R{r}={c}" for r, c in sorted(self.renewable_avail.items())) + "\n"
            f"  metrics0: {self.metrics0}\n"
        )

    def __str__(self) -> str:
        return self.summary()


# -----------------------------
# Расписание
# -----------------------------

@dataclass(slots=True)
class Schedule:
    """Расписание: start/finish по job id."""
    start: Dict[int, int]
    finish: Dict[int, int]
    feasible: bool = True
    notes: str = ""

    @property
    def makespan(self) -> int:
        return max(self.finish.values()) if self.finish else 0

    def validate_precedence(self, project: Project) -> None:
        for i, succs in project.successors.items():
            fi = self.finish.get(i)
            if fi is None:
                raise ValueError(f"Schedule missing finish time for task {i}")
            for j in succs:
                sj = self.start.get(j)
                if sj is None:
                    raise ValueError(f"Schedule missing start time for task {j}")
                if sj < fi:
                    raise ValueError(f"Precedence violated in schedule: {i}->{j}, start[{j}]={sj} < finish[{i}]={fi}")

    def validate_resources(self, project: Project) -> None:
        """Проверка renewable ресурсов дискретно по t=0..makespan-1 (для тестов/отладки)."""
        T = self.makespan
        if T == 0:
            return

        # precompute per time usage
        for t in range(T):
            usage: Dict[int, int] = {r: 0 for r in project.renewable_avail}
            for j, sj in self.start.items():
                fj = self.finish[j]
                if sj <= t < fj:
                    task = project.tasks[j]
                    for r, u in task.req.items():
                        usage[r] += u
            for r, cap in project.renewable_avail.items():
                if usage[r] > cap:
                    raise ValueError(f"Resource violated at t={t}: R{r} usage={usage[r]} > cap={cap}")

    def summary(self, project: Optional[Project] = None, *, max_jobs: int = 8) -> str:
        ids = sorted(self.start.keys())
        show = ids[:max_jobs]
        items = []
        for j in show:
            items.append(f"{j}: {self.start[j]}→{self.finish[j]}")
        if len(ids) > max_jobs:
            items.append("...")
        base = f"Schedule(makespan={self.makespan}, feasible={self.feasible}) " + " ".join(items)
        if project is not None:
            base += f"\n  project: jobs={project.n_jobs}, R={project.n_resources}"
        return base

    def __str__(self) -> str:
        return self.summary()


# -----------------------------
# Значение целевой функции
# -----------------------------

@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    value: float
    revenue: float = 0.0
    cost: float = 0.0
    penalty: float = 0.0
    makespan: int = 0

    def __str__(self) -> str:
        return (
            f"Objective(value={self.value:.4f}, revenue={self.revenue:.2f}, "
            f"cost={self.cost:.2f}, penalty={self.penalty:.2f}, makespan={self.makespan})"
        )
