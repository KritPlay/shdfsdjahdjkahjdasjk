# rcpsp_marketing/core/objective.py
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from operator import itemgetter
from typing import Iterable, Optional, List, Tuple, Dict

from rcpsp_marketing.data.models import (
    Project,
    Schedule,
    ObjectiveValue,
    MetricEffectsPct,
    ServiceMetrics,
)


# =============================================================================
# Metric dynamics
# =============================================================================

@dataclass(slots=True)
class MetricState:
    """
    Состояние сервисных метрик, определяющее текущую прибыльность (profit_rate).

    Модель:
      profit_rate(t) = AC(t) * (LT(t) - CAC(t))
      CAC(t) = CPC(t) / (LCR(t) * PCR(t))

    Эффекты задач применяются *в момент завершения* (finish[j]).
    """
    LT: float
    AC: float
    CPC: float
    LCR: float
    PCR: float

    CAC: float = 0.0
    margin: float = 0.0

    @classmethod
    def from_metrics0(cls, m: ServiceMetrics) -> "MetricState":
        st = cls(
            LT=float(m.LT_0),
            AC=float(m.AC_0),
            CPC=float(m.CPC_0),
            LCR=float(m.LCR_0),
            PCR=float(m.PCR_0),
        )
        st.recompute()
        return st

    def recompute(self) -> None:
        denom = max(1e-12, self.LCR * self.PCR)
        self.CAC = self.CPC / denom
        self.margin = self.LT - self.CAC

    def apply_effects_pct(self, eff: MetricEffectsPct) -> None:
        # мультипликативные эффекты: x <- x * (1 + pct/100)
        self.AC *= (1.0 + float(eff.AC) / 100.0)
        self.LT *= (1.0 + float(eff.LT) / 100.0)
        self.CPC *= (1.0 + float(eff.CPC) / 100.0)
        self.LCR *= (1.0 + float(eff.LCR) / 100.0)
        self.PCR *= (1.0 + float(eff.PCR) / 100.0)

        # защиты (чтобы не взорваться)
        self.AC = max(0.0, self.AC)
        self.LT = max(0.0, self.LT)
        self.CPC = max(1e-6, self.CPC)
        self.LCR = min(max(1e-6, self.LCR), 1.0)
        self.PCR = min(max(1e-6, self.PCR), 1.0)

        self.recompute()

    def profit_rate(self) -> float:
        return self.AC * self.margin


# =============================================================================
# Helpers: events + costs
# =============================================================================

def _as_int(x) -> int:
    return int(x)  # centralize casts


def _collect_events(
    schedule: Schedule,
    *,
    selected_jobs: Iterable[int],
    T: int,
) -> List[Tuple[int, int]]:
    """
    События = (finish_time, job), только если finish_time <= T.
    Стабильная сортировка по (t, job) для детерминизма.
    """
    finish = schedule.finish
    events: List[Tuple[int, int]] = []
    ev_append = events.append

    for j0 in selected_jobs:
        j = _as_int(j0)
        fj = finish.get(j)
        if fj is None:
            continue
        t = _as_int(fj)
        if t <= T:
            ev_append((t, j))

    events.sort(key=itemgetter(0, 1))
    return events


def _cost_finished_by_T(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Iterable[int],
    T: int,
    include_dummy_costs: bool,
) -> float:
    """
    COST-правило (как ты описал):
      расходы включаются только для задач, которые УСПЕЛИ завершиться к T (finish[j] <= T),
      иначе расходы по задаче = 0.

    Dummy-работы (source/sink) можно исключать.
    """
    tasks = project.tasks
    finish = schedule.finish

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    cost = 0.0
    for j0 in selected_jobs:
        j = _as_int(j0)

        if not include_dummy_costs and (j == src or j == snk):
            continue

        fj = finish.get(j)
        if fj is None or _as_int(fj) > T:
            continue  # не завершилась к T => расход 0

        task = tasks.get(j)
        if task is not None:
            cost += float(task.total_cost)

    return cost


# =============================================================================
# Objective functions (main / reference)
# =============================================================================

def evaluate_profit_over_horizon(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    include_dummy_costs: bool = False,
) -> ObjectiveValue:
    """
    Reference (понятная) версия.

    Прибыль на [0, T]:
      revenue = ∫ profit_rate(t) dt
      cost    = сумма total_cost только тех задач, которые завершились к T (finish[j] <= T)
      value   = revenue - cost

    Эффекты задач применяются в момент завершения (finish[j]).
    """

    if project.metrics0 is None:
        raise ValueError("project.metrics0 is None: нужен extended-инстанс с SERVICE METRICS (INITIAL).")

    if T is None:
        T = schedule.makespan
    T = _as_int(T)

    if selected_jobs is None:
        selected_jobs = schedule.finish.keys()

    events = _collect_events(schedule, selected_jobs=selected_jobs, T=T)

    st = MetricState.from_metrics0(project.metrics0)

    revenue = 0.0
    prev_t = 0
    idx = 0

    while idx < len(events):
        t = events[idx][0]
        if t > T:
            break

        # прибыль на [prev_t, t)
        if t > prev_t:
            revenue += st.profit_rate() * float(t - prev_t)
            prev_t = t

        # применяем эффекты всех задач, завершившихся в t
        while idx < len(events) and events[idx][0] == t:
            j = events[idx][1]
            task = project.tasks.get(j)
            if task is not None:
                st.apply_effects_pct(task.effects_pct)
            idx += 1

    # хвост до T
    if prev_t < T:
        revenue += st.profit_rate() * float(T - prev_t)

    cost = _cost_finished_by_T(
        project,
        schedule,
        selected_jobs=selected_jobs,
        T=T,
        include_dummy_costs=include_dummy_costs,
    )

    value = revenue - cost
    return ObjectiveValue(value=value, revenue=revenue, cost=cost, penalty=0.0, makespan=schedule.makespan)


def evaluate_revenue_over_horizon(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
) -> float:
    """
    Только доход (без cost). Удобно если cost считаешь отдельно/кешируешь.
    """
    obj = evaluate_profit_over_horizon(project, schedule, selected_jobs=selected_jobs, T=T, include_dummy_costs=True)
    return float(obj.revenue)


def evaluate_cost_over_horizon(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    include_dummy_costs: bool = False,
) -> float:
    """
    Только cost по правилу: считаем только задачи, завершившиеся к T.
    """
    if T is None:
        T = schedule.makespan
    T = _as_int(T)

    if selected_jobs is None:
        selected_jobs = schedule.finish.keys()

    return float(
        _cost_finished_by_T(
            project,
            schedule,
            selected_jobs=selected_jobs,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
    )


# =============================================================================
# Objective functions (fast drop-in)
# =============================================================================

def evaluate_profit_over_horizon_fast(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    include_dummy_costs: bool = False,
) -> ObjectiveValue:
    """
    Быстрая drop-in версия:
    - меньше аллокаций
    - меньше вызовов методов (работаем на float)
    - cost считается по правилу: только завершившиеся к T
    """

    m0 = project.metrics0
    if m0 is None:
        raise ValueError("project.metrics0 is None: нужен extended-инстанс с SERVICE METRICS (INITIAL).")

    finish = schedule.finish
    tasks = project.tasks

    if T is None:
        T = schedule.makespan
    T = _as_int(T)

    if selected_jobs is None:
        selected_jobs = finish.keys()

    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    # events + cost in one pass
    events: List[Tuple[int, int]] = []
    ev_append = events.append
    cost = 0.0

    for j0 in selected_jobs:
        j = _as_int(j0)
        fj = finish.get(j)
        fj_int = None
        if fj is not None:
            fj_int = _as_int(fj)
            if fj_int <= T:
                ev_append((fj_int, j))

        # cost only if finished by T
        if not include_dummy_costs and (j == src or j == snk):
            continue
        if fj_int is None or fj_int > T:
            continue

        task = tasks.get(j)
        if task is not None:
            cost += float(task.total_cost)

    events.sort(key=itemgetter(0, 1))

    # state floats
    LT = float(m0.LT_0)
    AC = float(m0.AC_0)
    CPC = float(m0.CPC_0)
    LCR = float(m0.LCR_0)
    PCR = float(m0.PCR_0)

    def recompute_rate(LT: float, AC: float, CPC: float, LCR: float, PCR: float) -> float:
        denom = LCR * PCR
        if denom < 1e-12:
            denom = 1e-12
        CAC = CPC / denom
        margin = LT - CAC
        return AC * margin

    rate = recompute_rate(LT, AC, CPC, LCR, PCR)

    revenue = 0.0
    prev_t = 0
    idx = 0
    n = len(events)

    while idx < n:
        t = events[idx][0]
        if t > T:
            break

        dt = t - prev_t
        if dt:
            revenue += rate * float(dt)
            prev_t = t

        while idx < n and events[idx][0] == t:
            j = events[idx][1]
            task = tasks.get(j)
            if task is not None:
                eff = task.effects_pct

                # multiplicative effects
                AC *= (1.0 + float(eff.AC) / 100.0)
                LT *= (1.0 + float(eff.LT) / 100.0)
                CPC *= (1.0 + float(eff.CPC) / 100.0)
                LCR *= (1.0 + float(eff.LCR) / 100.0)
                PCR *= (1.0 + float(eff.PCR) / 100.0)

                # clamps
                if AC < 0.0: AC = 0.0
                if LT < 0.0: LT = 0.0
                if CPC < 1e-6: CPC = 1e-6
                if LCR < 1e-6: LCR = 1e-6
                elif LCR > 1.0: LCR = 1.0
                if PCR < 1e-6: PCR = 1e-6
                elif PCR > 1.0: PCR = 1.0

                rate = recompute_rate(LT, AC, CPC, LCR, PCR)

            idx += 1

    if prev_t < T:
        revenue += rate * float(T - prev_t)

    value = revenue - cost
    return ObjectiveValue(value=value, revenue=revenue, cost=cost, penalty=0.0, makespan=schedule.makespan)


# =============================================================================
# Incremental / cached evaluation for SA
# =============================================================================

@dataclass(slots=True)
class ProfitCache:
    """
    Кэш профита по breakpoints.

    times:   [t0=0, t1, ..., tk=T]
    prefix:  prefix[i] = revenue accumulated up to times[i]
    states:  states[i] = (LT, AC, CPC, LCR, PCR) at the beginning of interval [times[i], times[i+1])
    rates:   profit_rate on interval [times[i], times[i+1])
    cost:    cost computed with rule "only finished by T"
    T:       horizon
    """
    times: List[int]
    prefix: List[float]
    states: List[Tuple[float, float, float, float, float]]
    rates: List[float]
    cost: float
    T: int


def _rate_from_state(st: Tuple[float, float, float, float, float]) -> float:
    LT, AC, CPC, LCR, PCR = st
    denom = LCR * PCR
    if denom < 1e-12:
        denom = 1e-12
    CAC = CPC / denom
    margin = LT - CAC
    return AC * margin


def _apply_effect_to_state(
    st: Tuple[float, float, float, float, float],
    eff: MetricEffectsPct,
) -> Tuple[float, float, float, float, float]:
    LT, AC, CPC, LCR, PCR = st

    AC *= (1.0 + float(eff.AC) / 100.0)
    LT *= (1.0 + float(eff.LT) / 100.0)
    CPC *= (1.0 + float(eff.CPC) / 100.0)
    LCR *= (1.0 + float(eff.LCR) / 100.0)
    PCR *= (1.0 + float(eff.PCR) / 100.0)

    # clamps
    if AC < 0.0: AC = 0.0
    if LT < 0.0: LT = 0.0
    if CPC < 1e-6: CPC = 1e-6
    if LCR < 1e-6: LCR = 1e-6
    elif LCR > 1.0: LCR = 1.0
    if PCR < 1e-6: PCR = 1e-6
    elif PCR > 1.0: PCR = 1.0

    return (LT, AC, CPC, LCR, PCR)


def build_profit_cache(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    include_dummy_costs: bool = False,
) -> ProfitCache:
    """
    Построить полный кэш для данного (schedule, selected_jobs).
    """
    m0 = project.metrics0
    if m0 is None:
        raise ValueError("project.metrics0 is None: нужен extended-инстанс с SERVICE METRICS (INITIAL).")

    if T is None:
        T = schedule.makespan
    T = _as_int(T)

    if selected_jobs is None:
        selected_jobs = schedule.finish.keys()

    events = _collect_events(schedule, selected_jobs=selected_jobs, T=T)

    # cost by rule "only finished by T"
    cost = _cost_finished_by_T(
        project,
        schedule,
        selected_jobs=selected_jobs,
        T=T,
        include_dummy_costs=include_dummy_costs,
    )

    # initial state at t=0
    st = (float(m0.LT_0), float(m0.AC_0), float(m0.CPC_0), float(m0.LCR_0), float(m0.PCR_0))
    rate = _rate_from_state(st)

    times: List[int] = [0]
    prefix: List[float] = [0.0]
    states: List[Tuple[float, float, float, float, float]] = [st]
    rates: List[float] = [rate]

    revenue = 0.0
    prev_t = 0
    idx = 0
    n = len(events)

    while idx < n:
        t = events[idx][0]
        if t > T:
            break

        dt = t - prev_t
        if dt:
            revenue += rate * float(dt)
            prev_t = t

            # breakpoint BEFORE applying effects at time t
            times.append(t)
            prefix.append(revenue)
            states.append(st)
            rates.append(rate)

        # apply all effects at time t
        while idx < n and events[idx][0] == t:
            j = events[idx][1]
            task = project.tasks.get(j)
            if task is not None:
                st = _apply_effect_to_state(st, task.effects_pct)
                rate = _rate_from_state(st)
            idx += 1

    # tail to T
    if prev_t < T:
        revenue += rate * float(T - prev_t)

    # ensure final point at T
    if times[-1] != T:
        times.append(T)
        prefix.append(revenue)
        states.append(st)
        rates.append(rate)

    return ProfitCache(times=times, prefix=prefix, states=states, rates=rates, cost=cost, T=T)


def cache_prefix_state_at(
    cache: ProfitCache,
    t: int,
) -> Tuple[float, Tuple[float, float, float, float, float], float]:
    """
    Возвращает:
      revenue_up_to_t,
      state_at_interval_start (для интервала содержащего t),
      rate_on_that_interval
    """
    t = _as_int(t)

    if t <= cache.times[0]:
        return 0.0, cache.states[0], cache.rates[0]
    if t >= cache.T:
        return cache.prefix[-1], cache.states[-1], cache.rates[-1]

    i = bisect_right(cache.times, t) - 1  # times[i] <= t < times[i+1]
    base_rev = cache.prefix[i]
    base_t = cache.times[i]
    st = cache.states[i]
    rate = cache.rates[i]
    return base_rev + rate * float(t - base_t), st, rate


def update_cost_delta_finished_by_T(
    project: Project,
    *,
    changed_jobs: Iterable[int],
    old_finish: Dict[int, int],
    new_finish: Dict[int, int],
    T: int,
    include_dummy_costs: bool = False,
) -> float:
    """
    Быстрое инкрементальное обновление cost, если известны changed_jobs:
    cost учитывается только если finish[j] <= T.

    Возвращает delta_cost = new_cost - old_cost (только вклад changed_jobs).
    """
    src = getattr(project, "source_id", None)
    snk = getattr(project, "sink_id", None)

    delta = 0.0
    for j0 in changed_jobs:
        j = _as_int(j0)

        if not include_dummy_costs and (j == src or j == snk):
            continue

        task = project.tasks.get(j)
        if task is None:
            continue

        fo = old_finish.get(j)
        fn = new_finish.get(j)

        old_ok = (fo is not None and _as_int(fo) <= T)
        new_ok = (fn is not None and _as_int(fn) <= T)

        if old_ok != new_ok:
            c = float(task.total_cost)
            delta += c if new_ok else -c

    return delta


def update_revenue_from_t0(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Iterable[int],
    old_cache: ProfitCache,
    t0: int,
) -> float:
    """
    Инкрементально пересчитать revenue на [t0, T], используя old_cache для [0, t0).
    COST здесь не трогаем (отдельно).
    """
    T = old_cache.T
    t0 = max(0, min(_as_int(t0), T))

    revenue, st, rate = cache_prefix_state_at(old_cache, t0)
    prev_t = t0

    # events начиная с t0
    events: List[Tuple[int, int]] = []
    ev_append = events.append
    finish = schedule.finish

    for j0 in selected_jobs:
        j = _as_int(j0)
        fj = finish.get(j)
        if fj is None:
            continue
        t = _as_int(fj)
        if t0 <= t <= T:
            ev_append((t, j))

    events.sort(key=itemgetter(0, 1))

    idx = 0
    n = len(events)
    while idx < n:
        t = events[idx][0]
        if t > T:
            break

        dt = t - prev_t
        if dt:
            revenue += rate * float(dt)
            prev_t = t

        while idx < n and events[idx][0] == t:
            j = events[idx][1]
            task = project.tasks.get(j)
            if task is not None:
                st = _apply_effect_to_state(st, task.effects_pct)
                rate = _rate_from_state(st)
            idx += 1

    if prev_t < T:
        revenue += rate * float(T - prev_t)

    return revenue


def evaluate_profit_with_cache(
    project: Project,
    schedule: Schedule,
    *,
    selected_jobs: Optional[Iterable[int]] = None,
    T: Optional[int] = None,
    include_dummy_costs: bool = False,
) -> Tuple[ObjectiveValue, ProfitCache]:
    """
    Удобный враппер: вернуть ObjectiveValue + cache.
    """
    cache = build_profit_cache(
        project,
        schedule,
        selected_jobs=selected_jobs,
        T=T,
        include_dummy_costs=include_dummy_costs,
    )
    revenue = float(cache.prefix[-1])
    cost = float(cache.cost)
    value = revenue - cost
    obj = ObjectiveValue(value=value, revenue=revenue, cost=cost, penalty=0.0, makespan=schedule.makespan)
    return obj, cache


# =============================================================================
# (Optional) trivial auxiliary objectives
# =============================================================================

def evaluate_makespan(schedule: Schedule) -> int:
    """Если где-то нужно в качестве метрики/цели."""
    return int(schedule.makespan)
