# rcpsp-marketing

Экспериментальный стенд для **RCPSP** (Resource-Constrained Project Scheduling Problem — планирование проекта при ограниченных ресурсах) с **“маркетинговой” целевой функцией**: прибыль/выручка зависит от того, *когда* выполняются активности, и какие задачи выбираются/пропускаются (селективное расписание).

Проект содержит:

* парсер инстансов в формате PSPLIB (single-mode) и расширение с “маркетинговыми” блоками,
* работу с предшествованиями (DAG, топологические порядки),
* декодеры расписания (SGS): **serial** и **parallel** (несколько вариантов),
* вычисление **profit/revenue/cost по горизонту** (`T`),
* инкрементальные (частичные) пересчёты декодеров/цели,
* локальный поиск (HC/RLS/SA) по priority list,
* визуализацию (Gantt на Plotly, HTML).

---

## Содержание

* [Быстрый старт](#быстрый-старт)
* [Ключевые понятия](#ключевые-понятия)
* [Установка](#установка)
* [Типовые сценарии использования](#типовые-сценарии-использования)
* [API Reference (точные сигнатуры)](#api-reference-точные-сигнатуры)
* [Эксперименты](#эксперименты)
* [Воспроизводимость](#воспроизводимость)
* [Зависимости](#зависимости)
* [Roadmap](#roadmap)

---

## Быстрый старт

### 1) Установка (dev-режим)

Если у тебя настроен `pyproject.toml`, то:

```bash
pip install -e .
```

Иначе — добавь корень репо в `PYTHONPATH` или запускай из корня, где лежит `src/`.

---

### 2) Минимальный пайплайн: инстанс → порядок → SGS → profit

```python
from pathlib import Path

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.scheduling import parallel_sgs_selective_fast
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon

# 1) Парсим инстанс
proj = PSPLibExtendedParser().parse(Path("data/examples/your_instance.sm"))  # путь подставь свой

# 2) Получаем topological priority list
order = random_topo_sort_fixed_ends(proj, seed=42)

# 3) Декодируем в расписание (селективно)
T = 200  # горизонт (подставь свой)
res = parallel_sgs_selective_fast(proj, order, T=T, include_dummies=True, include_sink=False)

# 4) Считаем цель (profit = revenue - cost - penalty)
obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)
print(obj)
```

---

## Ключевые понятия

* **Project** — объект инстанса RCPSP + расширения: задачи, ресурсы, граф предшествований, начальные метрики сервиса, стоимости.
* **Priority list (order)** — топологически корректный порядок задач, который оптимизируется алгоритмами поиска.
* **SGS (Schedule Generation Scheme)** — “декодер” из priority list в расписание:

  * `serial_*` — последовательно ставит задачи,
  * `parallel_*` — событийно-ориентированная параллельная схема.
* **Selective scheduling** — декодер может **пропускать** задачи (не включать в расписание), если так выгоднее по цели.
* **Horizon T** — горизонт, на котором считается revenue/cost/profit.
* **ObjectiveValue** — структура с `value`, `revenue`, `cost`, `penalty`, `makespan`.

---

## Установка

Рекомендуется оформить проект как пакет (через `pyproject.toml`) и зафиксировать зависимости.

Минимально для текущего `src` используются:

* `pandas` (эксперименты/таблицы),
* `plotly` (Gantt/HTML),
* `networkx` (внутренние графовые утилиты, если используются в отдельных местах),
* `numpy` (точечно).

---

## Типовые сценарии использования

### Сравнить декодеры на одном priority list

В `rcpsp_marketing.experiments.decoder_compare` есть функция, которая прогоняет несколько SGS на одном порядке и возвращает `DataFrame`.

```python
from pathlib import Path
from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.experiments.decoder_compare import compare_decoders

proj = PSPLibExtendedParser().parse(Path("data/examples/your_instance.sm"))
df = compare_decoders(proj, T=200, order_mode="random_fixed_ends", seed=42)
print(df)
```

### Визуализировать “упаковки” (Gantt) для разных декодеров

```python
from pathlib import Path
from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.experiments.decoder_visualize import visualize_decoder_packings

proj = PSPLibExtendedParser().parse(Path("data/examples/your_instance.sm"))
order = random_topo_sort_fixed_ends(proj, seed=42)

out = visualize_decoder_packings(proj, order=order, T=200, out_dir="data/experiments/viz")
print(out)  # словарь {name -> Path}, где лежат HTML
```

### Быстро считать profit на одном расписании (fast objective)

Если ты много раз пересчитываешь цель на одном расписании (или небольших изменениях), используй fast-версию и/или кэш.

```python
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon_fast,
    build_profit_cache,
    evaluate_profit_with_cache,
)

obj_fast = evaluate_profit_over_horizon_fast(proj, res.schedule, selected_jobs=res.selected, T=200)

obj, cache = evaluate_profit_with_cache(proj, res.schedule, selected_jobs=res.selected, T=200)
```

---

## API Reference (точные сигнатуры)

Ниже перечислены публичные классы/функции **с реальными сигнатурами** из текущего `src`.

### `rcpsp_marketing.data.models`

* `MetricEffectsPct(AC: float = 0.0, LT: float = 0.0, CPC: float = 0.0, LCR: float = 0.0, PCR: float = 0.0) -> None`
* `ObjectiveValue(value: float, revenue: float = 0.0, cost: float = 0.0, penalty: float = 0.0, makespan: int = 0) -> None`
* `Project(tasks: Dict[int, Task], renewable_avail: Dict[int, int], successors: Dict[int, List[int]], predecessors: Dict[int, List[int]], metrics0: Optional[ServiceMetrics] = None, resource_costs: Dict[int, float] = <factory>, source_id: int = 1, sink_id: int = 0, name: str = '', meta: dict[str, object] = <factory>) -> None`
* `Schedule(start: Dict[int, int], finish: Dict[int, int], feasible: bool = True, notes: str = '') -> None`
* `ServiceMetrics(LT_0: float, AC_0: float, CPC_0: float, LCR_0: float, PCR_0: float, CAC_0: float, MARGIN0: float) -> None`
* `Task(id: int, duration: int, req: Dict[int, int] = <factory>, total_cost: float = 0.0, job_type: str = 'unknown', effects_pct: MetricEffectsPct = <factory) -> None`

### `rcpsp_marketing.io.psplib_base`

* `parse_jobs_count(content: str) -> int`
* `parse_num_renewable_resources(content: str) -> int`
* `parse_project_info(content: str) -> Dict[str, int]`
* `parse_precedence_arcs(content: str) -> List[Tuple[int, int]]`
* `parse_requests_durations(content: str, nR: int) -> Dict[int, Task]`
* `parse_resource_availabilities(content: str, nR: int) -> Dict[int, int]`
* `PSPLibBaseParser() -> None`

### `rcpsp_marketing.io.psplib_extended`

* `parse_service_metrics(content: str) -> Optional[ServiceMetrics]`
* `parse_resource_costs(content: str) -> Dict[int, float]`
* `parse_task_costs(content: str) -> Dict[int, float]`
* `parse_metric_changes(content: str) -> Dict[int, tuple[str, MetricEffectsPct]]`
* `PSPLibExtendedParser(base: PSPLibBaseParser = <factory>) -> None`

### `rcpsp_marketing.core.precedence`

* `topo_sort(project, *, prefer: str = 'smallest') -> List[int]`
* `is_topological_order(project, order: Iterable[int]) -> bool`
* `order_without_dummies(project, order: List[int]) -> List[int]`
* `random_topo_sort(project, *, seed: int | None = None) -> List[int]`
* `random_topo_sort_fixed_ends(project, *, seed: int | None = None) -> List[int]`

### `rcpsp_marketing.core.scheduling`

* `SGSResult(schedule: Schedule, selected: List[int], skipped: List[int]) -> None`
* `serial_sgs_selective(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True) -> SGSResult`
* `parallel_sgs_selective(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True, include_sink: bool = False) -> SGSResult`
* `parallel_sgs_selective_fast(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True, include_sink: bool = False) -> SGSResult`
* `parallel_sgs_selective_greedy(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True, include_sink: bool = False, min_score: float = -1e18, unlock_weight: float = 0.0) -> SGSResult`

### `rcpsp_marketing.core.scheduling_incremental`

* `SGSCheckpoint(idx: int, usage: Dict[int, List[int]], start: Dict[int, int], finish: Dict[int, int], done: Set[int], selected: List[int], skipped: List[int]) -> None`
* `serial_sgs_selective_with_checkpoints(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True, checkpoint_every: int = 10) -> Tuple[SGSResult, List[SGSCheckpoint]]`
* `serial_sgs_selective_resume(project: Project, priority_list: List[int], *, T: int, include_dummies: bool, checkpoint: SGSCheckpoint) -> SGSResult`
* `SerialSSGSIncremental(project: Project, T: int, include_dummies: bool = True, checkpoint_every: int = 10, _cur_order: Optional[List[int]] = None, _cur_res: Optional[SGSResult] = None, _cur_cps: Optional[List[SGSCheckpoint]] = None) -> None`

### `rcpsp_marketing.core.scheduling_parallel_incremental`

* `ParallelSnapshot(t0: int, usage: Dict[int, List[int]], start: Dict[int, int], finish: Dict[int, int], selected_prefix: List[int], done: Set[int], running: List[Tuple[int, int]], remaining: Set[int]) -> None`
* `snapshot_parallel_prefix_by_time(project: Project, priority_list: List[int], *, T: int, include_dummies: bool, include_sink: bool, base_res: SGSResult, t0: int) -> ParallelSnapshot`
* `parallel_sgs_selective_resume_from_snapshot(project: Project, priority_list: List[int], *, T: int, include_dummies: bool = True, include_sink: bool = False, snap: ParallelSnapshot) -> SGSResult`

### `rcpsp_marketing.core.objective`

* `evaluate_profit_over_horizon(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, include_dummy_costs: bool = False) -> ObjectiveValue`
* `evaluate_revenue_over_horizon(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None) -> float`
* `evaluate_cost_over_horizon(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, include_dummy_costs: bool = False) -> float`
* `evaluate_profit_over_horizon_fast(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, include_dummy_costs: bool = False) -> ObjectiveValue`
* `build_profit_cache(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, include_dummy_costs: bool = False) -> ProfitCache`
* `evaluate_profit_with_cache(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, include_dummy_costs: bool = False) -> Tuple[ObjectiveValue, ProfitCache]`
* `evaluate_makespan(schedule: Schedule) -> int`

### `rcpsp_marketing.core.improvement`

* `left_shift(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, hide_dummies: bool = True) -> ImproveResult`
* `right_shift(project: Project, schedule: Schedule, *, selected_jobs: Optional[Iterable[int]] = None, T: Optional[int] = None, hide_dummies: bool = True) -> ImproveResult`

### `rcpsp_marketing.viz.schedule`

* `plot_schedule_gantt(project: Project, schedule: Schedule, *, selected: Optional[Iterable[int]] = None, hide_dummies: bool = True, title: str = 'Schedule', T: Optional[int] = None)`
* `save_schedule_html(fig, out_path: str | Path) -> Path`

### `rcpsp_marketing.algorithms.local_search.*`

* `hill_climb(...) -> HCResult`
* `randomized_local_search(...) -> RLSResult`
* `simulated_annealing(...) -> SAResult`

---

## Эксперименты

Экспериментальные модули лежат в `rcpsp_marketing.experiments.*`.

* `decoder_compare.compare_decoders(...) -> pandas.DataFrame`
* `decoder_visualize.visualize_decoder_packings(...) -> dict[str, Path]`
* `compare_objective_ref_vs_fast` — сравнение `ref` vs `fast` objective
* `compare_end2end_ref_vs_fast` — end-to-end сравнение

---

## Воспроизводимость

* фиксируй `seed` при генерации topological order;
* фиксируй параметры декодеров (`T`, `include_dummies`, `include_sink`, `min_score`, `unlock_weight`);
* фиксируй параметры локального поиска (`seed`, `iters`, `max_profit_evals`);
* сохраняй результаты (CSV) и артефакты (HTML) в отдельную папку `data/experiments/...`.

---

## Зависимости

По текущему `src` используются:

* `pandas`
* `plotly`
* `networkx`
* `numpy`

---

## Roadmap

1. Нормальный CLI:

   * `rcpsp-marketing demo ...`
   * `rcpsp-marketing compare-decoders ...`
   * `rcpsp-marketing sa-batch ...`
2. Единый формат конфигов экспериментов (YAML/JSON).
3. Тесты:

   * корректность precedence,
   * валидность расписаний,
   * совпадение ref vs fast objective,
   * sanity-check на инстансах.
4. CI (ruff/pytest) + линтер/форматтер.
