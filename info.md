rcpsp-marketing/
├── pyproject.toml / pyproject.toml (или setup.cfg + setup.py)
├── README.md
├── LICENSE
├── .gitignore
├── data/
│   ├── raw/            # исходные PSPLib-инстансы (*.sm)
│   ├── extended/       # сгенерированные расширенные инстансы
│   └── experiments/    # CSV/JSON-выходы экспериментов (можно иначе)
├── notebooks/
│   └── ...             # Jupyter для анализа/графиков результата
├── scripts/
│   └── ...             # маленькие утилиты CLI (по желанию)
├── tests/
│   └── ...             # pytest-тесты
└── src/
    └── rcpsp_marketing/
        ├── __init__.py
        ├── config/
        │   ├── __init__.py
        │   └── defaults.py       # дефолтные параметры SA/RLS, T_max и т.п.
        ├── data/
        │   ├── __init__.py
        │   └── models.py         # Project, Task, ServiceMetrics, MetricEffectsPct
        ├── io/
        │   ├── __init__.py
        │   ├── psplib_base.py    # парсер базового PSPLib (без маркетинга)
        │   ├── psplib_extended.py# парсер твоего расширенного формата
        │   ├── generator.py      # генерация extended-инстансов из base
        │   └── schedule_export.py# CSV/JSON экспорт расписаний, лог стартов
        ├── core/
        │   ├── __init__.py
        │   ├── precedence.py     # граф предшествований, topo_sort, проверки
        │   ├── objective.py      # evaluate_chain_objective, evaluate_schedule_objective
        │   └── scheduling.py     # build_chain_schedule, serial_sgs_pack, утилиты ресурсов
        ├── algorithms/
        │   ├── __init__.py
        │   ├── local_search/   
        │   │   ├── __init__.py
        │   │   ├── hill_climb.py
        │   │   ├── simulated_annealing.py
        │   │   └── randomized_local_search.py
        │   ├── exact/
        │   │   ├── __init__.py
        │   │   └── exact_solver.py     # полный перебор топопорядков
        │   └── analytics/
        │       ├── __init__.py
        │       └── topo_stats.py       # count_topo_orders, MC-оценка
        ├── viz/
        │   ├── __init__.py
        │   ├── packing.py              # Plotly-графики по ресурсам
        │   └── search_history.py       # current vs best, SA + температура
        ├── experiments/
        │   ├── __init__.py
        │   ├── common.py               # load_project, make_initial_chain, утилиты CSV
        │   ├── sa_batch.py             # батч SA по инстансам
        │   └── compare_sa_rls.py       # сравнение SA vs RLS
        └── cli/
            ├── __init__.py
            ├── main_demo.py            # пример: один инстанс → SA → SGS → визуализации
            ├── generate_instances.py   # генерация extended-инстансов
            ├── run_sa_batch.py         # обёртки над experiments.sa_batch
            └── run_compare.py          # обёртка над experiments.compare_sa_rls
