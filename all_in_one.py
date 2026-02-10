# === 2app.py ===
# app.py
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Make imports work when running from repo root (Streamlit)
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Helpers
# ============================================================
def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def list_dirs(root: str) -> List[str]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


@st.cache_data(show_spinner=False)
def list_instances(out_root: str, cls: str) -> List[str]:
    p = Path(out_root) / cls
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_schedule_html_for_row(inst_dir: Path, decoder: str, searcher: str, seed: int) -> Optional[Path]:
    """
    Try in priority:
      1) decoder=.../search=.../seedX/schedule.html
      2) decoder=.../search=.../BEST_seedX/schedule.html
      3) any BEST_seed*/schedule.html for that combo
      4) any schedule.html inside combo
    """
    combo_dir = inst_dir / f"decoder={decoder}" / f"search={searcher}"
    if not combo_dir.exists():
        return None

    p1 = combo_dir / f"seed{seed}" / "schedule.html"
    if p1.exists():
        return p1

    p2 = combo_dir / f"BEST_seed{seed}" / "schedule.html"
    if p2.exists():
        return p2

    best_any = sorted(combo_dir.glob("BEST_seed*/schedule.html"))
    if best_any:
        return best_any[0]

    any_sched = sorted(combo_dir.glob("**/schedule.html"))
    if any_sched:
        return any_sched[0]

    return None


def find_sa_diagnostics_html(inst_dir: Path, decoder: str) -> Optional[Path]:
    """
    Expected:
      decoder=DEC/search=SA/sa_diagnostics_first_run.html
    """
    p = inst_dir / f"decoder={decoder}" / "search=SA" / "sa_diagnostics_first_run.html"
    return p if p.exists() else None


def boxplot_values(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=title)
        return fig

    for (dec, sea), g in df.groupby(["decoder", "searcher"]):
        fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))

    fig.update_layout(
        title=title,
        xaxis=dict(title="decoder+searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    return fig


def bar_mean_values(df: pd.DataFrame, title: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    g = (
        df.groupby(["decoder", "searcher"], as_index=False)
        .agg(mean_value=("value", "mean"), best_value=("value", "max"), runs=("value", "count"))
        .sort_values("mean_value", ascending=False)
    )
    x = [f"{d}+{s}" for d, s in zip(g["decoder"], g["searcher"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=g["mean_value"], name="mean_value"))
    fig.update_layout(title=title, xaxis=dict(title="combo"), yaxis=dict(title="mean value"))
    return fig


def agg_combo_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    g = df.groupby(["decoder", "searcher"], as_index=False).agg(
        runs=("value", "count"),
        mean_value=("value", "mean"),
        std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        best_value=("value", "max"),
        mean_time=("time_sec", "mean") if "time_sec" in df.columns else ("value", "count"),
        mean_selected=("selected", "mean") if "selected" in df.columns else ("value", "count"),
        mean_skipped=("skipped", "mean") if "skipped" in df.columns else ("value", "count"),
        mean_makespan=("makespan", "mean") if "makespan" in df.columns else ("value", "count"),
    )
    return g.sort_values(["mean_value"], ascending=False)


def agg_class_combo_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    class x combo aggregates (when scope=ALL)
    """
    if df.empty or "class" not in df.columns:
        return pd.DataFrame()

    g = df.groupby(["class", "decoder", "searcher"], as_index=False).agg(
        runs=("value", "count"),
        mean_value=("value", "mean"),
        best_value=("value", "max"),
        std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        mean_time=("time_sec", "mean") if "time_sec" in df.columns else ("value", "count"),
    )
    return g.sort_values(["class", "mean_value"], ascending=[True, False])


# ============================================================
# Streamlit UI (viewer-only)
# ============================================================
st.set_page_config(page_title="RCPSP-Marketing | Results Viewer", layout="wide")
st.title("RCPSP-Marketing — Results Viewer")

with st.sidebar:
    st.header("Путь к результатам")
    OUT_ROOT = Path(st.text_input("OUT_ROOT", r"data/experiments/algorithm_comparison"))

    st.divider()
    st.header("Фильтры таблицы")
    ONLY_ONE_SEED = st.checkbox("Показывать только один seed", value=False)
    ONE_SEED_VALUE = st.number_input("seed", min_value=1, value=1, step=1, disabled=not ONLY_ONE_SEED)

    st.divider()
    st.header("Сортировка")
    SORT_COL = st.text_input("sort column", "value")
    SORT_ASC = st.checkbox("ascending", value=False)

tab_overview, tab_instance = st.tabs(["Глобальный обзор", "Инстанс / просмотр упаковки"])


# ============================================================
# Global overview
# ============================================================
with tab_overview:
    st.subheader("Глобальный обзор")

    if not OUT_ROOT.exists():
        st.warning(f"OUT_ROOT не найден: {OUT_ROOT}")
        st.stop()

    # available classes from folder names (excluding _global etc.)
    classes_available = [c for c in list_dirs(str(OUT_ROOT)) if re.match(r"^j\d+$", c.lower())]
    scope_options = ["ALL"] + classes_available if classes_available else ["ALL"]

    scope_cls = st.selectbox("Scope (какие классы включать в глобальный отчёт)", scope_options, index=0)

    global_dir = OUT_ROOT / "_global"
    all_results_path = global_dir / "all_results.csv"

    # Load global or scan results.csv
    if all_results_path.exists():
        all_df = read_csv(str(all_results_path))
        st.caption("Источник: _global/all_results.csv")
    else:
        result_files = sorted(OUT_ROOT.rglob("results.csv"))
        result_files = [p for p in result_files if "_global" not in p.parts]
        if not result_files:
            st.info("Не найдено ни одного results.csv в OUT_ROOT.")
            st.stop()

        frames = []
        for rf in result_files:
            try:
                df = read_csv(str(rf))
                df["results_csv_path"] = str(rf)
                frames.append(df)
            except Exception:
                pass

        if not frames:
            st.info("results.csv есть, но прочитать не удалось.")
            st.stop()

        all_df = pd.concat(frames, ignore_index=True)
        st.caption("Источник: сканирование results.csv ( _global/all_results.csv не найден )")

    # sanity
    required = {"class", "instance", "decoder", "searcher", "seed", "value"}
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        st.error(f"В глобальных данных не хватает колонок: {missing}")
        st.stop()

    # Seed filter
    if ONLY_ONE_SEED:
        all_df = all_df[all_df["seed"] == int(ONE_SEED_VALUE)]

    # Class scope filter
    if scope_cls != "ALL":
        all_df = all_df[all_df["class"] == scope_cls]

    # Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Агрегаты по combo")
        agg = agg_combo_table(all_df)
        st.dataframe(agg, use_container_width=True, height=340)

        if scope_cls == "ALL":
            st.markdown("### Агрегаты по классам отдельно (class × combo)")
            agg_cc = agg_class_combo_table(all_df)
            st.dataframe(agg_cc, use_container_width=True, height=340)

    with col2:
        st.markdown("### Boxplot по combo (value)")
        fig = boxplot_values(all_df, title=f"All results ({scope_cls}): value distribution by decoder+searcher")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### Топ-строки (по value)")
    show_cols = [
        "class", "instance", "decoder", "searcher", "seed",
        "value", "revenue", "cost", "makespan", "selected", "skipped", "time_sec",
    ]
    show_cols = [c for c in show_cols if c in all_df.columns]

    topn = st.number_input("Top N", min_value=5, value=30, step=5)
    st.dataframe(
        all_df.sort_values("value", ascending=False)[show_cols].head(int(topn)),
        use_container_width=True,
        height=360,
    )


# ============================================================
# Instance viewer
# ============================================================
with tab_instance:
    st.subheader("Инстанс / просмотр упаковки")

    if not OUT_ROOT.exists():
        st.warning(f"OUT_ROOT не найден: {OUT_ROOT}")
        st.stop()

    classes = [c for c in list_dirs(str(OUT_ROOT)) if re.match(r"^j\d+$", c.lower())]
    if not classes:
        st.info("В OUT_ROOT нет папок классов (например j30/j60/...).")
        st.stop()

    cls = st.selectbox("Class", classes, index=0)
    instances = list_instances(str(OUT_ROOT), cls)
    if not instances:
        st.info(f"В классе {cls} нет инстансов.")
        st.stop()

    inst = st.selectbox("Instance", instances, index=0)
    inst_dir = OUT_ROOT / cls / inst

    meta_path = inst_dir / "meta.json"
    results_path = inst_dir / "results.csv"
    summary_path = inst_dir / "summary.csv"
    boxplot_path = inst_dir / "comparison_boxplot.html"

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        st.markdown("### meta.json")
        if meta_path.exists():
            try:
                st.json(json.loads(safe_read_text(meta_path)))
            except Exception:
                st.warning("meta.json есть, но не удалось распарсить JSON.")
        else:
            st.info("meta.json не найден")

        st.markdown("### summary.csv")
        if summary_path.exists():
            st.dataframe(read_csv(str(summary_path)), use_container_width=True, height=260)
        else:
            st.info("summary.csv не найден")

    with colB:
        st.markdown("### results.csv")
        if not results_path.exists():
            st.error("results.csv не найден — нечего анализировать.")
            st.stop()

        df = read_csv(str(results_path))

        # filters inside instance
        decoders = sorted(df["decoder"].dropna().unique().tolist())
        searchers = sorted(df["searcher"].dropna().unique().tolist())
        seeds = sorted(df["seed"].dropna().unique().tolist())

        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        with fcol1:
            dec_pick = st.multiselect("decoder", decoders, default=decoders)
        with fcol2:
            sea_pick = st.multiselect("searcher", searchers, default=searchers)
        with fcol3:
            if ONLY_ONE_SEED:
                seed_pick = [int(ONE_SEED_VALUE)]
                st.text_input("seed (locked by sidebar)", str(seed_pick[0]), disabled=True)
            else:
                seed_pick = st.multiselect("seed", seeds, default=seeds)

        view = df[
            df["decoder"].isin(dec_pick)
            & df["searcher"].isin(sea_pick)
            & df["seed"].isin(seed_pick)
        ].copy()

        if SORT_COL in view.columns:
            view = view.sort_values(SORT_COL, ascending=bool(SORT_ASC))
        else:
            st.warning(f"sort column '{SORT_COL}' не найден в results.csv. Сортировка отключена.")

        st.dataframe(view, use_container_width=True, height=360)

        # quick charts
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("### Boxplot (value) по combo в этом инстансе")
            st.plotly_chart(boxplot_values(view, title=f"{inst} | value distribution"), use_container_width=True)

        with c2:
            st.markdown("### Mean value по combo")
            st.plotly_chart(bar_mean_values(view, title=f"{inst} | mean value by combo"), use_container_width=True)

    st.divider()

    # ============================================================
    # Schedule viewer: pick a row
    # ============================================================
    st.markdown("## Просмотр конкретной упаковки (schedule.html)")

    if "row_select_idx" not in st.session_state:
        st.session_state["row_select_idx"] = 0

    pick_cols = ["decoder", "searcher", "seed", "value", "makespan", "selected", "skipped", "time_sec"]
    pick_cols = [c for c in pick_cols if c in view.columns]
    pick_df = view[pick_cols].reset_index(drop=True)

    if pick_df.empty:
        st.info("После фильтров нет строк. Ослабь фильтры.")
        st.stop()

    idx = st.number_input(
        "row index",
        min_value=0,
        max_value=len(pick_df) - 1,
        value=int(st.session_state["row_select_idx"]),
        step=1,
    )
    st.session_state["row_select_idx"] = int(idx)

    chosen = pick_df.iloc[int(idx)].to_dict()
    st.write("Выбрано:", chosen)

    decoder = str(chosen["decoder"])
    searcher = str(chosen["searcher"])
    seed = int(chosen["seed"])

    sched_path = find_schedule_html_for_row(inst_dir, decoder=decoder, searcher=searcher, seed=seed)
    if sched_path is None:
        st.warning("schedule.html не найден для этого прогона. Если ты сохранял только BEST — открой BEST_seed* в папке.")
    else:
        st.caption(f"Файл: {sched_path}")
        # ВАЖНО: HTML может быть тяжёлым → добавим чекбокс безопасности
        show_html = st.checkbox("Показать schedule.html (может быть тяжёлым)", value=True)
        if show_html:
            components.html(safe_read_text(sched_path), height=680, scrolling=True)

    st.divider()

    # ============================================================
    # SA diagnostics viewer
    # ============================================================
    st.markdown("## SA графики (diagnostics)")
    st.caption("Показывается файл sa_diagnostics_first_run.html, если он был сохранён в decoder=.../search=SA/")

    sa_diag = find_sa_diagnostics_html(inst_dir, decoder=decoder)
    if sa_diag is None:
        st.info("SA diagnostics не найден для этого decoder. (Возможно SA не запускался, или диагностика не сохранялась.)")
    else:
        st.caption(f"Файл: {sa_diag}")
        components.html(safe_read_text(sa_diag), height=600, scrolling=True)

    st.divider()

    # optional: show instance-level saved comparison_boxplot.html
    st.markdown("## comparison_boxplot.html (если есть)")
    if boxplot_path.exists():
        show_cmp = st.checkbox("Показать comparison_boxplot.html (HTML)", value=False)
        if show_cmp:
            components.html(safe_read_text(boxplot_path), height=520, scrolling=True)
    else:
        st.info("comparison_boxplot.html не найден (не критично).")
    

# === all_files.py ===
from pathlib import Path

# корень проекта
ROOT = Path(".")

# список директорий, которые надо пропустить (по имени папки)
EXCLUDED_DIRS = {"venv", ".git", "__pycache__", ".idea"}

OUTPUT_FILE = ROOT / "all_in_one.py"


def is_excluded(path: Path) -> bool:
    # если какая-то часть пути совпадает с именем исключённой директории — пропускаем
    return any(part in EXCLUDED_DIRS for part in path.parts)


def main():
    files = []

    for p in ROOT.rglob("*.py"):
        if p == OUTPUT_FILE:
            continue  # не включаем сам файл сборки
        if is_excluded(p):
            continue  # пропускаем файлы из исключённых директорий
        files.append(p)

    files.sort()  # фиксированный порядок

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for f in files:
            out.write(f"# === {f} ===\n")
            out.write(f.read_text(encoding="utf-8"))
            out.write("\n\n")


if __name__ == "__main__":
    main()


# === app.py ===
# app.py
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Make imports work when running from repo root (Streamlit)
# (We don't run experiments here, but keeping this is harmless)
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# Helpers
# ============================================================
def safe_read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


@st.cache_data(show_spinner=False)
def list_dirs(root: str) -> List[str]:
    p = Path(root)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


@st.cache_data(show_spinner=False)
def list_instances(out_root: str, cls: str) -> List[str]:
    p = Path(out_root) / cls
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_schedule_html_for_row(inst_dir: Path, decoder: str, searcher: str, seed: int) -> Optional[Path]:
    """
    Try in priority:
      1) decoder=.../search=.../seedX/schedule.html
      2) decoder=.../search=.../BEST_seedX/schedule.html
      3) any BEST_seed*/schedule.html for that combo
    """
    combo_dir = inst_dir / f"decoder={decoder}" / f"search={searcher}"
    if not combo_dir.exists():
        return None

    p1 = combo_dir / f"seed{seed}" / "schedule.html"
    if p1.exists():
        return p1

    p2 = combo_dir / f"BEST_seed{seed}" / "schedule.html"
    if p2.exists():
        return p2

    best_any = sorted(combo_dir.glob("BEST_seed*/schedule.html"))
    if best_any:
        return best_any[0]

    # if SAVE_ALL_RUNS=True but seed folder naming differs, try any schedule.html
    any_sched = sorted(combo_dir.glob("**/schedule.html"))
    if any_sched:
        return any_sched[0]

    return None


def find_sa_diagnostics_html(inst_dir: Path, decoder: str) -> Optional[Path]:
    """
    We saved SA diagnostics as:
      decoder=DEC/search=SA/sa_diagnostics_first_run.html
    """
    p = inst_dir / f"decoder={decoder}" / "search=SA" / "sa_diagnostics_first_run.html"
    return p if p.exists() else None


def boxplot_values(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for (dec, sea), g in df.groupby(["decoder", "searcher"]):
        fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
    fig.update_layout(
        title=title,
        xaxis=dict(title="decoder+searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    return fig


def bar_mean_values(df: pd.DataFrame, title: str) -> go.Figure:
    g = (
        df.groupby(["decoder", "searcher"], as_index=False)
        .agg(mean_value=("value", "mean"), best_value=("value", "max"), runs=("value", "count"))
        .sort_values("mean_value", ascending=False)
    )
    x = [f"{d}+{s}" for d, s in zip(g["decoder"], g["searcher"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=g["mean_value"], name="mean_value"))
    fig.update_layout(title=title, xaxis=dict(title="combo"), yaxis=dict(title="mean value"))
    return fig


def agg_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["decoder", "searcher"], as_index=False).agg(
        runs=("value", "count"),
        mean_value=("value", "mean"),
        std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
        best_value=("value", "max"),
        mean_time=("time_sec", "mean"),
        mean_selected=("selected", "mean"),
        mean_skipped=("skipped", "mean"),
        mean_makespan=("makespan", "mean"),
    )
    return g.sort_values(["best_value"], ascending=False)


# ============================================================
# Streamlit UI (viewer-only)
# ============================================================
st.set_page_config(page_title="RCPSP-Marketing | Results Viewer", layout="wide")
st.title("RCPSP-Marketing — Results Viewer")

with st.sidebar:
    st.header("Путь к результатам")
    OUT_ROOT = Path(st.text_input("OUT_ROOT", r"data/experiments/algorithm_comparison"))

    st.divider()
    st.header("Фильтры таблицы")
    ONLY_ONE_SEED = st.checkbox("Показывать только один seed", value=False)
    ONE_SEED_VALUE = st.number_input("seed", min_value=1, value=1, step=1, disabled=not ONLY_ONE_SEED)

    st.divider()
    st.header("Сортировка")
    SORT_COL = st.text_input("sort column", "value")
    SORT_ASC = st.checkbox("ascending", value=False)

tab_overview, tab_instance = st.tabs(["Глобальный обзор", "Инстанс / просмотр упаковки"])


# ============================================================
# Global overview (reads _global if exists, else builds from all results.csv found)
# ============================================================
with tab_overview:
    st.subheader("Глобальный обзор")

    if not OUT_ROOT.exists():
        st.warning(f"OUT_ROOT не найден: {OUT_ROOT}")
    else:
        global_dir = OUT_ROOT / "_global"
        # If _global exists and has all_results.csv -> use it, else build on the fly from results.csv
        all_results_path = global_dir / "all_results.csv"
        if all_results_path.exists():
            all_df = read_csv(str(all_results_path))
            st.caption("Источник: _global/all_results.csv")
        else:
            result_files = sorted(OUT_ROOT.rglob("results.csv"))
            if not result_files:
                st.info("Не найдено ни одного results.csv в OUT_ROOT.")
                st.stop()

            frames = []
            for rf in result_files:
                try:
                    df = read_csv(str(rf))
                    df["results_csv_path"] = str(rf)
                    frames.append(df)
                except Exception:
                    pass

            if not frames:
                st.info("results.csv есть, но прочитать не удалось.")
                st.stop()

            all_df = pd.concat(frames, ignore_index=True)
            st.caption("Источник: сканирование results.csv ( _global/all_results.csv не найден )")

        # Optional global seed filter
        if ONLY_ONE_SEED:
            all_df = all_df[all_df["seed"] == int(ONE_SEED_VALUE)]

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown("### Агрегаты по combo")
            agg = (
                all_df.groupby(["decoder", "searcher"], as_index=False)
                .agg(
                    runs=("value", "count"),
                    mean_value=("value", "mean"),
                    std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
                    best_value=("value", "max"),
                    mean_time=("time_sec", "mean"),
                )
                .sort_values("mean_value", ascending=False)
            )
            st.dataframe(agg, use_container_width=True, height=340)

        with col2:
            st.markdown("### Boxplot по combo (value)")
            fig = go.Figure()
            for (dec, sea), g in all_df.groupby(["decoder", "searcher"]):
                fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
            fig.update_layout(
                title="All results: value distribution by decoder+searcher",
                xaxis=dict(title="combo"),
                yaxis=dict(title="value"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### Топ-строки (по value)")
        show_cols = [
            "class", "instance", "decoder", "searcher", "seed",
            "value", "revenue", "cost", "makespan", "selected", "skipped", "time_sec",
        ]
        topn = st.number_input("Top N", min_value=5, value=30, step=5)
        st.dataframe(
            all_df.sort_values("value", ascending=False)[show_cols].head(int(topn)),
            use_container_width=True,
            height=360,
        )


# ============================================================
# Instance viewer
# ============================================================
with tab_instance:
    st.subheader("Инстанс / просмотр упаковки")

    if not OUT_ROOT.exists():
        st.warning(f"OUT_ROOT не найден: {OUT_ROOT}")
        st.stop()

    classes = [c for c in list_dirs(str(OUT_ROOT)) if not c.startswith("_")]
    if not classes:
        st.info("В OUT_ROOT нет папок классов (например j30/j60/...).")
        st.stop()

    cls = st.selectbox("Class", classes, index=0)
    instances = list_instances(str(OUT_ROOT), cls)
    if not instances:
        st.info(f"В классе {cls} нет инстансов.")
        st.stop()

    inst = st.selectbox("Instance", instances, index=0)
    inst_dir = OUT_ROOT / cls / inst

    meta_path = inst_dir / "meta.json"
    results_path = inst_dir / "results.csv"
    summary_path = inst_dir / "summary.csv"
    boxplot_path = inst_dir / "comparison_boxplot.html"

    colA, colB = st.columns([1, 2], gap="large")

    with colA:
        st.markdown("### meta.json")
        if meta_path.exists():
            st.json(json.loads(safe_read_text(meta_path)))
        else:
            st.info("meta.json не найден")

        st.markdown("### summary.csv")
        if summary_path.exists():
            st.dataframe(read_csv(str(summary_path)), use_container_width=True, height=260)
        else:
            st.info("summary.csv не найден")

    with colB:
        st.markdown("### results.csv")
        if not results_path.exists():
            st.error("results.csv не найден — нечего анализировать.")
            st.stop()

        df = read_csv(str(results_path))

        # ---- filters inside instance
        decoders = sorted(df["decoder"].dropna().unique().tolist())
        searchers = sorted(df["searcher"].dropna().unique().tolist())
        seeds = sorted(df["seed"].dropna().unique().tolist())

        fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
        with fcol1:
            dec_pick = st.multiselect("decoder", decoders, default=decoders)
        with fcol2:
            sea_pick = st.multiselect("searcher", searchers, default=searchers)
        with fcol3:
            if ONLY_ONE_SEED:
                seed_pick = [int(ONE_SEED_VALUE)]
                st.text_input("seed (locked by sidebar)", str(seed_pick[0]), disabled=True)
            else:
                seed_pick = st.multiselect("seed", seeds, default=seeds)

        view = df[
            df["decoder"].isin(dec_pick)
            & df["searcher"].isin(sea_pick)
            & df["seed"].isin(seed_pick)
        ].copy()

        if SORT_COL in view.columns:
            view = view.sort_values(SORT_COL, ascending=bool(SORT_ASC))
        else:
            st.warning(f"sort column '{SORT_COL}' не найден в results.csv. Сортировка отключена.")

        st.dataframe(view, use_container_width=True, height=360)

        # quick charts
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            st.markdown("### Boxplot (value) по combo в этом инстансе")
            st.plotly_chart(boxplot_values(view, title=f"{inst} | value distribution"), use_container_width=True)

        with c2:
            st.markdown("### Mean value по combo")
            st.plotly_chart(bar_mean_values(view, title=f"{inst} | mean value by combo"), use_container_width=True)

    st.divider()

    # ============================================================
    # Schedule viewer: pick a row
    # ============================================================
    st.markdown("## Просмотр конкретной упаковки (schedule.html)")

    if "row_select_idx" not in st.session_state:
        st.session_state["row_select_idx"] = 0

    # show a compact table to pick from
    pick_cols = ["decoder", "searcher", "seed", "value", "makespan", "selected", "skipped", "time_sec"]
    pick_df = view[pick_cols].reset_index(drop=True)

    if pick_df.empty:
        st.info("После фильтров нет строк. Ослабь фильтры.")
        st.stop()

    # select row
    idx = st.number_input("row index", min_value=0, max_value=len(pick_df) - 1, value=int(st.session_state["row_select_idx"]), step=1)
    st.session_state["row_select_idx"] = int(idx)

    chosen = pick_df.iloc[int(idx)].to_dict()
    st.write("Выбрано:", chosen)

    decoder = str(chosen["decoder"])
    searcher = str(chosen["searcher"])
    seed = int(chosen["seed"])

    sched_path = find_schedule_html_for_row(inst_dir, decoder=decoder, searcher=searcher, seed=seed)
    if sched_path is None:
        st.warning("schedule.html не найден для этого прогона. Если ты сохранял только BEST — открой BEST_seed* в папке.")
    else:
        st.caption(f"Файл: {sched_path}")
        components.html(safe_read_text(sched_path), height=680, scrolling=True)

    st.divider()

    # ============================================================
    # SA diagnostics viewer
    # ============================================================
    st.markdown("## SA графики (diagnostics)")
    st.caption("Показывается файл sa_diagnostics_first_run.html, если он был сохранён в decoder=.../search=SA/")

    sa_diag = find_sa_diagnostics_html(inst_dir, decoder=decoder)
    if sa_diag is None:
        st.info("SA diagnostics не найден для этого decoder. (Возможно SA не запускался, или диагностика не сохранялась.)")
    else:
        st.caption(f"Файл: {sa_diag}")
        components.html(safe_read_text(sa_diag), height=600, scrolling=True)

    st.divider()

    # optional: show instance-level saved comparison_boxplot.html
    st.markdown("## comparison_boxplot.html (если есть)")
    if boxplot_path.exists():
        components.html(safe_read_text(boxplot_path), height=520, scrolling=True)
    else:
        st.info("comparison_boxplot.html не найден (не критично).")


# === main.py ===
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import plotly.graph_objects as go

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.scheduling import serial_sgs_selective, parallel_sgs_selective_greedy, parallel_sgs_selective
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.core.improvement import left_shift
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing

def fmt_obj(obj) -> str:
    return f"value={obj.value:_.2f} revenue={obj.revenue:_.2f} cost={obj.cost:_.2f} makespan={obj.makespan}"


def save_sa_history_html(best_hist: List[float], cur_hist: List[float], out_path: Path, title: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best_hist, mode="lines", name="best_value"))
    fig.add_trace(go.Scatter(y=cur_hist, mode="lines", name="current_value"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="iteration"),
        yaxis=dict(title="value"),
        showlegend=True,
    )
    fig.write_html(str(out_path))
    return out_path


def main():
    # ====== параметры ======
    instance_path = r"data\extended\j120.sm\j1201_2_with_metrics.sm"
    T = 80
    seed_order = 42
    seed_sa = 123

    # PSGS-greedy (если хочешь финально упаковать этим декодером)
    greedy_min_score = -1e18
    greedy_unlock_weight = 0.0

    # SA параметры
    sa_iters = 50_000          # 500k на j120 может быть очень долго
    sa_T0 = 2e4
    sa_alpha = 0.9998
    sa_Tmin = 1.0
    sa_neighbor = "insert"      # "swap" быстрее, "insert" сильнее
    sa_tries = 30

    out_dir = Path("data/experiments/viz/sa_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== загрузка проекта ======
    proj = PSPLibExtendedParser().parse(instance_path)
    print("[ok] parsed:", proj.name)
    print("[info] T =", T)

    # ====== декодер для SA: SSGS selective ======
    def decode_ssgs(proj_, order_, T_):
        return serial_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
        )

    # (опционально) декодер для финальной упаковки/визуализации: PSGS greedy
    def decode_psgs_greedy(proj_, order_, T_):
        return parallel_sgs_selective_greedy(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    def decode_psgs(proj_, order_, T_):
        return parallel_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
        )
    # ====== стартовый порядок ======
    order0 = random_topo_sort_fixed_ends(proj, seed=seed_order)
    print("[info] start order len =", len(order0), "first10 =", order0[:10], "last5 =", order0[-5:])

    # ====== baseline (до SA) на SSGS ======
    res0 = decode_ssgs(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline SSGS]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    res0 = decode_psgs(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline PSGS]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    res0 = decode_psgs_greedy(proj, order0, T)
    obj0 = evaluate_profit_over_horizon(proj, res0.schedule, selected_jobs=res0.selected, T=T)
    print("\n[baseline PSGS with gready]", fmt_obj(obj0), "selected=", len(res0.selected), "skipped=", len(res0.skipped))

    fig0 = plot_schedule_gantt(

        proj, res0.schedule,
        selected=res0.selected,
        title=f"{proj.name} | BASELINE SSGS | T={T} | {fmt_obj(obj0)}",
        T=T,
    )
    p0 = save_schedule_html(fig0, out_dir / f"{proj.name}_T{T}_baseline_ssgs.html")
    print("[ok] saved:", p0)

    # ====== SA (оценка через SSGS) ======
    print("\n[sa] start (SSGS decode)...")
    t0 = perf_counter()
    sa = simulated_annealing(
        proj,
        T=T,
        start_order=order0,
        decode_fn=decode_psgs,
        seed=seed_sa,
        iters=sa_iters,
        T0=sa_T0,
        Tmin=sa_Tmin,
        alpha=sa_alpha,
        neighbor=sa_neighbor,
        tries_per_iter=sa_tries,
        keep_history=True,
    )
    dt_sa = perf_counter() - t0
    print("[sa] done in", f"{dt_sa:.3f}s",
          "| accepted=", sa.accepted,
          "| improved_best=", sa.improved_best,
          "| best_value=", f"{sa.best_value:_.2f}",
          "| last_value=", f"{sa.last_value:_.2f}")

    if sa.history_best is not None and sa.history_cur is not None:
        p_hist = save_sa_history_html(
            sa.history_best,
            sa.history_cur,
            out_dir / f"{proj.name}_T{T}_sa_history_ssgs.html",
            title=f"{proj.name} SA history (SSGS) | T={T} | neighbor={sa_neighbor} | alpha={sa_alpha}",
        )
        print("[ok] saved:", p_hist)

    # ====== decode best SA order (SSGS) ======
    res_best = decode_psgs(proj, sa.best_order, T)
    obj_best = evaluate_profit_over_horizon(proj, res_best.schedule, selected_jobs=res_best.selected, T=T)
    print("\n[best after SA | SSGS]", fmt_obj(obj_best), "selected=", len(res_best.selected), "skipped=", len(res_best.skipped))

    fig_best = plot_schedule_gantt(
        proj, res_best.schedule,
        selected=res_best.selected,
        title=f"{proj.name} | BEST after SA (SSGS) | T={T} | {fmt_obj(obj_best)}",
        T=T,
    )
    p_best = save_schedule_html(fig_best, out_dir / f"{proj.name}_T{T}_best_after_sa_ssgs.html")
    print("[ok] saved:", p_best)

    # ====== пост-улучшатель (left_shift) на SSGS-расписании ======
    impL = left_shift(proj, res_best.schedule, selected_jobs=res_best.selected, T=T, hide_dummies=True)
    objL = evaluate_profit_over_horizon(proj, impL.schedule, selected_jobs=res_best.selected, T=T)
    print("\n[post left_shift | SSGS] moved=", impL.moved, "|", fmt_obj(objL))

    figL = plot_schedule_gantt(
        proj, impL.schedule,
        selected=res_best.selected,
        title=f"{proj.name} | AFTER left_shift (SSGS) moved={impL.moved} | T={T} | {fmt_obj(objL)}",
        T=T,
    )
    pL = save_schedule_html(figL, out_dir / f"{proj.name}_T{T}_after_left_shift_ssgs.html")
    print("[ok] saved:", pL)

    # ====== (опционально) “перепаковать” лучший порядок PSGS-greedy и тоже сохранить ======
    try:
        res_g = decode_psgs_greedy(proj, sa.best_order, T)
        obj_g = evaluate_profit_over_horizon(proj, res_g.schedule, selected_jobs=res_g.selected, T=T)
        print("\n[best order repacked | PSGS greedy]", fmt_obj(obj_g), "selected=", len(res_g.selected), "skipped=", len(res_g.skipped))

        fig_g = plot_schedule_gantt(
            proj, res_g.schedule,
            selected=res_g.selected,
            title=f"{proj.name} | BEST order repacked (PSGS greedy) | T={T} | {fmt_obj(obj_g)}",
            T=T,
        )
        p_g = save_schedule_html(fig_g, out_dir / f"{proj.name}_T{T}_best_order_psgs_greedy.html")
        print("[ok] saved:", p_g)
    except Exception as e:
        print("[warn] PSGS greedy repack skipped:", type(e).__name__, e)

    # ====== выбрать лучший вариант по value ======
    candidates: List[Tuple[str, float]] = [
        ("baseline_ssgs", obj0.value),
        ("best_after_sa_ssgs", obj_best.value),
        ("after_left_shift_ssgs", objL.value),
    ]
    best_tag, best_val = max(candidates, key=lambda x: x[1])
    print("\n[best overall] =", best_tag, "value=", f"{best_val:_.2f}")

    print("\n[files]")
    print(" baseline        :", p0)
    print(" best_after_sa   :", p_best)
    print(" after_left      :", pL)
    print(" out_dir         :", out_dir)


if __name__ == "__main__":
    main()


# === scripts\algorithm_comparison.py ===
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, is_topological_order
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html

# --- Decoders (adjust if your functions live elsewhere)
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.scheduling import parallel_sgs_selective, parallel_sgs_selective_greedy

# --- Searchers
from rcpsp_marketing.algorithms.local_search.hill_climb import hill_climb
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def save_schedule_csv(proj: Any, sched: Any, selected: Iterable[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for j in selected:
        if j not in sched.start:
            continue
        rows.append({
            "job": int(j),
            "start": int(sched.start[j]),
            "finish": int(sched.finish[j]),
            "dur": int(sched.finish[j] - sched.start[j]),
            "type": getattr(proj.tasks[j], "job_type", "unknown") if hasattr(proj, "tasks") and j in proj.tasks else "unknown",
            "cost": float(getattr(proj.tasks[j], "total_cost", 0.0)) if hasattr(proj, "tasks") and j in proj.tasks else 0.0,
        })
    df = pd.DataFrame(rows).sort_values(["start", "job"])
    df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def save_order_json(order: List[int], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(order, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def boxplot_values(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for (dec, sea), g in df.groupby(["decoder", "searcher"]):
        fig.add_trace(go.Box(
            y=g["value"],
            name=f"{dec}+{sea}",
            boxmean=True,
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="decoder + searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    return fig


# =========================
# Decoder registry
# =========================

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    decode_fn: Callable[[Any, List[int], int], Any]
    allow_search: bool = True


def make_decoders(*, greedy_min_score: float, greedy_unlock_weight: float) -> List[DecoderSpec]:
    def decode_ssgs(proj: Any, order: List[int], T: int):
        return serial_sgs_selective(proj, order, T=T, include_dummies=True)

    def decode_psgs(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective(proj, order, T=T, include_dummies=True, include_sink=False)

    def decode_psgs_greedy(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_greedy(
            proj, order,
            T=T,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return [
        DecoderSpec("SSGS", decode_ssgs, True),
        DecoderSpec("PSGS", decode_psgs, True),
        DecoderSpec("PSGS_greedy", decode_psgs_greedy, False),  # baseline only
    ]


# =========================
# Search registry
# =========================

@dataclass(frozen=True)
class SearchSpec:
    name: str
    run: Callable[..., Tuple[List[int], Dict[str, Any]]]


def rls_generic(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int,
    iters: int,
    tries_per_iter: int,
    max_profit_evals: Optional[int],
) -> Tuple[List[int], Dict[str, Any]]:
    rnd = __import__("random").Random(seed)

    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    movable_idx = [i for i, j in enumerate(start_order) if j != src and j != snk]
    if len(movable_idx) < 2:
        return list(start_order), {
            "accepted": 0,
            "profit_evals": 0,
            "iters_done": 0,
            "stopped_reason": "not_enough_movable",
        }

    profit_evals = 0
    accepted = 0
    iters_done = 0
    stopped_reason = ""

    def profit_eval(order: List[int]) -> float:
        nonlocal profit_evals
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")
        res = decode_fn(proj, order, T)
        obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)
        profit_evals += 1
        return float(obj.value)

    cur = list(start_order)
    try:
        cur_val = profit_eval(cur)
    except StopIteration:
        return list(start_order), {
            "accepted": 0,
            "profit_evals": profit_evals,
            "iters_done": 0,
            "stopped_reason": "budget",
        }

    best = list(cur)
    best_val = cur_val

    for it in range(iters):
        iters_done = it + 1
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        improved = False
        for _ in range(tries_per_iter):
            if max_profit_evals is not None and profit_evals >= max_profit_evals:
                stopped_reason = "budget"
                break

            i, k = rnd.sample(movable_idx, 2)
            cand = list(cur)
            cand[i], cand[k] = cand[k], cand[i]
            if not is_topological_order(proj, cand):
                continue

            try:
                cand_val = profit_eval(cand)
            except StopIteration:
                stopped_reason = "budget"
                break

            if cand_val > cur_val:
                cur = cand
                cur_val = cand_val
                accepted += 1
                improved = True
                if cand_val > best_val:
                    best_val = cand_val
                    best = list(cand)
                break

        if stopped_reason == "budget":
            break
        if not improved:
            pass

    if not stopped_reason:
        stopped_reason = "iters"

    return best, {
        "accepted": accepted,
        "profit_evals": profit_evals,
        "iters_done": iters_done,
        "stopped_reason": stopped_reason,
    }


def make_searchers(
    *,
    hc_iters: int,
    hc_tries: int,
    hc_neighbor: str,
    sa_iters: int,
    sa_tries: int,
    sa_neighbor: str,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,
    rls_iters: int,
    rls_tries: int,
    max_profit_evals: Optional[int],
) -> List[SearchSpec]:

    def run_none(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        return list(kwargs["start_order"]), {"stopped_reason": "baseline"}

    def run_hc(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        res = hill_climb(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=hc_iters,
            tries_per_iter=hc_tries,
            neighbor=hc_neighbor,
            max_profit_evals=max_profit_evals,
        )
        info = {
            "accepted": getattr(res, "accepted", 0),
            "profit_evals": getattr(res, "profit_evals", 0),
            "iters_done": getattr(res, "iters_done", 0),
            "stopped_reason": getattr(res, "stopped_reason", ""),
        }
        return list(res.best_order), info

    def run_sa(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        res = simulated_annealing(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=sa_iters,
            tries_per_iter=sa_tries,
            neighbor=sa_neighbor,
            T0=sa_T0,
            alpha=sa_alpha,
            Tmin=sa_Tmin,
            keep_history=False,
            max_profit_evals=max_profit_evals,
        )
        info = {
            "accepted": getattr(res, "accepted", 0),
            "profit_evals": getattr(res, "profit_evals", 0),
            "iters_done": getattr(res, "iters_done", 0),
            "stopped_reason": getattr(res, "stopped_reason", ""),
            "improved_best": getattr(res, "improved_best", 0),
        }
        return list(res.best_order), info

    def run_rls(**kwargs) -> Tuple[List[int], Dict[str, Any]]:
        best, info = rls_generic(
            kwargs["proj"],
            T=kwargs["T"],
            start_order=kwargs["start_order"],
            decode_fn=kwargs["decode_fn"],
            seed=kwargs["seed_algo"],
            iters=rls_iters,
            tries_per_iter=rls_tries,
            max_profit_evals=max_profit_evals,
        )
        return list(best), info

    return [
        SearchSpec("none", run_none),
        SearchSpec("HC", run_hc),
        SearchSpec("RLS", run_rls),
        SearchSpec("SA", run_sa),
    ]


# =========================
# Run one instance
# =========================

def run_instance(
    instance_path: Path,
    *,
    out_root: Path,
    T: int,
    seeds: List[int],
    greedy_min_score: float,
    greedy_unlock_weight: float,
    max_profit_evals: Optional[int],
    # HC
    hc_iters: int,
    hc_tries: int,
    hc_neighbor: str,
    # SA
    sa_iters: int,
    sa_tries: int,
    sa_neighbor: str,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,
    # RLS
    rls_iters: int,
    rls_tries: int,
    # saving policy
    save_all_runs: bool,
) -> None:
    proj = PSPLibExtendedParser().parse(instance_path)

    cls = detect_class_from_path(instance_path)
    inst_name = safe_name(proj.name or instance_path.stem)

    inst_dir = out_root / cls / inst_name
    inst_dir.mkdir(parents=True, exist_ok=True)

    decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)
    searchers = make_searchers(
        hc_iters=hc_iters, hc_tries=hc_tries, hc_neighbor=hc_neighbor,
        sa_iters=sa_iters, sa_tries=sa_tries, sa_neighbor=sa_neighbor, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
        rls_iters=rls_iters, rls_tries=rls_tries,
        max_profit_evals=max_profit_evals,
    )

    starts_dir = inst_dir / "start_orders"
    starts_dir.mkdir(parents=True, exist_ok=True)

    start_orders: Dict[int, List[int]] = {}
    for s in seeds:
        order0 = random_topo_sort_fixed_ends(proj, seed=s)
        start_orders[s] = order0
        save_order_json(order0, starts_dir / f"start_order_seed{s}.json")

    rows: List[Dict[str, Any]] = []
    best_per_combo: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}

    for dec in decoders:
        for sea in searchers:
            if (not dec.allow_search) and sea.name != "none":
                continue

            combo_dir = inst_dir / f"decoder={dec.name}" / f"search={sea.name}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            for s in seeds:
                start_order = start_orders[s]
                seed_algo = 10_000 + s

                t0 = perf_counter()
                best_order, info = sea.run(
                    proj=proj,
                    T=T,
                    start_order=start_order,
                    decode_fn=dec.decode_fn,
                    seed_algo=seed_algo,
                )

                res = dec.decode_fn(proj, best_order, T)
                obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)
                dt = perf_counter() - t0

                run_tag = f"seed{s}"
                run_dir = combo_dir / run_tag
                html_path = run_dir / "schedule.html"

                if save_all_runs:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    fig = plot_schedule_gantt(
                        proj, res.schedule,
                        selected=res.selected,
                        title=f"{proj.name} | {dec.name}+{sea.name} | seed={s} | T={T} | value={obj.value:_.2f}",
                        T=T,
                    )
                    save_schedule_html(fig, html_path)
                    save_schedule_csv(proj, res.schedule, res.selected, run_dir / "schedule.csv")
                    save_order_json(best_order, run_dir / "best_order.json")

                row = {
                    "class": cls,
                    "instance": inst_name,
                    "instance_path": str(instance_path),
                    "T": T,
                    "seed": s,
                    "seed_algo": seed_algo,
                    "decoder": dec.name,
                    "searcher": sea.name,

                    "value": float(obj.value),
                    "revenue": float(obj.revenue),
                    "cost": float(obj.cost),
                    "makespan": int(obj.makespan),
                    "selected": len(res.selected),
                    "skipped": len(getattr(res, "skipped", [])),
                    "time_sec": float(dt),

                    "profit_evals": info.get("profit_evals"),
                    "accepted": info.get("accepted"),
                    "iters_done": info.get("iters_done"),
                    "stopped_reason": info.get("stopped_reason"),
                    "improved_best": info.get("improved_best"),

                    "saved_schedule_html": str(html_path) if save_all_runs else "",
                }
                rows.append(row)

                key = (dec.name, sea.name)
                if key not in best_per_combo or obj.value > best_per_combo[key][0]:
                    best_per_combo[key] = (float(obj.value), row)

    df = pd.DataFrame(rows).sort_values(["decoder", "searcher", "seed"])
    (inst_dir / "results.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    # summary
    summary_rows = []
    for (dec_name, sea_name), (best_val, best_row) in best_per_combo.items():
        g = df[(df["decoder"] == dec_name) & (df["searcher"] == sea_name)]
        summary_rows.append({
            "decoder": dec_name,
            "searcher": sea_name,
            "best_value": best_val,
            "mean_value": float(g["value"].mean()),
            "std_value": float(g["value"].std(ddof=1)) if len(g) > 1 else 0.0,
            "best_seed": int(best_row["seed"]),
            "mean_time": float(g["time_sec"].mean()),
            "mean_selected": float(g["selected"].mean()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["decoder", "searcher"])
    (inst_dir / "summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")

    # boxplot
    fig_cmp = boxplot_values(df, title=f"{proj.name} | value distribution | T={T}")
    fig_cmp.write_html(str(inst_dir / "comparison_boxplot.html"))

    # =========================
    # META (added)
    # =========================
    meta = {
        "instance_name": proj.name,
        "instance_path": str(instance_path),
        "class": cls,
        "T": T,
        "seeds": list(seeds),
        "max_profit_evals": max_profit_evals,
        "HC": {"iters": hc_iters, "tries": hc_tries, "neighbor": hc_neighbor},
        "SA": {"iters": sa_iters, "tries": sa_tries, "neighbor": sa_neighbor, "T0": sa_T0, "alpha": sa_alpha, "Tmin": sa_Tmin},
        "RLS": {"iters": rls_iters, "tries": rls_tries},
        "greedy": {"min_score": greedy_min_score, "unlock_weight": greedy_unlock_weight},
        "save_all_runs": save_all_runs,
        "decoders": [d.name for d in decoders],
        "searchers": [s.name for s in searchers],
    }
    (inst_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # save ONLY best schedules if save_all_runs=False
    if not save_all_runs:
        for (dec_name, sea_name), (_, best_row) in best_per_combo.items():
            s = int(best_row["seed"])
            start_order = start_orders[s]
            seed_algo = 10_000 + s

            dec = next(d for d in decoders if d.name == dec_name)
            sea = next(x for x in searchers if x.name == sea_name)

            best_order, _info = sea.run(
                proj=proj,
                T=T,
                start_order=start_order,
                decode_fn=dec.decode_fn,
                seed_algo=seed_algo,
            )
            res = dec.decode_fn(proj, best_order, T)
            obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

            combo_dir = inst_dir / f"decoder={dec_name}" / f"search={sea_name}"
            run_dir = combo_dir / f"BEST_seed{s}"
            run_dir.mkdir(parents=True, exist_ok=True)

            fig = plot_schedule_gantt(
                proj, res.schedule,
                selected=res.selected,
                title=f"{proj.name} | {dec_name}+{sea_name} | BEST seed={s} | T={T} | value={obj.value:_.2f}",
                T=T,
            )
            save_schedule_html(fig, run_dir / "schedule.html")
            save_schedule_csv(proj, res.schedule, res.selected, run_dir / "schedule.csv")
            save_order_json(best_order, run_dir / "best_order.json")

    print("[ok]", proj.name, "->", inst_dir)


def build_global_reports(out_root: Path) -> None:
    """
    Собирает глобальный отчёт по всем уже просчитанным инстансам в out_root:
      - all_results.csv  (все прогоны)
      - by_instance.csv  (агрегаты по каждому файлу/инстансу)
      - by_class.csv     (агрегаты по категориям j30/j60/j90/j120)
      - by_combo.csv     (агрегаты по комбинациям decoder+searcher)
      - by_combo_objective.csv (агрегаты по decoder+searcher+objective)
      - by_objective.csv (агрегаты только по objective)
      - overview_*.html  (boxplot + лидерборды)
      - report.md        (краткий текстовый отчёт)
    """

    out_root = Path(out_root)
    result_files = sorted(out_root.rglob("results.csv"))

    if not result_files:
        print("[report] no results.csv found in:", out_root)
        return

    # 1) Load all run-level results
    frames = []
    for rf in result_files:
        try:
            df = pd.read_csv(rf)
            df["results_csv_path"] = str(rf)
            frames.append(df)
        except Exception as e:
            print("[report][skip]", rf, type(e).__name__, e)

    if not frames:
        print("[report] failed to read any results.csv")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # ---- Backward compatibility: objective column may be missing in older runs
    if "objective" not in all_df.columns:
        all_df["objective"] = "unknown"  # или "ref" если хочешь по умолчанию так
    else:
        # чистим NaN/пустые
        all_df["objective"] = all_df["objective"].fillna("unknown").astype(str)
        all_df.loc[all_df["objective"].str.strip() == "", "objective"] = "unknown"

    global_dir = out_root / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # 2) Save all runs
    all_path = global_dir / "all_results.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8")
    print("[report] saved:", all_path)

    # Helpers
    def agg_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        g = df.groupby(group_cols, as_index=False)
        out = g.agg(
            runs=("value", "count"),
            mean_value=("value", "mean"),
            std_value=("value", lambda x: float(x.std(ddof=1)) if len(x) > 1 else 0.0),
            best_value=("value", "max"),
            mean_time=("time_sec", "mean"),
            mean_selected=("selected", "mean"),
            mean_skipped=("skipped", "mean"),
            mean_makespan=("makespan", "mean"),
        )
        # best seed per group
        idx = df.groupby(group_cols)["value"].idxmax()
        best_rows = df.loc[idx, group_cols + ["seed", "value"]].rename(
            columns={"seed": "best_seed", "value": "best_value_check"}
        )
        out = out.merge(best_rows, on=group_cols, how="left")
        out = out.drop(columns=["best_value_check"])
        return out

    # 3) Per-instance summary
    by_instance = agg_table(all_df, ["class", "instance", "decoder", "searcher"]).sort_values(
        ["class", "instance", "decoder", "searcher"]
    )
    p_inst = global_dir / "by_instance.csv"
    by_instance.to_csv(p_inst, index=False, encoding="utf-8")
    print("[report] saved:", p_inst)

    # 4) By-class summary
    by_class = agg_table(all_df, ["class", "decoder", "searcher"]).sort_values(["class", "decoder", "searcher"])
    p_cls = global_dir / "by_class.csv"
    by_class.to_csv(p_cls, index=False, encoding="utf-8")
    print("[report] saved:", p_cls)

    # 5) By-combo summary (old)
    by_combo = agg_table(all_df, ["decoder", "searcher"]).sort_values(["decoder", "searcher"])
    p_combo = global_dir / "by_combo.csv"
    by_combo.to_csv(p_combo, index=False, encoding="utf-8")
    print("[report] saved:", p_combo)

    # ===== NEW: Objective-aware aggregates =====
    # 5.1) By-objective only
    by_objective = agg_table(all_df, ["objective"]).sort_values(["objective"])
    p_obj = global_dir / "by_objective.csv"
    by_objective.to_csv(p_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_obj)

    # 5.2) By combo + objective (главное для сравнения ref vs fast)
    by_combo_obj = agg_table(all_df, ["decoder", "searcher", "objective"]).sort_values(
        ["decoder", "searcher", "objective"]
    )
    p_combo_obj = global_dir / "by_combo_objective.csv"
    by_combo_obj.to_csv(p_combo_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_combo_obj)

    # 5.3) By class + objective (если хочешь видеть влияние objective на классах)
    by_class_obj = agg_table(all_df, ["class", "objective", "decoder", "searcher"]).sort_values(
        ["class", "objective", "decoder", "searcher"]
    )
    p_class_obj = global_dir / "by_class_objective.csv"
    by_class_obj.to_csv(p_class_obj, index=False, encoding="utf-8")
    print("[report] saved:", p_class_obj)

    # 6) Visualizations
    # 6.1 overall boxplot (all values by combo)
    fig_all = go.Figure()
    for (dec, sea), g in all_df.groupby(["decoder", "searcher"]):
        fig_all.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
    fig_all.update_layout(
        title="All instances: value distribution by decoder+searcher",
        xaxis=dict(title="decoder+searcher"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    p_fig_all = global_dir / "overview_all_boxplot.html"
    fig_all.write_html(str(p_fig_all))

    # 6.2 per-objective boxplots (NEW)
    objectives = sorted(all_df["objective"].dropna().unique().tolist())
    for obj in objectives:
        sub = all_df[all_df["objective"] == obj]
        if sub.empty:
            continue
        fig = go.Figure()
        for (dec, sea), g in sub.groupby(["decoder", "searcher"]):
            fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
        fig.update_layout(
            title=f"Objective={obj}: value distribution by decoder+searcher",
            xaxis=dict(title="decoder+searcher"),
            yaxis=dict(title="value"),
            showlegend=False,
        )
        fig.write_html(str(global_dir / f"overview_objective_{obj}_boxplot.html"))

    # 6.3 “ref vs fast” comparison per combo (NEW)
    # делаем бокс-плоты где имя = dec+sea+objective
    fig_objcmp = go.Figure()
    for (dec, sea, obj), g in all_df.groupby(["decoder", "searcher", "objective"]):
        fig_objcmp.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}+{obj}", boxmean=True))
    fig_objcmp.update_layout(
        title="Objective-aware: value distribution by decoder+searcher+objective",
        xaxis=dict(title="decoder+searcher+objective"),
        yaxis=dict(title="value"),
        showlegend=False,
    )
    fig_objcmp.write_html(str(global_dir / "overview_combo_objective_boxplot.html"))

    # 6.4 per-class boxplots (old)
    classes = sorted(all_df["class"].dropna().unique().tolist())
    for cls in classes:
        sub = all_df[all_df["class"] == cls]
        if sub.empty:
            continue
        fig = go.Figure()
        for (dec, sea), g in sub.groupby(["decoder", "searcher"]):
            fig.add_trace(go.Box(y=g["value"], name=f"{dec}+{sea}", boxmean=True))
        fig.update_layout(
            title=f"{cls}: value distribution by decoder+searcher",
            xaxis=dict(title="decoder+searcher"),
            yaxis=dict(title="value"),
            showlegend=False,
        )
        fig.write_html(str(global_dir / f"overview_{cls}_boxplot.html"))

    # 6.5 leaderboards
    top_mean = by_combo.sort_values("mean_value", ascending=False).head(15)
    top_best = by_combo.sort_values("best_value", ascending=False).head(15)

    # NEW leaderboards: per objective
    top_mean_obj = by_combo_obj.sort_values("mean_value", ascending=False).head(20)
    top_best_obj = by_combo_obj.sort_values("best_value", ascending=False).head(20)

    def bar_leaderboard_combo(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
        x = [f"{d}+{s}" for d, s in zip(df["decoder"], df["searcher"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=df[metric]))
        fig.update_layout(title=title, xaxis=dict(title="combo"), yaxis=dict(title=metric))
        return fig

    def bar_leaderboard_combo_obj(df: pd.DataFrame, metric: str, title: str) -> go.Figure:
        x = [f"{d}+{s}+{o}" for d, s, o in zip(df["decoder"], df["searcher"], df["objective"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=df[metric]))
        fig.update_layout(title=title, xaxis=dict(title="combo+objective"), yaxis=dict(title=metric))
        return fig

    bar_leaderboard_combo(top_mean, "mean_value", "Top combos by MEAN value").write_html(
        str(global_dir / "leaderboard_top_mean.html")
    )
    bar_leaderboard_combo(top_best, "best_value", "Top combos by BEST value").write_html(
        str(global_dir / "leaderboard_top_best.html")
    )

    # NEW
    bar_leaderboard_combo_obj(top_mean_obj, "mean_value", "Top combo+objective by MEAN value").write_html(
        str(global_dir / "leaderboard_top_mean_objective.html")
    )
    bar_leaderboard_combo_obj(top_best_obj, "best_value", "Top combo+objective by BEST value").write_html(
        str(global_dir / "leaderboard_top_best_objective.html")
    )

    # 7) Markdown short report
    n_instances = all_df[["class", "instance"]].drop_duplicates().shape[0]
    n_runs = len(all_df)
    combos = all_df[["decoder", "searcher"]].drop_duplicates().shape[0]
    obj_count = all_df[["objective"]].drop_duplicates().shape[0]

    lines = []
    lines.append("# Algorithm comparison report\n")
    lines.append(f"- Instances: **{n_instances}**")
    lines.append(f"- Runs total: **{n_runs}**")
    lines.append(f"- Unique combos (decoder+searcher): **{combos}**")
    lines.append(f"- Objectives present: **{obj_count}** ({', '.join(objectives)})")
    lines.append("")

    lines.append("## By objective (mean/best)")
    for _, r in by_objective.iterrows():
        lines.append(f"- objective={r['objective']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Top-10 combos by mean value (objective-agnostic)")
    for _, r in by_combo.sort_values("mean_value", ascending=False).head(10).iterrows():
        lines.append(f"- {r['decoder']}+{r['searcher']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Top-10 combo+objective by mean value")
    for _, r in by_combo_obj.sort_values("mean_value", ascending=False).head(10).iterrows():
        lines.append(f"- {r['decoder']}+{r['searcher']}+{r['objective']}: mean={r['mean_value']:.4f}, best={r['best_value']:.4f}, runs={int(r['runs'])}")
    lines.append("")

    lines.append("## Files generated")
    lines.append(f"- {all_path.name}")
    lines.append(f"- {p_inst.name}")
    lines.append(f"- {p_cls.name}")
    lines.append(f"- {p_combo.name}")
    lines.append(f"- {p_obj.name}")
    lines.append(f"- {p_combo_obj.name}")
    lines.append(f"- by_class_objective.csv")
    lines.append(f"- overview_all_boxplot.html")
    lines.append(f"- overview_objective_<objective>_boxplot.html")
    lines.append(f"- overview_combo_objective_boxplot.html")
    lines.append(f"- overview_<class>_boxplot.html")
    lines.append(f"- leaderboard_top_mean.html")
    lines.append(f"- leaderboard_top_best.html")
    lines.append(f"- leaderboard_top_mean_objective.html")
    lines.append(f"- leaderboard_top_best_objective.html")

    rep_path = global_dir / "report.md"
    rep_path.write_text("\n".join(lines), encoding="utf-8")
    print("[report] saved:", rep_path)
    print("[report] global dir:", global_dir)


# =========================
# MAIN (edit params here)
# =========================
def main():
    # ====== CONFIG: edit here ======
    MODE = "dir"   # "single" | "dir"

    # --- single mode ---
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # --- dir mode ---
    IN_DIR = Path(r"data/extended/j60.sm")
    PATTERN = "*_with_metrics.sm"

    OUT_ROOT = Path(r"data/experiments/algorithm_comparison")

    T = 50
    SEEDS = list(range(1, 11))          # 10 запусков (paired starts)
    SAVE_ALL_RUNS = False               # True = сохранять расписание для каждого прогона (много файлов)

    # compute budget (calls to profit). None = no budget
    MAX_PROFIT_EVALS: Optional[int] = 50_000

    # PSGS greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0

    # HC params
    HC_ITERS = 10_000_000
    HC_TRIES = 50
    HC_NEIGHBOR = "insert"              # "swap" or "insert"

    # SA params
    SA_ITERS = 10_000_000
    SA_TRIES = 30
    SA_NEIGHBOR = "insert"
    SA_T0 = 2e5
    SA_ALPHA = 0.9998
    SA_TMIN = 1.0

    # RLS params
    RLS_ITERS = 10_000_000
    RLS_TRIES = 20

    # optional limit (only for dir mode; 0 = no limit)
    MAX_INSTANCES = 10
    # ====== END CONFIG ======

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)}")

    for p in files:
        if not p.exists() or not p.is_file():
            print("[skip] not found or not a file:", p)
            continue

        print("\n==============================================")
        print("[instance]", p)

        run_instance(
            p,
            out_root=OUT_ROOT,
            T=T,
            seeds=SEEDS,
            greedy_min_score=GREEDY_MIN_SCORE,
            greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
            max_profit_evals=MAX_PROFIT_EVALS,

            hc_iters=HC_ITERS,
            hc_tries=HC_TRIES,
            hc_neighbor=HC_NEIGHBOR,

            sa_iters=SA_ITERS,
            sa_tries=SA_TRIES,
            sa_neighbor=SA_NEIGHBOR,
            sa_T0=SA_T0,
            sa_alpha=SA_ALPHA,
            sa_Tmin=SA_TMIN,

            rls_iters=RLS_ITERS,
            rls_tries=RLS_TRIES,

            save_all_runs=SAVE_ALL_RUNS,
        )
    
    build_global_reports(OUT_ROOT)


if __name__ == "__main__":
    start_time = perf_counter()
    main()
    finish_time = perf_counter()
    elapsed = finish_time - start_time
    print(f"\n[total time] {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

# === scripts\print_topo.py ===
from pathlib import Path

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.io.psplib_base import PSPLibBaseParser
from rcpsp_marketing.core.precedence import topo_sort, is_topological_order, order_without_dummies, random_topo_sort


def main():
    # поменяй на нужный файл
    path = Path(r"data\extended\j30.sm\j301_1_with_metrics.sm")

    if "with_metrics" in path.name:
        proj = PSPLibExtendedParser().parse(path)
        print("[ok] parsed EXTENDED")
    else:
        proj = PSPLibBaseParser().parse(path)
        print("[ok] parsed RAW")

    for s in [1, 2, 3]:
        order = random_topo_sort(proj, seed=s)
        print(f"\nseed={s} ok={is_topological_order(proj, order)} len={len(order)}")
        print("full order:", order)
        print("real only :", order_without_dummies(proj, order)) # без фиктивных задач


if __name__ == "__main__":
    main()


# === src\__init__.py ===


# === src\rcpsp_marketing\__init__.py ===


# === src\rcpsp_marketing\algorithms\__init__.py ===
"""
Алгоритмы оптимизации:
- локальный поиск (SA, hill-climbing, RLS),
- точный перебор,
- аналитика топологических порядков.
"""













# === src\rcpsp_marketing\algorithms\local_search\__init__.py ===
"""Пакет локального поиска (hill climbing, SA, RLS)."""













# === src\rcpsp_marketing\algorithms\local_search\hill_climb.py ===
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Any, Literal

from rcpsp_marketing.core.precedence import is_topological_order
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert", "mixed"]


@dataclass(slots=True)
class HCResult:
    best_order: List[int]
    best_value: float
    best_obj: object
    iters: int                 # requested iters
    iters_done: int            # executed iters
    accepted: int              # how many improving moves accepted
    profit_evals: int          # calls to objective function
    stopped_reason: str        # "budget" | "iters" | "no_improve"


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def hill_climb(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int = 0,
    iters: int = 50_000,

    # neighborhood
    neighbor: NeighborType = "swap",          # "swap" | "insert" | "mixed"
    tries_per_iter: int = 50,                 # attempts to find a valid neighbor per iteration
    p_swap: float = 0.8,                      # only for mixed
    insert_max_shift: Optional[int] = None,   # local insert window (e.g. 10)

    # budget
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",      # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,
    include_dummy_costs: bool = False,
) -> HCResult:
    """
    Hill Climbing (first-improvement) on priority list.

    - Only accepts strictly improving moves.
    - Stops when:
        * budget reached (max_profit_evals),
        * iters reached,
        * no improving neighbor found in this iteration (local optimum).

    Budget counts ONLY calls to objective_fn (includes initial evaluation).
    """

    # choose objective
    if objective_fn is None:
        if objective == "ref":
            objective_fn = evaluate_profit_over_horizon
        elif objective == "fast":
            objective_fn = evaluate_profit_over_horizon_fast
        else:
            raise ValueError(f"Unknown objective='{objective}'. Use 'ref' or 'fast' or pass objective_fn=...")

    if neighbor == "mixed" and not (0.0 <= p_swap <= 1.0):
        raise ValueError("p_swap must be in [0, 1] for neighbor='mixed'")

    rnd = random.Random(seed)

    profit_evals = 0

    def _profit_eval(order: List[int]) -> tuple[float, object]:
        nonlocal profit_evals
        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")

        res = decode_fn(proj, order, T)
        obj = objective_fn(
            proj,
            res.schedule,
            selected_jobs=res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        profit_evals += 1
        return float(obj.value), obj

    def propose_neighbor(order: List[int]) -> Optional[List[int]]:
        """
        Try up to tries_per_iter to find a topologically valid neighbor.
        """
        src = getattr(proj, "source_id", None)
        snk = getattr(proj, "sink_id", None)

        def do_swap(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False
            i, k = rnd.sample(mov, 2)
            _swap(cand, i, k)
            return True

        def do_insert(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False

            i = rnd.choice(mov)

            if insert_max_shift is None:
                k = rnd.choice([x for x in mov if x != i])
            else:
                lo = max(0, i - insert_max_shift)
                hi = min(len(cand) - 1, i + insert_max_shift)
                window = [x for x in mov if x != i and lo <= x <= hi]
                k = rnd.choice(window) if window else rnd.choice([x for x in mov if x != i])

            if i == k:
                return False

            job = cand.pop(i)
            if k > i:
                k -= 1
            cand.insert(k, job)
            return True

        for _ in range(tries_per_iter):
            cand = list(order)
            mov = _make_movable_indices(proj, cand)
            if len(mov) < 2:
                return None

            if neighbor == "swap":
                ok = do_swap(cand, mov)
            elif neighbor == "insert":
                ok = do_insert(cand, mov)
            elif neighbor == "mixed":
                step = "swap" if rnd.random() < p_swap else "insert"
                ok = do_swap(cand, mov) if step == "swap" else do_insert(cand, mov)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if ok and is_topological_order(proj, cand):
                return cand

        return None

    # initial evaluation
    cur_order = list(start_order)
    try:
        cur_val, cur_obj = _profit_eval(cur_order)
    except StopIteration:
        return HCResult(
            best_order=list(cur_order),
            best_value=float("-inf"),
            best_obj=None,
            iters=iters,
            iters_done=0,
            accepted=0,
            profit_evals=profit_evals,
            stopped_reason="budget",
        )

    best_order = list(cur_order)
    best_val = cur_val
    best_obj = cur_obj

    accepted = 0
    iters_done = 0
    stopped_reason = ""

    for it in range(iters):
        iters_done = it + 1

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        cand_order = propose_neighbor(cur_order)
        if cand_order is None:
            stopped_reason = "no_improve"
            break

        try:
            cand_val, cand_obj = _profit_eval(cand_order)
        except StopIteration:
            stopped_reason = "budget"
            break

        if cand_val > cur_val:
            # accept (first improvement)
            cur_order = cand_order
            cur_val = cand_val
            cur_obj = cand_obj
            accepted += 1

            if cand_val > best_val:
                best_val = cand_val
                best_obj = cand_obj
                best_order = list(cand_order)
        else:
            # для HC "плохие" соседи не принимаем; но продолжаем итерации
            # (если хочешь "до первого улучшения" внутри итерации — тогда возвращаемся к старой схеме)
            pass

    if not stopped_reason:
        stopped_reason = "iters"

    return HCResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        iters_done=iters_done,
        accepted=accepted,
        profit_evals=profit_evals,
        stopped_reason=stopped_reason,
    )


# === src\rcpsp_marketing\algorithms\local_search\randomized_local_search.py ===
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Any, Callable, Literal

from rcpsp_marketing.core.precedence import is_topological_order
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert"]


@dataclass(slots=True)
class RLSResult:
    best_order: List[int]
    best_value: float
    best_obj: object
    iters: int
    accepted: int

    profit_evals: int = 0
    iters_done: int = 0
    stopped_reason: str = ""


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _insert(order: List[int], i: int, k: int) -> None:
    """
    Move element at index i to index k (Python-style insert after pop).
    If i < k, after pop indices shift left, so target becomes k-1.
    """
    if i == k:
        return
    x = order.pop(i)
    if i < k:
        k -= 1
    order.insert(k, x)


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def randomized_local_search(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    seed: int = 0,
    iters: int = 2000,
    tries_per_iter: int = 1,

    # neighbor type
    neighbor: NeighborType = "swap",
    insert_max_shift: Optional[int] = None,  # например 10, чтобы insert был локальным

    # compute budget
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",  # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,
    include_dummy_costs: bool = False,
) -> RLSResult:
    """
    Randomized Local Search (first improvement):
    - сосед: swap или insert (move)
    - принимаем, если стало лучше
    """

    # choose objective
    if objective_fn is None:
        if objective == "ref":
            objective_fn = evaluate_profit_over_horizon
        elif objective == "fast":
            objective_fn = evaluate_profit_over_horizon_fast
        else:
            raise ValueError(f"Unknown objective='{objective}'. Use 'ref' or 'fast' or pass objective_fn=...")

    rnd = random.Random(seed)

    movable_idx = _make_movable_indices(proj, start_order)
    if len(movable_idx) < 2:
        raise ValueError("Not enough movable jobs to perform moves (need >=2).")

    cur_order = list(start_order)

    profit_evals = 0
    accepted = 0
    iters_done = 0
    stopped_reason = ""

    def _profit_eval(order: List[int]) -> tuple[float, object]:
        nonlocal profit_evals

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            raise StopIteration("budget")

        res = serial_sgs_selective(proj, order, T=T, include_dummies=True)
        obj = objective_fn(
            proj,
            res.schedule,
            selected_jobs=res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        profit_evals += 1
        return float(obj.value), obj

    def _sample_two_indices() -> tuple[int, int]:
        """
        Берём два индекса для соседа.
        Для insert можно ограничить дальность сдвига insert_max_shift (локальный insert).
        """
        if neighbor == "swap" or insert_max_shift is None:
            return tuple(rnd.sample(movable_idx, 2))  # type: ignore

        # local insert: выбираем i, потом k близко к i
        i = rnd.choice(movable_idx)
        # кандидаты k в окне
        lo = max(0, i - insert_max_shift)
        hi = min(len(cur_order) - 1, i + insert_max_shift)
        candidates = [k for k in range(lo, hi + 1) if k != i and cur_order[k] not in (getattr(proj, "source_id", None), getattr(proj, "sink_id", None))]
        if not candidates:
            # fallback
            k = rnd.choice([x for x in movable_idx if x != i])
            return i, k
        return i, rnd.choice(candidates)

    # --- initial evaluation
    try:
        cur_val, cur_obj = _profit_eval(cur_order)
    except StopIteration:
        return RLSResult(
            best_order=list(cur_order),
            best_value=float("-inf"),
            best_obj=None,
            iters=iters,
            accepted=0,
            profit_evals=profit_evals,
            iters_done=0,
            stopped_reason="budget",
        )

    best_order = list(cur_order)
    best_obj = cur_obj
    best_val = cur_val

    # --- main loop
    for it in range(iters):
        iters_done = it + 1

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_reason = "budget"
            break

        improved = False

        for _ in range(tries_per_iter):
            if max_profit_evals is not None and profit_evals >= max_profit_evals:
                stopped_reason = "budget"
                break

            i, k = _sample_two_indices()
            cand = list(cur_order)

            if neighbor == "swap":
                _swap(cand, i, k)
            elif neighbor == "insert":
                _insert(cand, i, k)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if not is_topological_order(proj, cand):
                continue

            try:
                cand_val, cand_obj = _profit_eval(cand)
            except StopIteration:
                stopped_reason = "budget"
                break

            if cand_val > cur_val:
                cur_order = cand
                cur_val = cand_val
                cur_obj = cand_obj
                accepted += 1
                improved = True

                if cand_val > best_val:
                    best_val = cand_val
                    best_obj = cand_obj
                    best_order = list(cand)

                break  # first improvement

        if stopped_reason == "budget":
            break

        # если нет улучшения — остаёмся на месте
        if not improved:
            pass

    if not stopped_reason:
        stopped_reason = "iters"

    return RLSResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        accepted=accepted,
        profit_evals=profit_evals,
        iters_done=iters_done,
        stopped_reason=stopped_reason,
    )


# === src\rcpsp_marketing\algorithms\local_search\simulated_annealing.py ===
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Literal

from rcpsp_marketing.core.precedence import is_topological_order

# оба варианта одной и той же цели
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

NeighborType = Literal["swap", "insert", "mixed"]


@dataclass(slots=True)
class SAResult:
    best_order: List[int]
    best_value: float
    best_obj: object

    iters: int
    accepted: int
    improved_best: int
    last_value: float

    profit_evals: int
    stopped_by_budget: bool

    # histories (downsampled by log_every)
    history_it: Optional[List[int]] = None
    history_temp: Optional[List[float]] = None
    history_best: Optional[List[float]] = None
    history_cur: Optional[List[float]] = None
    history_accept: Optional[List[int]] = None          # 1 accepted else 0
    history_delta: Optional[List[float]] = None         # cand-cur (0 if no cand)
    history_selected: Optional[List[int]] = None        # len(selected)
    history_makespan: Optional[List[int]] = None        # schedule.makespan
    history_no_neighbor: Optional[List[int]] = None     # 1 if no valid neighbor found
    history_eval_ms: Optional[List[float]] = None       # время оценки цели (ms), если включишь


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def simulated_annealing(
    proj: Any,
    *,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed: int = 0,
    iters: int = 50_000,

    # temperature schedule
    T0: float = 1.0,
    Tmin: float = 1e-6,
    alpha: float = 0.9995,

    # neighborhood
    neighbor: NeighborType = "swap",          # "swap" | "insert" | "mixed"
    tries_per_iter: int = 20,                 # attempts to find valid neighbor
    p_swap: float = 0.8,                      # only for mixed: P(use swap)
    insert_max_shift: Optional[int] = None,   # local insert window (e.g. 10)

    # compute budget (limit number of objective evaluations)
    max_profit_evals: Optional[int] = None,

    # objective selection
    objective: str = "ref",       # "ref" | "fast"
    objective_fn: Optional[Callable[..., Any]] = None,  # если хочешь подать свою функцию напрямую
    include_dummy_costs: bool = False,

    # logging
    keep_history: bool = False,
    log_every: int = 200,         # store one point each N iterations
    log_eval_time: bool = False,  # писать время оценки (чтобы сравнить ref vs fast)
) -> SAResult:
    """
    SA for priority list.

    decode_fn(proj, order, T) -> object with fields:
      - schedule
      - selected (iterable job ids)

    objective:
      - "ref": evaluate_profit_over_horizon
      - "fast": evaluate_profit_over_horizon_fast
      - or pass objective_fn=...

    Budget:
      profit evaluation == one call to objective_fn(...)
    """
    import time  # локально, чтобы не тащить всегда

    rnd = random.Random(seed)

    # choose objective
    if objective_fn is None:
        if objective == "ref":
            objective_fn = evaluate_profit_over_horizon
        elif objective == "fast":
            objective_fn = evaluate_profit_over_horizon_fast
        else:
            raise ValueError(f"Unknown objective='{objective}'. Use 'ref' or 'fast' or pass objective_fn=...")

    if neighbor == "mixed" and not (0.0 <= p_swap <= 1.0):
        raise ValueError("p_swap must be in [0, 1] for neighbor='mixed'")

    cur_order = list(start_order)

    # histories (downsampled)
    hist_it = [] if keep_history else None
    hist_temp = [] if keep_history else None
    hist_best = [] if keep_history else None
    hist_cur = [] if keep_history else None
    hist_accept = [] if keep_history else None
    hist_delta = [] if keep_history else None
    hist_selected = [] if keep_history else None
    hist_makespan = [] if keep_history else None
    hist_no_neighbor = [] if keep_history else None
    hist_eval_ms = [] if (keep_history and log_eval_time) else None

    def log_point(
        it: int, *,
        temp_: float, best_: float, cur_: float,
        acc: int, delta_: float,
        sel: int, ms: int, no_nb: int,
        eval_ms: float = 0.0,
    ) -> None:
        if not keep_history:
            return
        if log_every <= 1 or it % log_every == 0:
            hist_it.append(it)
            hist_temp.append(float(temp_))
            hist_best.append(float(best_))
            hist_cur.append(float(cur_))
            hist_accept.append(int(acc))
            hist_delta.append(float(delta_))
            hist_selected.append(int(sel))
            hist_makespan.append(int(ms))
            hist_no_neighbor.append(int(no_nb))
            if hist_eval_ms is not None:
                hist_eval_ms.append(float(eval_ms))

    def propose_neighbor(order: List[int]) -> Optional[List[int]]:
        src = getattr(proj, "source_id", None)
        snk = getattr(proj, "sink_id", None)

        def movable_indices(cur: List[int]) -> List[int]:
            # пересчитываем каждый раз от текущего порядка (важно для insert)
            return [idx for idx, j in enumerate(cur) if j != src and j != snk]

        def do_swap(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False
            i, k = rnd.sample(mov, 2)
            _swap(cand, i, k)
            return True

        def do_insert(cand: List[int], mov: List[int]) -> bool:
            if len(mov) < 2:
                return False

            i = rnd.choice(mov)

            if insert_max_shift is None:
                k = rnd.choice([x for x in mov if x != i])
            else:
                lo = max(0, i - insert_max_shift)
                hi = min(len(cand) - 1, i + insert_max_shift)
                window = [x for x in mov if x != i and lo <= x <= hi]
                if not window:
                    k = rnd.choice([x for x in mov if x != i])
                else:
                    k = rnd.choice(window)

            # move i -> k
            job = cand.pop(i)
            if k > i:
                k -= 1
            cand.insert(k, job)
            return True

        for _ in range(tries_per_iter):
            cand = list(order)
            mov = movable_indices(cand)
            if len(mov) < 2:
                return None

            # выбираем тип шага
            if neighbor == "swap":
                ok = do_swap(cand, mov)
            elif neighbor == "insert":
                ok = do_insert(cand, mov)
            elif neighbor == "mixed":
                step = "swap" if rnd.random() < p_swap else "insert"
                ok = do_swap(cand, mov) if step == "swap" else do_insert(cand, mov)
            else:
                raise ValueError(f"Unknown neighbor='{neighbor}'")

            if not ok:
                continue

            if is_topological_order(proj, cand):
                return cand

        return None

    # baseline eval
    cur_res = decode_fn(proj, cur_order, T)

    t0 = time.perf_counter()
    cur_obj = objective_fn(
        proj,
        cur_res.schedule,
        selected_jobs=cur_res.selected,
        T=T,
        include_dummy_costs=include_dummy_costs,
    )
    eval_ms0 = (time.perf_counter() - t0) * 1000.0

    profit_evals = 1
    cur_val = float(cur_obj.value)

    best_order = list(cur_order)
    best_obj = cur_obj
    best_val = cur_val

    accepted = 0
    improved_best = 0
    stopped_by_budget = False

    temp = float(T0)

    # initial log
    log_point(
        0,
        temp_=temp,
        best_=best_val,
        cur_=cur_val,
        acc=1,
        delta_=0.0,
        sel=len(getattr(cur_res, "selected", [])),
        ms=int(getattr(cur_res, "schedule").makespan),
        no_nb=0,
        eval_ms=eval_ms0,
    )

    for it in range(1, iters + 1):
        if temp < Tmin:
            break

        if max_profit_evals is not None and profit_evals >= max_profit_evals:
            stopped_by_budget = True
            break

        cand_order = propose_neighbor(cur_order)

        if cand_order is None:
            temp *= alpha
            log_point(
                it,
                temp_=temp,
                best_=best_val,
                cur_=cur_val,
                acc=0,
                delta_=0.0,
                sel=len(getattr(cur_res, "selected", [])),
                ms=int(getattr(cur_res, "schedule").makespan),
                no_nb=1,
                eval_ms=0.0,
            )
            continue

        cand_res = decode_fn(proj, cand_order, T)

        t1 = time.perf_counter()
        cand_obj = objective_fn(
            proj,
            cand_res.schedule,
            selected_jobs=cand_res.selected,
            T=T,
            include_dummy_costs=include_dummy_costs,
        )
        eval_ms = (time.perf_counter() - t1) * 1000.0

        profit_evals += 1
        cand_val = float(cand_obj.value)
        delta = cand_val - cur_val

        # accept rule
        if delta >= 0.0:
            accept = True
        else:
            p = math.exp(delta / max(1e-12, temp))
            accept = (rnd.random() < p)

        acc_flag = 1 if accept else 0

        if accept:
            cur_order = cand_order
            cur_res = cand_res
            cur_obj = cand_obj
            cur_val = cand_val
            accepted += 1

            if cand_val > best_val:
                best_val = cand_val
                best_obj = cand_obj
                best_order = list(cand_order)
                improved_best += 1

        temp *= alpha

        log_point(
            it,
            temp_=temp,
            best_=best_val,
            cur_=cur_val,
            acc=acc_flag,
            delta_=delta,
            sel=len(getattr(cur_res, "selected", [])),
            ms=int(getattr(cur_res, "schedule").makespan),
            no_nb=0,
            eval_ms=eval_ms,
        )

    return SAResult(
        best_order=best_order,
        best_value=best_val,
        best_obj=best_obj,
        iters=iters,
        accepted=accepted,
        improved_best=improved_best,
        last_value=cur_val,
        profit_evals=profit_evals,
        stopped_by_budget=stopped_by_budget,
        history_it=hist_it,
        history_temp=hist_temp,
        history_best=hist_best,
        history_cur=hist_cur,
        history_accept=hist_accept,
        history_delta=hist_delta,
        history_selected=hist_selected,
        history_makespan=hist_makespan,
        history_no_neighbor=hist_no_neighbor,
        history_eval_ms=hist_eval_ms,
    )


# === src\rcpsp_marketing\cli\__init__.py ===


# === src\rcpsp_marketing\cli\generate_instances.py ===
import random
import re
import math
from typing import Dict, List, Tuple

# Диапазоны цен на ресурсы (за 1 ед. ресурса в 1 ед. времени)
# Ключи — индексы ресурсов из PSPLib (R1..Rk → 1..k)
RESOURCE_COST_RANGES = {
    1: (100.0, 250.0),   # R1: труд/дни разработки
    2: (80.0, 200.0),    # R2: маркетинг/баинг
    3: (50.0, 120.0),    # R3: инфраструктура/серверы
    4: (30.0, 80.0),     # R4: прочие/накладные
    # при наличии >4 ресурсов диапазоны для 5+ сгенерируются автоматически
}

# Вероятности типов задач
JOB_TYPES_PROBS = {
    "marketing": 0.40,
    "product":   0.25,
    "opt":       0.20,
    "partner":   0.10,
    "internal":  0.05,
}

# Эффекты типов (мультипликаторы вокруг 1.0 с шумом)
# Запись: var -> (mean_mult, std_of_noise), где noise ~ N(0, std)
EFFECTS = {
    "marketing": {"AC": (1.08, 0.05), "CPC": (1.04, 0.03), "LCR": (0.99, 0.01), "PCR": (0.99, 0.01), "LT": (1.00, 0.01)},
    "product":   {"AC": (1.01, 0.02), "CPC": (1.00, 0.01), "LCR": (1.04, 0.03), "PCR": (1.05, 0.03), "LT": (1.01, 0.01)},
    "opt":       {"AC": (1.00, 0.01), "CPC": (0.94, 0.03), "LCR": (1.01, 0.01), "PCR": (1.00, 0.01), "LT": (1.00, 0.01)},
    "partner":   {"AC": (1.03, 0.02), "CPC": (1.00, 0.01), "LCR": (1.01, 0.02), "PCR": (1.01, 0.02), "LT": (1.00, 0.01)},
    "internal":  {"AC": (1.00, 0.005), "CPC": (1.00, 0.005), "LCR": (1.00, 0.005), "PCR": (1.00, 0.005), "LT": (1.00, 0.005)},
}

# Коридоры значений (клипперы), чтобы метрики не «улетали»
BOUNDS = {
    "LT":  (50.0, 1500.0),
    "AC":  (0,  10_000_000),
    "CPC": (0.05, 5.0),
    "LCR": (0.005, 0.50),
    "PCR": (0.01, 0.60),
}

# Базовые «реалистичные» медианы для инициализации
INIT_MEDIANS = {
    "LT": 220.0,
    "AC": 1200.0,
    "CPC": 0.55,
    "LCR": 0.12,
    "PCR": 0.18,
}

# Степень масштабирования эффекта длительностью: scale = (dur/median_dur)^DUR_POW
DUR_POW = 0.5

# Концентрации для бета-распределений (чем больше, тем уже разброс)
BETA_KAPPA = 90.0

# Интенсивность насыщения рынка (чем больше, тем сильнее «тормозит» рост AC)
# Реальный K рынка возьмем пропорционально стартовому AC (ниже)
MARKET_K_MULTIPLIER = 250.0


# ============================================================
# Вспомогательные функции распределений и клипов
# ============================================================

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def sample_lognormal(median: float, iqr_ratio: float = 1.8) -> float:
    # iqr_ratio ~ e^{1.349*σ} → σ ≈ ln(iqr_ratio)/1.349
    sigma = math.log(iqr_ratio)/1.349
    mu = math.log(median)
    return math.exp(random.gauss(mu, sigma))

def sample_beta(mean: float, kappa: float = BETA_KAPPA) -> float:
    mean = clip(mean, 1e-6, 1 - 1e-6)
    alpha = mean * kappa
    beta = (1 - mean) * kappa
    x = random.gammavariate(alpha, 1.0)
    y = random.gammavariate(beta, 1.0)
    return x/(x+y)


# ============================================================
# Генерация начальных метрик (реалистичные распределения)
# ============================================================

def generate_initial_metrics() -> Dict[str, float]:
    LT  = clip(sample_lognormal(INIT_MEDIANS["LT"], 1.9), *BOUNDS["LT"])
    AC  = int(clip(sample_lognormal(INIT_MEDIANS["AC"], 2.2), *BOUNDS["AC"]))
    CPC = clip(sample_lognormal(INIT_MEDIANS["CPC"], 1.8), *BOUNDS["CPC"])
    LCR = clip(sample_beta(INIT_MEDIANS["LCR"], BETA_KAPPA), *BOUNDS["LCR"])
    PCR = clip(sample_beta(INIT_MEDIANS["PCR"], BETA_KAPPA), *BOUNDS["PCR"])
    CAC = CPC / max(1e-9, (LCR * PCR))
    margin = LT - CAC
    return {
        "LT": round(LT, 2),
        "AC": AC,
        "CPC": round(CPC, 3),
        "LCR": round(LCR, 4),
        "PCR": round(PCR, 4),
        "CAC": round(CAC, 2),
        "MARGIN": round(margin, 2),
    }


# ============================================================
# Парсинг PSPLib (минимально-надёжный)
# ============================================================

def parse_jobs_count(content: str) -> int:
    m = re.search(r'jobs.*?:\s*(\d+)', content, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'number\s+of\s+activities.*?:\s*(\d+)', content, flags=re.I)
    if m2:
        return int(m2.group(1))
    # дефолт
    return 32

def parse_num_renewable_resources(content: str) -> int:
    m = re.search(r'renewable\s*resources.*?:\s*(\d+)', content, flags=re.I)
    if m:
        return int(m.group(1))
    m2 = re.search(r'-\s*renewable\s*:?\s*(\d+)', content, flags=re.I)
    if m2:
        return int(m2.group(1))
    return 4


def extract_requests_durations_block(content: str) -> List[str]:
    """
    Возвращает строки табличного блока REQUESTS/DURATIONS (если найден).
    """
    start = re.search(r'REQUESTS/DURATIONS', content, flags=re.I)
    if not start:
        return []
    # берём до следующего заголовка или конца файла
    tail = content[start.end():]
    stop = re.search(r'(PRECEDENCE|RESOURCEAVAILABILITIES|PROJECT|NR\s*=\s*|^\*{3,})', tail, flags=re.I | re.M)
    block = tail[:stop.start()] if stop else tail
    # отфильтруем строки с числами
    lines = [ln.strip() for ln in block.splitlines() if re.search(r'\d', ln)]
    return lines

def parse_requests_durations(content: str):
    durations = {}
    reqs = {}
    R = parse_num_renewable_resources(content)

    lines = extract_requests_durations_block(content)
    if not lines:
        return durations, reqs

    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue

        # пропускаем заголовки/линии-разделители
        low = ln.lower()
        if "jobnr" in low or "mode" in low or "duration" in low:
            continue
        if set(ln) <= set("- "):
            continue

        ints = re.findall(r'-?\d+', ln)
        if not ints:
            continue

        # Формат PSPLIB обычно: job, mode, duration, R1..Rk
        if len(ints) >= 3 + R:
            j = int(ints[0])
            mode = int(ints[1])  # можно не использовать, но полезно для контроля
            d = int(ints[2])
            res = list(map(int, ints[3:3 + R]))

        # Иногда встречается укороченный формат: job, duration, R1..Rk
        elif len(ints) >= 2 + R:
            j = int(ints[0])
            d = int(ints[1])
            res = list(map(int, ints[2:2 + R]))
        else:
            continue

        durations[j] = d
        reqs[j] = {i + 1: res[i] for i in range(R)}

    return durations, reqs

# ============================================================
# Стоимости ресурсов и задач
# ============================================================

def generate_resource_costs(num_resources: int) -> Dict[int, float]:
    costs = {}
    for i in range(1, num_resources + 1):
        if i in RESOURCE_COST_RANGES:
            lo, hi = RESOURCE_COST_RANGES[i]
        else:
            # для «неожиданно» больших R — разумный дефолт
            lo, hi = 40.0, 120.0
        costs[i] = round(random.uniform(lo, hi), 2)
    return costs

def compute_task_cost(duration: int, usage: Dict[int, int], resource_costs: Dict[int, float]) -> float:
    total = 0.0
    if duration is None or duration <= 0:
        return 0.0
    for r, amount in (usage or {}).items():
        total += float(amount) * resource_costs.get(r, 0.0) * float(duration)
    return round(total, 2)


# ============================================================
# Эффекты задач (запись в файл в %), масштабирование, насыщение
# ============================================================

def choose_job_type() -> str:
    r = random.random()
    s = 0.0
    for jt, p in JOB_TYPES_PROBS.items():
        s += p
        if r <= s:
            return jt
    return "internal"

def duration_scale(job: int, durations: Dict[int, int]) -> float:
    if job not in durations or not durations:
        return 1.0
    d = durations[job]
    med = sorted(durations.values())[len(durations)//2]
    if med <= 0:
        return 1.0
    return (d / med) ** DUR_POW

def draw_effect_pct(var: str, jt: str, dur_scale: float) -> float:
    mean_mult, std = EFFECTS[jt][var]
    noise = random.gauss(0.0, std)  # шум «вокруг» (mean_mult-1.0)
    factor = (mean_mult - 1.0) + noise
    return round(100.0 * factor * dur_scale, 2)

def diminishing_AC_factor(ac_now: int, K: float) -> float:
    # чем больше AC относительно K, тем меньше чистый прирост
    return 1.0 / (1.0 + float(ac_now) / max(1.0, K))


# ============================================================
# Главная функция генерации
# ============================================================

def generate_psplib_with_metrics(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    n_jobs = parse_jobs_count(content)
    durations, reqs = parse_requests_durations(content)
    R = parse_num_renewable_resources(content)

    # 1) Инициал метрики
    M0 = generate_initial_metrics()
    K_market = max(1_000.0, float(M0["AC"]) * MARKET_K_MULTIPLIER)

    service_block = (
        "\nSERVICE METRICS (INITIAL):\n"
        f"LT_0    = {M0['LT']}\n"
        f"AC_0    = {M0['AC']}\n"
        f"CPC_0   = {M0['CPC']}\n"
        f"LCR_0   = {M0['LCR']}\n"
        f"PCR_0   = {M0['PCR']}\n"
        f"CAC_0   = {M0['CAC']}\n"
        f"MARGIN0 = {M0['MARGIN']}\n"
    )

    # 2) Стоимости ресурсов
    resource_costs = generate_resource_costs(R)
    res_costs_block = "\nRESOURCE COSTS (per unit per time):\n"
    for i in range(1, R+1):
        res_costs_block += f"R{i} = {resource_costs[i]:.2f}\n"

    # 3) Стоимость задач
    task_costs_block = "\nTASK COSTS (total):\n# jobnr : total_cost\n"
    # supersource (1) и supersink (n_jobs) — по нулям
    task_costs: Dict[int, float] = {}
    for j in range(1, n_jobs + 1):
        dur = durations.get(j, 0)
        usage = reqs.get(j, {})
        cost = compute_task_cost(dur, usage, resource_costs) if (j not in (1, n_jobs)) else 0.0
        task_costs[j] = cost
        task_costs_block += f"{j} : {cost:.2f}\n"

    # 4) Эффекты задач (в процентах, для чтения человеком/парсером)
    metric_changes_block = "\nMETRIC CHANGES (per job, multiplicative % effects):\n"
    metric_changes_block += "# jobnr : type        AC%    LT%    CPC%    LCR%    PCR%\n"
    metric_changes_block += "1 : supersource   0 0 0 0 0\n"

    # эффект генерируем для 2..n_jobs-1
    for j in range(2, n_jobs):
        jt = choose_job_type()
        dsc = duration_scale(j, durations)

        # рисуем «сырые» проценты (с учётом длительности),
        # насыщение AC фактически будет применяться уже в моделировании,
        # а здесь мы просто логируем базовый %-эффект
        acp  = draw_effect_pct("AC",  jt, dsc)
        ltp  = draw_effect_pct("LT",  jt, dsc)
        cpcp = draw_effect_pct("CPC", jt, dsc)
        lcrp = draw_effect_pct("LCR", jt, dsc)
        pcrp = draw_effect_pct("PCR", jt, dsc)

        metric_changes_block += f"{j} : {jt:<11} {acp:+.2f} {ltp:+.2f} {cpcp:+.2f} {lcrp:+.2f} {pcrp:+.2f}\n"

    metric_changes_block += f"{n_jobs} : supersink     0 0 0 0 0\n"

    # Финальная сборка
    new_content = (
        content.rstrip()
        + "\n"
        + service_block
        + res_costs_block
        + task_costs_block
        + metric_changes_block
        + "\nEND OF FILE\n"
        + ("*" * 72) + "\n"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[ok] Файл с метриками и стоимостями создан: {output_file}")


# ============================================================
# Пример запуска
# ============================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import random

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/extended")
    ap.add_argument("--pattern", default="*.sm")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    inp = Path(args.in_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # берём только файлы
    files = [p for p in sorted(inp.rglob(args.pattern)) if p.is_file()]
    print(f"[info] found {len(files)} files in {inp}")

    ok = 0
    fail = 0

    for p in files:
        # сохраняем структуру подпапок: raw/j30.sm/j301_1.sm -> extended/j30.sm/j301_1_with_metrics.sm
        rel = p.relative_to(inp)
        out_file = (out / rel).with_suffix("")  # убираем .sm
        out_file = out_file.with_name(out_file.name + "_with_metrics.sm")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            generate_psplib_with_metrics(str(p), str(out_file))
            ok += 1
        except Exception as e:
            print(f"[skip] {p}: {type(e).__name__}: {e}")
            fail += 1

    print(f"[done] ok={ok} fail={fail} out={out}")



# === src\rcpsp_marketing\cli\main_demo.py ===
"""
Пример: один инстанс → SA → SGS → визуализации.

Сейчас это только заглушка CLI.
"""

def main() -> None:
    raise NotImplementedError("main_demo.main() ещё не реализована")


if __name__ == "__main__":
    main()













# === src\rcpsp_marketing\cli\run_compare.py ===
"""Обёртка над `rcpsp_marketing.experiments.compare_sa_rls` для CLI."""


def main() -> None:
    raise NotImplementedError("run_compare.main() ещё не реализована")


if __name__ == "__main__":
    main()













# === src\rcpsp_marketing\cli\run_sa_batch.py ===
"""Обёртка над `rcpsp_marketing.experiments.sa_batch` для запуска из CLI."""

def main() -> None:
    raise NotImplementedError("run_sa_batch.main() ещё не реализована")


if __name__ == "__main__":
    main()




# === src\rcpsp_marketing\config\__init__.py ===
"""
Конфигурационные параметры алгоритмов (SA, RLS и т.п.).
См. `defaults.py` для стандартных значений.
"""













# === src\rcpsp_marketing\config\defaults.py ===
"""
Заглушка для дефолтных параметров алгоритмов SA/RLS и т.п.
Заполните реальными значениями по мере необходимости.
"""

SA_DEFAULTS = {
    "initial_temperature": 1.0,
    "cooling_rate": 0.995,
    "iterations": 10_000,
}

RLS_DEFAULTS = {
    "iterations": 10_000,
}














# === src\rcpsp_marketing\core\__init__.py ===
"""
Ядро планировщика:
- граф предшествования,
- функции цели,
- построение расписаний.
"""













# === src\rcpsp_marketing\core\improvement.py ===
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


# === src\rcpsp_marketing\core\objective.py ===
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


# === src\rcpsp_marketing\core\precedence.py ===
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

# === src\rcpsp_marketing\core\scheduling.py ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

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


# === src\rcpsp_marketing\core\scheduling_incremental.py ===
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


# === src\rcpsp_marketing\core\scheduling_parallel_incremental.py ===
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


# === src\rcpsp_marketing\data\__init__.py ===
"""
Модели данных для RCPSP-маркетинга.
См. `models.py`.
"""













# === src\rcpsp_marketing\data\models.py ===
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


# === src\rcpsp_marketing\experiments\__init__.py ===
"""Эксперименты: батчи SA/RLS, сравнения и утилиты."""













# === src\rcpsp_marketing\experiments\bench_incremental_serial_sgs.py ===
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, is_topological_order
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)
from rcpsp_marketing.core.scheduling import serial_sgs_selective
from rcpsp_marketing.core.scheduling_incremental import SerialSSGSIncremental


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _insert(order: List[int], i: int, k: int) -> None:
    if i == k:
        return
    x = order.pop(i)
    if k > i:
        k -= 1
    order.insert(k, x)


def _pick_neighbor(
    proj: Any,
    rnd: random.Random,
    cur_order: List[int],
    *,
    movable_idx: List[int],
    neighbor: str,
    tries: int,
    insert_max_shift: Optional[int],
) -> Optional[Tuple[List[int], int]]:
    """
    Возвращает (cand_order, changed_from_idx) или None, если не нашли валидного.
    changed_from_idx = min(i,k) — корректная граница пересчёта префикса для swap/insert.
    """
    n = len(cur_order)

    for _ in range(tries):
        cand = list(cur_order)

        if neighbor == "swap":
            i, k = rnd.sample(movable_idx, 2)
            _swap(cand, i, k)
            changed = min(i, k)

        elif neighbor == "insert":
            i = rnd.choice(movable_idx)
            if insert_max_shift is None:
                k = rnd.choice([x for x in movable_idx if x != i])
            else:
                lo = max(0, i - insert_max_shift)
                hi = min(n - 1, i + insert_max_shift)
                window = [x for x in movable_idx if x != i and lo <= x <= hi]
                k = rnd.choice(window) if window else rnd.choice([x for x in movable_idx if x != i])

            if i == k:
                continue
            _insert(cand, i, k)
            changed = min(i, k)

        else:
            raise ValueError("neighbor must be 'swap' or 'insert'")

        if is_topological_order(proj, cand):
            return cand, changed

    return None


# =========================
# Experiment
# =========================

@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    fn: Callable[..., Any]


def run_experiment(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],

    # neighbor generation
    neighbor: str,                 # "swap" | "insert"
    steps: int,                    # how many neighbor evaluations per (instance, seed)
    tries_per_step: int,           # attempts to find topo-valid neighbor
    insert_max_shift: Optional[int],

    # incremental
    checkpoint_every: int,         # e.g. 10

    # objective selection
    objective: str,                # "ref" | "fast"
    include_dummy_costs: bool,

    # correctness
    abs_tol: float,

    # simulate real search (commit accepted moves)
    commit_prob: float,            # 0.0 => never commit (pure eval benchmark), e.g. 0.15 to simulate SA/HC
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PSPLibExtendedParser()

    # pick objective
    if objective == "ref":
        obj_spec = ObjectiveSpec("ref", evaluate_profit_over_horizon)
    elif objective == "fast":
        obj_spec = ObjectiveSpec("fast", evaluate_profit_over_horizon_fast)
    else:
        raise ValueError("objective must be 'ref' or 'fast'")

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)
        print(f"[instance] {cls} / {inst}")

        for s in seeds:
            rnd = random.Random(s)

            cur_order = random_topo_sort_fixed_ends(proj, seed=s)
            movable_idx = _make_movable_indices(proj, cur_order)

            inc = SerialSSGSIncremental(proj, T=T, include_dummies=True, checkpoint_every=checkpoint_every)
            inc.set_current(cur_order)

            ok_cnt = 0
            bad_cnt = 0
            no_nb = 0

            # separate timing buckets
            full_decode_time = 0.0
            full_obj_time = 0.0
            incr_decode_time = 0.0
            incr_obj_time = 0.0

            evals = 0

            for _ in range(steps):
                got = _pick_neighbor(
                    proj, rnd, cur_order,
                    movable_idx=movable_idx,
                    neighbor=neighbor,
                    tries=tries_per_step,
                    insert_max_shift=insert_max_shift,
                )
                if got is None:
                    no_nb += 1
                    continue

                cand_order, changed_from = got

                # --- FULL: decode + objective (split) ---
                t0 = perf_counter()
                full_res = serial_sgs_selective(proj, cand_order, T=T, include_dummies=True)
                t1 = perf_counter()
                full_obj = obj_spec.fn(
                    proj,
                    full_res.schedule,
                    selected_jobs=full_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t2 = perf_counter()

                full_decode_time += (t1 - t0)
                full_obj_time += (t2 - t1)

                # --- INCR: resume + objective (split) ---
                t3 = perf_counter()
                inc_res = inc.decode_neighbor(cand_order, changed_from_order_idx=changed_from)
                t4 = perf_counter()
                inc_obj = obj_spec.fn(
                    proj,
                    inc_res.schedule,
                    selected_jobs=inc_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t5 = perf_counter()

                incr_decode_time += (t4 - t3)
                incr_obj_time += (t5 - t4)

                evals += 1

                # correctness
                same_sched = (full_res.schedule.start == inc_res.schedule.start) and (full_res.schedule.finish == inc_res.schedule.finish)
                same_lists = (full_res.selected == inc_res.selected) and (full_res.skipped == inc_res.skipped)
                dv = float(inc_obj.value) - float(full_obj.value)
                ok = same_sched and same_lists and (abs(dv) <= abs_tol)

                if ok:
                    ok_cnt += 1
                else:
                    bad_cnt += 1

                # simulate “accepted” moves (optional)
                if commit_prob > 0.0 and rnd.random() < commit_prob:
                    cur_order = cand_order
                    movable_idx = _make_movable_indices(proj, cur_order)
                    inc.set_current(cur_order)

            full_time = full_decode_time + full_obj_time
            incr_time = incr_decode_time + incr_obj_time

            full_ms = (full_time / max(1, evals)) * 1000.0
            incr_ms = (incr_time / max(1, evals)) * 1000.0
            speedup = (full_ms / incr_ms) if incr_ms > 0 else float("inf")

            full_decode_ms = (full_decode_time / max(1, evals)) * 1000.0
            full_obj_ms = (full_obj_time / max(1, evals)) * 1000.0
            incr_decode_ms = (incr_decode_time / max(1, evals)) * 1000.0
            incr_obj_ms = (incr_obj_time / max(1, evals)) * 1000.0

            rows.append({
                "class": cls,
                "instance": inst,
                "instance_path": str(fp),
                "T": int(T),
                "seed": int(s),

                "neighbor": neighbor,
                "steps": int(steps),
                "tries_per_step": int(tries_per_step),
                "insert_max_shift": (None if insert_max_shift is None else int(insert_max_shift)),

                "checkpoint_every": int(checkpoint_every),

                "objective": obj_spec.name,
                "include_dummy_costs": bool(include_dummy_costs),
                "commit_prob": float(commit_prob),

                "evals": int(evals),
                "no_neighbor": int(no_nb),
                "ok": int(ok_cnt),
                "bad": int(bad_cnt),

                # end-to-end
                "full_ms_per_eval": float(full_ms),
                "incr_ms_per_eval": float(incr_ms),
                "speedup": float(speedup),

                # breakdown
                "full_decode_ms_per_eval": float(full_decode_ms),
                "full_obj_ms_per_eval": float(full_obj_ms),
                "incr_decode_ms_per_eval": float(incr_decode_ms),
                "incr_obj_ms_per_eval": float(incr_obj_ms),
            })

    df = pd.DataFrame(rows)

    # --- save runs
    runs_path = out_dir / "decode_incremental_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # --- summary (simple)
    summary = (
        df.groupby(["class", "neighbor", "checkpoint_every", "objective"], as_index=False)
          .agg(
              runs=("seed", "count"),
              evals=("evals", "sum"),
              ok=("ok", "sum"),
              bad=("bad", "sum"),
              mean_full_ms=("full_ms_per_eval", "mean"),
              mean_incr_ms=("incr_ms_per_eval", "mean"),
              mean_speedup=("speedup", "mean"),
          )
          .sort_values(["class", "neighbor", "checkpoint_every", "objective"])
    )

    sum_path = out_dir / "decode_incremental_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # --- weighted overall report (like your style) ---
    if len(df) > 0:
        df["full_total_ms"] = df["evals"] * df["full_ms_per_eval"]
        df["incr_total_ms"] = df["evals"] * df["incr_ms_per_eval"]

        df["full_decode_total_ms"] = df["evals"] * df["full_decode_ms_per_eval"]
        df["full_obj_total_ms"] = df["evals"] * df["full_obj_ms_per_eval"]
        df["incr_decode_total_ms"] = df["evals"] * df["incr_decode_ms_per_eval"]
        df["incr_obj_total_ms"] = df["evals"] * df["incr_obj_ms_per_eval"]

        total_evals = int(df["evals"].sum())

        full_total = float(df["full_total_ms"].sum())
        incr_total = float(df["incr_total_ms"].sum())
        speedup_total = (full_total / incr_total) if incr_total > 0 else float("inf")

        full_decode_avg = float(df["full_decode_total_ms"].sum()) / max(1, total_evals)
        full_obj_avg = float(df["full_obj_total_ms"].sum()) / max(1, total_evals)
        incr_decode_avg = float(df["incr_decode_total_ms"].sum()) / max(1, total_evals)
        incr_obj_avg = float(df["incr_obj_total_ms"].sum()) / max(1, total_evals)

        decode_speedup = (full_decode_avg / incr_decode_avg) if incr_decode_avg > 0 else float("inf")
        obj_speedup = (full_obj_avg / incr_obj_avg) if incr_obj_avg > 0 else float("inf")

        full_share_decode = full_decode_avg / max(1e-12, (full_decode_avg + full_obj_avg))
        incr_share_decode = incr_decode_avg / max(1e-12, (incr_decode_avg + incr_obj_avg))

        ok_total = int(df["ok"].sum())
        bad_total = int(df["bad"].sum())
        ok_rate = ok_total / max(1, (ok_total + bad_total))

        full_total_avg = full_total / max(1, total_evals)
        incr_total_avg = incr_total / max(1, total_evals)

    else:
        total_evals = 0
        full_total = incr_total = 0.0
        speedup_total = 0.0
        full_decode_avg = full_obj_avg = incr_decode_avg = incr_obj_avg = 0.0
        decode_speedup = obj_speedup = 0.0
        full_share_decode = incr_share_decode = 0.0
        ok_total = bad_total = 0
        ok_rate = 0.0
        full_total_avg = incr_total_avg = 0.0

    rep_lines: List[str] = []
    rep_lines.append("End-to-end full vs incremental decode benchmark\n")
    rep_lines.append(f"- runs: {len(df)}")
    rep_lines.append(f"- ok(exact schedule+lists+value): {ok_total} ({ok_rate*100:.1f}%)")
    rep_lines.append(f"- bad: {bad_total}")
    rep_lines.append(f"- evals(total): {total_evals}")
    rep_lines.append("")

    rep_lines.append("Mean speedups (overall, weighted by evals):")
    rep_lines.append(
        f"- total={speedup_total:.3f}x, decode={decode_speedup:.3f}x, objective={obj_speedup:.3f}x, "
        f"full_share_decode={full_share_decode*100:.1f}%, incr_share_decode={incr_share_decode*100:.1f}%"
    )
    rep_lines.append("")
    rep_lines.append("Timing (ms per eval, weighted):")
    rep_lines.append(f"- full: total={full_total_avg:.4f}, decode={full_decode_avg:.4f}, obj={full_obj_avg:.4f}")
    rep_lines.append(f"- incr: total={incr_total_avg:.4f}, decode={incr_decode_avg:.4f}, obj={incr_obj_avg:.4f}")
    rep_lines.append("")
    rep_lines.append("Files:")
    rep_lines.append(f"- {runs_path.name}")
    rep_lines.append(f"- {sum_path.name}")

    rep_path = out_dir / "decode_incremental_report.txt"
    rep_path.write_text("\n".join(rep_lines), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    MODE = "dir"  # "single" | "dir"

    # single
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # dir
    IN_DIR = Path(r"data/extended/j30.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 30  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/decode_incremental_compare")

    # horizon and seeds
    T = 50
    SEEDS = list(range(1, 11))

    # neighbor benchmark config
    NEIGHBOR = "swap"          # "swap" | "insert"
    STEPS = 2000               # neighbor evals per (instance, seed)
    TRIES_PER_STEP = 30        # try to find topo-valid neighbor
    INSERT_MAX_SHIFT = 10      # None = global insert

    # incremental config
    CHECKPOINT_EVERY = 10      # 5..20 обычно нормально

    # objective
    OBJECTIVE = "fast"         # "ref" | "fast"
    INCLUDE_DUMMY_COSTS = False

    # correctness
    ABS_TOL = 1e-9

    # simulate real search acceptance (0.0 = pure eval benchmark)
    COMMIT_PROB = 0.15

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(
        f"[info] mode={MODE} instances={len(files)} neighbor={NEIGHBOR} steps={STEPS} "
        f"T={T} objective={OBJECTIVE} checkpoint_every={CHECKPOINT_EVERY} commit_prob={COMMIT_PROB}"
    )

    run_experiment(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,

        neighbor=NEIGHBOR,
        steps=STEPS,
        tries_per_step=TRIES_PER_STEP,
        insert_max_shift=INSERT_MAX_SHIFT,

        checkpoint_every=CHECKPOINT_EVERY,

        objective=OBJECTIVE,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,

        abs_tol=ABS_TOL,
        commit_prob=COMMIT_PROB,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")


# === src\rcpsp_marketing\experiments\common.py ===
"""Общие утилиты для экспериментов (загрузка проекта, стартовые решения, CSV-лог)."""

from pathlib import Path
from typing import Any


def load_project(instance_path: Path) -> Any:
    raise NotImplementedError("load_project() ещё не реализована")


def make_initial_chain(project: Any) -> Any:
    raise NotImplementedError("make_initial_chain() ещё не реализована")














# === src\rcpsp_marketing\experiments\compare_end2end_ref_vs_fast.py ===
# tools/compare_end2end_ref_vs_fast.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends

# objectives
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)

# decoders
from rcpsp_marketing.core.scheduling import (
    serial_sgs_selective,
    parallel_sgs_selective,
    parallel_sgs_selective_greedy,
)

# searchers (must be your UPDATED versions that accept objective/objective_fn)
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing
from rcpsp_marketing.algorithms.local_search.hill_climb import hill_climb
# if you updated your RLS to accept objective/objective_fn; otherwise set RUN_RLS=False
from rcpsp_marketing.algorithms.local_search.randomized_local_search import randomized_local_search


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def make_timed_decode(decode_fn: Callable[[Any, List[int], int], Any]) -> Tuple[Callable[[Any, List[int], int], Any], Dict[str, float]]:
    stats = {"calls": 0.0, "sec": 0.0}

    def wrapped(proj: Any, order: List[int], T: int) -> Any:
        t0 = perf_counter()
        out = decode_fn(proj, order, T)
        stats["sec"] += perf_counter() - t0
        stats["calls"] += 1.0
        return out

    return wrapped, stats


def make_timed_objective(obj_fn: Callable[..., Any]) -> Tuple[Callable[..., Any], Dict[str, float]]:
    stats = {"calls": 0.0, "sec": 0.0}

    def wrapped(proj: Any, schedule: Any, *, selected_jobs=None, T=None, include_dummy_costs: bool = False) -> Any:
        t0 = perf_counter()
        out = obj_fn(proj, schedule, selected_jobs=selected_jobs, T=T, include_dummy_costs=include_dummy_costs)
        stats["sec"] += perf_counter() - t0
        stats["calls"] += 1.0
        return out

    return wrapped, stats


# =========================
# Decoder registry
# =========================

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    decode_fn: Callable[[Any, List[int], int], Any]


def make_decoders(*, greedy_min_score: float, greedy_unlock_weight: float) -> Dict[str, DecoderSpec]:
    def decode_ssgs(proj: Any, order: List[int], T: int):
        return serial_sgs_selective(proj, order, T=T, include_dummies=True)

    def decode_psgs(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective(proj, order, T=T, include_dummies=True, include_sink=False)

    def decode_psgs_greedy(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_greedy(
            proj,
            order,
            T=T,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return {
        "SSGS": DecoderSpec("SSGS", decode_ssgs),
        "PSGS": DecoderSpec("PSGS", decode_psgs),
        "PSGS_greedy": DecoderSpec("PSGS_greedy", decode_psgs_greedy),
    }


# =========================
# End-to-end benchmark
# =========================

def run_one_searcher(
    *,
    searcher: str,  # "SA" | "HC" | "RLS"
    proj: Any,
    T: int,
    start_order: List[int],
    decode_fn: Callable[[Any, List[int], int], Any],
    seed_algo: int,

    # objective impl
    obj_impl: str,  # "ref" | "fast"
    include_dummy_costs: bool,

    # budgets/params
    max_profit_evals: Optional[int],
    sa_iters: int,
    sa_neighbor: str,
    sa_tries: int,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,

    hc_iters: int,
    hc_neighbor: str,
    hc_tries: int,

    rls_iters: int,
    rls_tries: int,
) -> Dict[str, Any]:
    if obj_impl == "ref":
        base_obj_fn = evaluate_profit_over_horizon
    elif obj_impl == "fast":
        base_obj_fn = evaluate_profit_over_horizon_fast
    else:
        raise ValueError("obj_impl must be 'ref' or 'fast'")

    timed_decode, dec_stats = make_timed_decode(decode_fn)
    timed_obj, obj_stats = make_timed_objective(base_obj_fn)

    t0 = perf_counter()

    if searcher == "SA":
        res = simulated_annealing(
            proj,
            T=T,
            start_order=start_order,
            decode_fn=timed_decode,
            seed=seed_algo,
            iters=sa_iters,
            neighbor=sa_neighbor,
            tries_per_iter=sa_tries,
            T0=sa_T0,
            alpha=sa_alpha,
            Tmin=sa_Tmin,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key: custom objective
            include_dummy_costs=include_dummy_costs,
            keep_history=False,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = "budget" if res.stopped_by_budget else "iters_or_temp"

    elif searcher == "HC":
        res = hill_climb(
            proj,
            T=T,
            start_order=start_order,
            decode_fn=timed_decode,
            seed=seed_algo,
            iters=hc_iters,
            neighbor=hc_neighbor,
            tries_per_iter=hc_tries,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key
            include_dummy_costs=include_dummy_costs,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = str(res.stopped_reason)

    elif searcher == "RLS":
        res = randomized_local_search(
            proj,
            T=T,
            start_order=start_order,
            seed=seed_algo,
            iters=rls_iters,
            tries_per_iter=rls_tries,
            max_profit_evals=max_profit_evals,
            objective_fn=timed_obj,              # <--- key (your updated RLS)
            include_dummy_costs=include_dummy_costs,
        )
        best_value = float(res.best_value)
        profit_evals = int(res.profit_evals)
        accepted = int(res.accepted)
        stopped = str(res.stopped_reason)

    else:
        raise ValueError("searcher must be 'SA' or 'HC' or 'RLS'")

    total_sec = perf_counter() - t0

    decode_sec = float(dec_stats["sec"])
    objective_sec = float(obj_stats["sec"])
    overhead_sec = max(0.0, total_sec - decode_sec - objective_sec)

    decode_calls = int(dec_stats["calls"])
    objective_calls = int(obj_stats["calls"])

    return {
        "best_value": best_value,
        "profit_evals": profit_evals,
        "accepted": accepted,
        "stopped_reason": stopped,

        "total_sec": total_sec,
        "decode_sec": decode_sec,
        "objective_sec": objective_sec,
        "overhead_sec": overhead_sec,

        "decode_calls": decode_calls,
        "objective_calls": objective_calls,
    }


def run_end2end(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],
    decoder_name: str,
    searchers: List[str],
    include_dummy_costs: bool,
    max_profit_evals: Optional[int],

    # params
    sa_iters: int,
    sa_neighbor: str,
    sa_tries: int,
    sa_T0: float,
    sa_alpha: float,
    sa_Tmin: float,

    hc_iters: int,
    hc_neighbor: str,
    hc_tries: int,

    rls_iters: int,
    rls_tries: int,

    greedy_min_score: float,
    greedy_unlock_weight: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PSPLibExtendedParser()
    decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)
    if decoder_name not in decoders:
        raise ValueError(f"Unknown decoder '{decoder_name}', choose from {list(decoders.keys())}")
    decoder = decoders[decoder_name]

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)

        print(f"[instance] {cls} / {inst}")

        # paired start orders
        start_orders = {s: random_topo_sort_fixed_ends(proj, seed=s) for s in seeds}

        for sea in searchers:
            for s in seeds:
                start_order = start_orders[s]
                seed_algo = 10_000 + s  # keep deterministic

                # RUN ref
                ref = run_one_searcher(
                    searcher=sea,
                    proj=proj,
                    T=T,
                    start_order=start_order,
                    decode_fn=decoder.decode_fn,
                    seed_algo=seed_algo,
                    obj_impl="ref",
                    include_dummy_costs=include_dummy_costs,
                    max_profit_evals=max_profit_evals,
                    sa_iters=sa_iters, sa_neighbor=sa_neighbor, sa_tries=sa_tries, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
                    hc_iters=hc_iters, hc_neighbor=hc_neighbor, hc_tries=hc_tries,
                    rls_iters=rls_iters, rls_tries=rls_tries,
                )

                # RUN fast
                fast = run_one_searcher(
                    searcher=sea,
                    proj=proj,
                    T=T,
                    start_order=start_order,
                    decode_fn=decoder.decode_fn,
                    seed_algo=seed_algo,
                    obj_impl="fast",
                    include_dummy_costs=include_dummy_costs,
                    max_profit_evals=max_profit_evals,
                    sa_iters=sa_iters, sa_neighbor=sa_neighbor, sa_tries=sa_tries, sa_T0=sa_T0, sa_alpha=sa_alpha, sa_Tmin=sa_Tmin,
                    hc_iters=hc_iters, hc_neighbor=hc_neighbor, hc_tries=hc_tries,
                    rls_iters=rls_iters, rls_tries=rls_tries,
                )

                # compare best_value equality (should match if objective impl is equivalent)
                dv = fast["best_value"] - ref["best_value"]
                ok_value = (dv == 0.0)

                # speedups
                speedup_total = (ref["total_sec"] / fast["total_sec"]) if fast["total_sec"] > 0 else float("inf")
                speedup_obj = (ref["objective_sec"] / fast["objective_sec"]) if fast["objective_sec"] > 0 else float("inf")

                rows.append({
                    "class": cls,
                    "instance": inst,
                    "instance_path": str(fp),
                    "decoder": decoder.name,
                    "searcher": sea,
                    "T": int(T),
                    "seed": int(s),
                    "seed_algo": int(seed_algo),
                    "include_dummy_costs": bool(include_dummy_costs),
                    "max_profit_evals": max_profit_evals if max_profit_evals is not None else "",

                    "ref_best_value": ref["best_value"],
                    "fast_best_value": fast["best_value"],
                    "diff_best_value": dv,
                    "ok_value_exact": bool(ok_value),

                    "ref_total_sec": ref["total_sec"],
                    "fast_total_sec": fast["total_sec"],
                    "speedup_total": speedup_total,

                    "ref_decode_sec": ref["decode_sec"],
                    "fast_decode_sec": fast["decode_sec"],
                    "ref_objective_sec": ref["objective_sec"],
                    "fast_objective_sec": fast["objective_sec"],
                    "speedup_objective": speedup_obj,

                    "ref_overhead_sec": ref["overhead_sec"],
                    "fast_overhead_sec": fast["overhead_sec"],

                    "ref_profit_evals": ref["profit_evals"],
                    "fast_profit_evals": fast["profit_evals"],
                    "ref_decode_calls": ref["decode_calls"],
                    "fast_decode_calls": fast["decode_calls"],
                    "ref_objective_calls": ref["objective_calls"],
                    "fast_objective_calls": fast["objective_calls"],

                    "ref_accepted": ref["accepted"],
                    "fast_accepted": fast["accepted"],
                    "ref_stopped_reason": ref["stopped_reason"],
                    "fast_stopped_reason": fast["stopped_reason"],

                    # shares (how much of total time is spent where)
                    "ref_share_objective": ref["objective_sec"] / ref["total_sec"] if ref["total_sec"] > 0 else 0.0,
                    "fast_share_objective": fast["objective_sec"] / fast["total_sec"] if fast["total_sec"] > 0 else 0.0,
                })

    df = pd.DataFrame(rows)
    runs_path = out_dir / "end2end_ref_vs_fast_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # summary
    def _std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    summary = (
        df.groupby(["class", "instance", "decoder", "searcher"], as_index=False)
          .agg(
              runs=("seed", "count"),
              ok_rate=("ok_value_exact", "mean"),
              mean_speedup_total=("speedup_total", "mean"),
              std_speedup_total=("speedup_total", _std),
              mean_speedup_objective=("speedup_objective", "mean"),
              std_speedup_objective=("speedup_objective", _std),
              mean_ref_total_sec=("ref_total_sec", "mean"),
              mean_fast_total_sec=("fast_total_sec", "mean"),
              mean_ref_share_objective=("ref_share_objective", "mean"),
              mean_fast_share_objective=("fast_share_objective", "mean"),
          )
          .sort_values(["class", "instance", "decoder", "searcher"])
    )
    sum_path = out_dir / "end2end_ref_vs_fast_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # short report
    total = len(df)
    ok = int(df["ok_value_exact"].sum()) if total else 0
    rep = []
    rep.append("End-to-end ref vs fast objective benchmark\n")
    rep.append(f"- runs: {total}")
    rep.append(f"- ok(best_value exact match): {ok} ({(ok/total*100 if total else 0):.1f}%)")
    rep.append("")
    rep.append("Mean speedups (overall):")
    for (sea,), g in df.groupby(["searcher"]):
        rep.append(f"- {sea}: total={g['speedup_total'].mean():.3f}x, objective={g['speedup_objective'].mean():.3f}x, "
                   f"ref_share_obj={g['ref_share_objective'].mean()*100:.1f}%")
    rep_path = out_dir / "end2end_ref_vs_fast_report.txt"
    rep_path.write_text("\n".join(rep), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    # ===== CONFIG =====
    MODE = "single"  # "single" | "dir"

    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    IN_DIR = Path(r"data/extended/j60.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 10  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/end2end_objective_compare")

    T = 50
    SEEDS = list(range(1, 11))

    DECODER = "SSGS"  # "SSGS" | "PSGS" | "PSGS_greedy"
    SEARCHERS = ["SA", "HC"]     # add "RLS" if your updated RLS is available/stable
    RUN_RLS = False             # safety switch if your RLS import/name differs

    INCLUDE_DUMMY_COSTS = False

    # budget (keeps runtimes sane)
    MAX_PROFIT_EVALS: Optional[int] = 10_000

    # SA params
    SA_ITERS = 200_000
    SA_TRIES = 30
    SA_NEIGHBOR = "insert"
    SA_T0 = 2e5
    SA_ALPHA = 0.9998
    SA_TMIN = 1.0

    # HC params
    HC_ITERS = 200_000
    HC_TRIES = 50
    HC_NEIGHBOR = "insert"

    # RLS params
    RLS_ITERS = 200_000
    RLS_TRIES = 20

    # PSGS_greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0
    # ==================

    if RUN_RLS and "RLS" not in SEARCHERS:
        SEARCHERS = SEARCHERS + ["RLS"]

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)} decoder={DECODER} searchers={SEARCHERS} T={T}")

    run_end2end(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,
        decoder_name=DECODER,
        searchers=SEARCHERS,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,
        max_profit_evals=MAX_PROFIT_EVALS,

        sa_iters=SA_ITERS,
        sa_neighbor=SA_NEIGHBOR,
        sa_tries=SA_TRIES,
        sa_T0=SA_T0,
        sa_alpha=SA_ALPHA,
        sa_Tmin=SA_TMIN,

        hc_iters=HC_ITERS,
        hc_neighbor=HC_NEIGHBOR,
        hc_tries=HC_TRIES,

        rls_iters=RLS_ITERS,
        rls_tries=RLS_TRIES,

        greedy_min_score=GREEDY_MIN_SCORE,
        greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")


# === src\rcpsp_marketing\experiments\compare_objective_ref_vs_fast.py ===
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)
from rcpsp_marketing.core.scheduling import (
    serial_sgs_selective,
    parallel_sgs_selective,
    parallel_sgs_selective_greedy,
)


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def _time_objective(
    fn: Callable[..., Any],
    *,
    proj: Any,
    schedule: Any,
    selected: Any,
    T: int,
    include_dummy_costs: bool,
    repeats: int,
) -> Tuple[Any, float]:
    """
    Возвращает (obj, avg_ms_per_call) для fn на фиксированном (schedule, selected).
    """
    # warmup (JIT нет, но прогреваем кэш/страницы/ветвления)
    obj = fn(proj, schedule, selected_jobs=selected, T=T, include_dummy_costs=include_dummy_costs)

    t0 = perf_counter()
    for _ in range(repeats):
        obj = fn(proj, schedule, selected_jobs=selected, T=T, include_dummy_costs=include_dummy_costs)
    dt = perf_counter() - t0
    avg_ms = (dt / max(1, repeats)) * 1000.0
    return obj, avg_ms


# =========================
# Decoder registry
# =========================

@dataclass(frozen=True)
class DecoderSpec:
    name: str
    decode_fn: Callable[[Any, List[int], int], Any]


def make_decoders(*, greedy_min_score: float, greedy_unlock_weight: float) -> Dict[str, DecoderSpec]:
    def decode_ssgs(proj: Any, order: List[int], T: int):
        return serial_sgs_selective(proj, order, T=T, include_dummies=True)

    def decode_psgs(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective(proj, order, T=T, include_dummies=True, include_sink=False)

    def decode_psgs_greedy(proj: Any, order: List[int], T: int):
        return parallel_sgs_selective_greedy(
            proj,
            order,
            T=T,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    decs = {
        "SSGS": DecoderSpec("SSGS", decode_ssgs),
        "PSGS": DecoderSpec("PSGS", decode_psgs),
        "PSGS_greedy": DecoderSpec("PSGS_greedy", decode_psgs_greedy),
    }
    return decs


# =========================
# Experiment
# =========================

def run_experiment(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],
    decoder_name: str,
    repeats: int,
    include_dummy_costs: bool,
    greedy_min_score: float,
    greedy_unlock_weight: float,
    abs_tol: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    decoders = make_decoders(greedy_min_score=greedy_min_score, greedy_unlock_weight=greedy_unlock_weight)
    if decoder_name not in decoders:
        raise ValueError(f"Unknown decoder_name='{decoder_name}'. Choose one of: {list(decoders.keys())}")
    decoder = decoders[decoder_name]

    parser = PSPLibExtendedParser()

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)

        print(f"[instance] {cls} / {inst}")

        for s in seeds:
            order = random_topo_sort_fixed_ends(proj, seed=s)

            # Decode ONCE
            dec_t0 = perf_counter()
            res = decoder.decode_fn(proj, order, T)
            dec_dt = perf_counter() - dec_t0

            # Evaluate both on the SAME (schedule, selected)
            obj_ref, ms_ref = _time_objective(
                evaluate_profit_over_horizon,
                proj=proj,
                schedule=res.schedule,
                selected=res.selected,
                T=T,
                include_dummy_costs=include_dummy_costs,
                repeats=repeats,
            )
            obj_fast, ms_fast = _time_objective(
                evaluate_profit_over_horizon_fast,
                proj=proj,
                schedule=res.schedule,
                selected=res.selected,
                T=T,
                include_dummy_costs=include_dummy_costs,
                repeats=repeats,
            )

            v_ref = float(obj_ref.value)
            v_fast = float(obj_fast.value)
            r_ref = float(obj_ref.revenue)
            r_fast = float(obj_fast.revenue)
            c_ref = float(obj_ref.cost)
            c_fast = float(obj_fast.cost)

            dv = v_fast - v_ref
            dr = r_fast - r_ref
            dc = c_fast - c_ref

            rel = dv / (abs(v_ref) + 1e-12)

            ok = (abs(dv) <= abs_tol) and (abs(dr) <= abs_tol) and (abs(dc) <= abs_tol)

            speedup = (ms_ref / ms_fast) if ms_fast > 0 else float("inf")

            rows.append({
                "class": cls,
                "instance": inst,
                "instance_path": str(fp),
                "T": int(T),
                "seed": int(s),
                "decoder": decoder.name,

                "decode_time_ms": dec_dt * 1000.0,

                "ref_value": v_ref,
                "fast_value": v_fast,
                "diff_value": dv,
                "rel_diff_value": rel,

                "ref_revenue": r_ref,
                "fast_revenue": r_fast,
                "diff_revenue": dr,

                "ref_cost": c_ref,
                "fast_cost": c_fast,
                "diff_cost": dc,

                "ref_ms_per_call": ms_ref,
                "fast_ms_per_call": ms_fast,
                "speedup": speedup,

                "repeats": int(repeats),
                "include_dummy_costs": bool(include_dummy_costs),
                "abs_tol": float(abs_tol),
                "ok": bool(ok),
            })

    df = pd.DataFrame(rows)

    # --- save runs
    runs_path = out_dir / "objective_compare_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    # --- summary
    def _std(x):
        return float(x.std(ddof=1)) if len(x) > 1 else 0.0

    summary = (
        df.groupby(["class", "instance", "decoder"], as_index=False)
          .agg(
              runs=("seed", "count"),
              ok_rate=("ok", "mean"),
              max_abs_diff_value=("diff_value", lambda x: float(x.abs().max())),
              max_abs_diff_revenue=("diff_revenue", lambda x: float(x.abs().max())),
              max_abs_diff_cost=("diff_cost", lambda x: float(x.abs().max())),
              mean_ref_ms=("ref_ms_per_call", "mean"),
              mean_fast_ms=("fast_ms_per_call", "mean"),
              mean_speedup=("speedup", "mean"),
              std_speedup=("speedup", _std),
              mean_decode_ms=("decode_time_ms", "mean"),
          )
          .sort_values(["class", "instance", "decoder"])
    )

    sum_path = out_dir / "objective_compare_summary.csv"
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # --- short report
    total = len(df)
    ok_cnt = int(df["ok"].sum()) if total else 0
    ok_rate = (ok_cnt / total) if total else 0.0

    overall = {
        "total_runs": total,
        "ok_runs": ok_cnt,
        "ok_rate": ok_rate,
        "max_abs_diff_value": float(df["diff_value"].abs().max()) if total else 0.0,
        "max_abs_diff_revenue": float(df["diff_revenue"].abs().max()) if total else 0.0,
        "max_abs_diff_cost": float(df["diff_cost"].abs().max()) if total else 0.0,
        "mean_ref_ms": float(df["ref_ms_per_call"].mean()) if total else 0.0,
        "mean_fast_ms": float(df["fast_ms_per_call"].mean()) if total else 0.0,
        "mean_speedup": float(df["speedup"].mean()) if total else 0.0,
    }

    rep_lines = []
    rep_lines.append("Objective ref vs fast comparison\n")
    rep_lines.append(f"- runs: {overall['total_runs']}")
    rep_lines.append(f"- ok:   {overall['ok_runs']} ({overall['ok_rate']*100:.1f}%)  tol={abs_tol}")
    rep_lines.append("")
    rep_lines.append("Max absolute diffs:")
    rep_lines.append(f"- value:   {overall['max_abs_diff_value']:.6g}")
    rep_lines.append(f"- revenue: {overall['max_abs_diff_revenue']:.6g}")
    rep_lines.append(f"- cost:    {overall['max_abs_diff_cost']:.6g}")
    rep_lines.append("")
    rep_lines.append("Timing (ms per call, averaged):")
    rep_lines.append(f"- ref:  {overall['mean_ref_ms']:.4f} ms")
    rep_lines.append(f"- fast: {overall['mean_fast_ms']:.4f} ms")
    rep_lines.append(f"- speedup: {overall['mean_speedup']:.2f}x")
    rep_lines.append("")
    rep_lines.append("Files:")
    rep_lines.append(f"- {runs_path.name}")
    rep_lines.append(f"- {sum_path.name}")

    rep_path = out_dir / "objective_compare_report.txt"
    rep_path.write_text("\n".join(rep_lines), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    # ===== CONFIG =====
    MODE = "dir"  # "single" | "dir"

    # single
    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")

    # dir
    IN_DIR = Path(r"data/extended/j120.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 30  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/objective_compare")

    # horizon and seeds
    T = 50
    SEEDS = list(range(1, 11))

    # decoder used to generate schedules (objective is measured on those schedules)
    DECODER = "SSGS"  # "SSGS" | "PSGS" | "PSGS_greedy"

    # timing: objective will be called N times on same schedule (average per call)
    REPEATS = 200

    # objective options
    INCLUDE_DUMMY_COSTS = False

    # PSGS_greedy params
    GREEDY_MIN_SCORE = -1e18
    GREEDY_UNLOCK_WEIGHT = 0.0

    # accuracy check tolerance
    ABS_TOL = 1e-9
    # ==================

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(f"[info] mode={MODE} instances={len(files)} decoder={DECODER} repeats={REPEATS} T={T}")

    run_experiment(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,
        decoder_name=DECODER,
        repeats=REPEATS,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,
        greedy_min_score=GREEDY_MIN_SCORE,
        greedy_unlock_weight=GREEDY_UNLOCK_WEIGHT,
        abs_tol=ABS_TOL,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")

# === src\rcpsp_marketing\experiments\compare_sa_rls.py ===
"""Заглушка сравнения SA и RLS на наборе инстансов."""

from pathlib import Path
from typing import Any, Iterable


def compare_sa_rls(instances: Iterable[Path]) -> list[dict[str, Any]]:
    raise NotImplementedError("compare_sa_rls() ещё не реализована")














# === src\rcpsp_marketing\experiments\decode_incremental_compare_psgs.py ===
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, is_topological_order
from rcpsp_marketing.core.objective import (
    evaluate_profit_over_horizon,
    evaluate_profit_over_horizon_fast,
)
from rcpsp_marketing.core.scheduling import parallel_sgs_selective
from rcpsp_marketing.core.scheduling_parallel_incremental import (
    snapshot_parallel_prefix_by_time,
    parallel_sgs_selective_resume_from_snapshot,
)


# =========================
# Helpers
# =========================

def detect_class_from_path(p: Path) -> str:
    for part in p.parts:
        m = re.match(r"^(j\d+)\.sm$", part.lower())
        if m:
            return m.group(1)
    m2 = re.match(r"^(j\d+)", p.stem.lower())
    return m2.group(1) if m2 else "unknown"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)


def _make_movable_indices(proj, order: List[int]) -> List[int]:
    src = getattr(proj, "source_id", None)
    snk = getattr(proj, "sink_id", None)
    return [idx for idx, j in enumerate(order) if j != src and j != snk]


def _swap(order: List[int], i: int, k: int) -> None:
    order[i], order[k] = order[k], order[i]


def _insert(order: List[int], i: int, k: int) -> None:
    if i == k:
        return
    x = order.pop(i)
    if k > i:
        k -= 1
    order.insert(k, x)


def _pick_neighbor_with_meta(
    proj: Any,
    rnd: random.Random,
    cur_order: List[int],
    *,
    movable_idx: List[int],
    neighbor: str,
    tries: int,
    insert_max_shift: Optional[int],
) -> Optional[Tuple[List[int], int, int, int]]:
    """
    Возвращает (cand_order, i, k, lo) где:
      - i,k: индексы операции (swap/insert)
      - lo = min(i,k) (граница по списку)
    """
    n = len(cur_order)
    for _ in range(tries):
        cand = list(cur_order)

        if neighbor == "swap":
            i, k = rnd.sample(movable_idx, 2)
            _swap(cand, i, k)

        elif neighbor == "insert":
            i = rnd.choice(movable_idx)
            if insert_max_shift is None:
                k = rnd.choice([x for x in movable_idx if x != i])
            else:
                lo_ = max(0, i - insert_max_shift)
                hi_ = min(n - 1, i + insert_max_shift)
                window = [x for x in movable_idx if x != i and lo_ <= x <= hi_]
                k = rnd.choice(window) if window else rnd.choice([x for x in movable_idx if x != i])

            if i == k:
                continue
            _insert(cand, i, k)

        else:
            raise ValueError("neighbor must be 'swap' or 'insert'")

        if is_topological_order(proj, cand):
            lo = min(i, k)
            return cand, i, k, lo

    return None


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    fn: Callable[..., Any]


def _choose_t0(
    *,
    cur_order: List[int],
    cur_res,
    i: int,
    k: int,
    lo: int,
    mode: str,
) -> int:
    """
    Эвристика для t0 (момент времени, с которого пересчитываем PSGS).
    """
    start = cur_res.schedule.start

    if mode == "moved_start":
        moved_jobs = [cur_order[i], cur_order[k]] if i != k else [cur_order[i]]
        t0 = None
        for j in moved_jobs:
            if j in start:
                t0 = start[j] if t0 is None else min(t0, start[j])
            else:
                return 0
        return int(t0 or 0)

    if mode == "changed_segment_min_start":
        hi = max(i, k)
        seg_jobs = cur_order[lo:hi + 1]
        t0 = None
        for j in seg_jobs:
            if j in start:
                t0 = start[j] if t0 is None else min(t0, start[j])
            else:
                return 0
        return int(t0 or 0)

    raise ValueError("Unknown T0_MODE")


# =========================
# Experiment
# =========================

def run_experiment(
    *,
    files: List[Path],
    out_dir: Path,
    T: int,
    seeds: List[int],

    neighbor: str,
    steps: int,
    tries_per_step: int,
    insert_max_shift: Optional[int],

    t0_mode: str,

    objective: str,
    include_dummy_costs: bool,
    abs_tol: float,
    commit_prob: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = PSPLibExtendedParser()

    if objective == "ref":
        obj_spec = ObjectiveSpec("ref", evaluate_profit_over_horizon)
    elif objective == "fast":
        obj_spec = ObjectiveSpec("fast", evaluate_profit_over_horizon_fast)
    else:
        raise ValueError("objective must be 'ref' or 'fast'")

    rows: List[Dict[str, Any]] = []

    for fp in files:
        proj = parser.parse(fp)
        cls = detect_class_from_path(fp)
        inst = safe_name(proj.name or fp.stem)
        print(f"[instance] {cls} / {inst}")

        for s in seeds:
            rnd = random.Random(s)

            cur_order = random_topo_sort_fixed_ends(proj, seed=s)
            movable_idx = _make_movable_indices(proj, cur_order)

            # current decode
            cur_res = parallel_sgs_selective(proj, cur_order, T=T, include_dummies=True, include_sink=False)

            ok_cnt = 0
            bad_cnt = 0
            no_nb = 0
            evals = 0

            full_decode_time = 0.0
            full_obj_time = 0.0
            incr_decode_time = 0.0
            incr_obj_time = 0.0

            for _ in range(steps):
                got = _pick_neighbor_with_meta(
                    proj, rnd, cur_order,
                    movable_idx=movable_idx,
                    neighbor=neighbor,
                    tries=tries_per_step,
                    insert_max_shift=insert_max_shift,
                )
                if got is None:
                    no_nb += 1
                    continue

                cand_order, i, k, lo = got

                # FULL
                t0 = perf_counter()
                full_res = parallel_sgs_selective(proj, cand_order, T=T, include_dummies=True, include_sink=False)
                t1 = perf_counter()
                full_obj = obj_spec.fn(
                    proj,
                    full_res.schedule,
                    selected_jobs=full_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t2 = perf_counter()

                full_decode_time += (t1 - t0)
                full_obj_time += (t2 - t1)

                # choose time boundary t0_time from CURRENT solution (cur_res)
                t0_time = _choose_t0(
                    cur_order=cur_order,
                    cur_res=cur_res,
                    i=i,
                    k=k,
                    lo=lo,
                    mode=t0_mode,
                )

                # INCR: snapshot from current schedule at t0_time, then resume with candidate order
                t3 = perf_counter()
                snap = snapshot_parallel_prefix_by_time(
                    proj,
                    cur_order,  # snapshot строим по текущему порядку и текущему расписанию
                    T=T,
                    include_dummies=True,
                    include_sink=False,
                    base_res=cur_res,
                    t0=t0_time,
                )
                inc_res = parallel_sgs_selective_resume_from_snapshot(
                    proj,
                    cand_order,
                    T=T,
                    include_dummies=True,
                    include_sink=False,
                    snap=snap,
                )
                t4 = perf_counter()
                inc_obj = obj_spec.fn(
                    proj,
                    inc_res.schedule,
                    selected_jobs=inc_res.selected,
                    T=T,
                    include_dummy_costs=include_dummy_costs,
                )
                t5 = perf_counter()

                incr_decode_time += (t4 - t3)
                incr_obj_time += (t5 - t4)

                evals += 1

                # correctness
                same_sched = (full_res.schedule.start == inc_res.schedule.start) and (full_res.schedule.finish == inc_res.schedule.finish)
                same_lists = (full_res.selected == inc_res.selected) and (full_res.skipped == inc_res.skipped)
                dv = float(inc_obj.value) - float(full_obj.value)
                ok = same_sched and same_lists and (abs(dv) <= abs_tol)

                if ok:
                    ok_cnt += 1
                else:
                    bad_cnt += 1

                # simulate accepted move -> update current baseline
                if commit_prob > 0.0 and rnd.random() < commit_prob:
                    cur_order = cand_order
                    movable_idx = _make_movable_indices(proj, cur_order)
                    cur_res = full_res  # приняли -> full_res становится текущим

            full_time = full_decode_time + full_obj_time
            incr_time = incr_decode_time + incr_obj_time

            full_ms = (full_time / max(1, evals)) * 1000.0
            incr_ms = (incr_time / max(1, evals)) * 1000.0
            speedup = (full_ms / incr_ms) if incr_ms > 0 else float("inf")

            full_decode_ms = (full_decode_time / max(1, evals)) * 1000.0
            full_obj_ms = (full_obj_time / max(1, evals)) * 1000.0
            incr_decode_ms = (incr_decode_time / max(1, evals)) * 1000.0
            incr_obj_ms = (incr_obj_time / max(1, evals)) * 1000.0

            rows.append({
                "class": cls,
                "instance": inst,
                "instance_path": str(fp),
                "T": int(T),
                "seed": int(s),

                "neighbor": neighbor,
                "steps": int(steps),
                "tries_per_step": int(tries_per_step),
                "insert_max_shift": (None if insert_max_shift is None else int(insert_max_shift)),

                "t0_mode": t0_mode,

                "objective": obj_spec.name,
                "include_dummy_costs": bool(include_dummy_costs),
                "commit_prob": float(commit_prob),

                "evals": int(evals),
                "no_neighbor": int(no_nb),
                "ok": int(ok_cnt),
                "bad": int(bad_cnt),

                "full_ms_per_eval": float(full_ms),
                "incr_ms_per_eval": float(incr_ms),
                "speedup": float(speedup),

                "full_decode_ms_per_eval": float(full_decode_ms),
                "full_obj_ms_per_eval": float(full_obj_ms),
                "incr_decode_ms_per_eval": float(incr_decode_ms),
                "incr_obj_ms_per_eval": float(incr_obj_ms),
            })

    df = pd.DataFrame(rows)

    runs_path = out_dir / "decode_incremental_psgs_runs.csv"
    df.to_csv(runs_path, index=False, encoding="utf-8")
    print("[saved]", runs_path)

    sum_path = out_dir / "decode_incremental_psgs_summary.csv"
    summary = (
        df.groupby(["class", "neighbor", "t0_mode", "objective"], as_index=False)
          .agg(
              runs=("seed", "count"),
              evals=("evals", "sum"),
              ok=("ok", "sum"),
              bad=("bad", "sum"),
              mean_full_ms=("full_ms_per_eval", "mean"),
              mean_incr_ms=("incr_ms_per_eval", "mean"),
              mean_speedup=("speedup", "mean"),
          )
          .sort_values(["class", "neighbor", "t0_mode", "objective"])
    )
    summary.to_csv(sum_path, index=False, encoding="utf-8")
    print("[saved]", sum_path)

    # weighted report
    if len(df) > 0:
        df["full_total_ms"] = df["evals"] * df["full_ms_per_eval"]
        df["incr_total_ms"] = df["evals"] * df["incr_ms_per_eval"]

        df["full_decode_total_ms"] = df["evals"] * df["full_decode_ms_per_eval"]
        df["full_obj_total_ms"] = df["evals"] * df["full_obj_ms_per_eval"]
        df["incr_decode_total_ms"] = df["evals"] * df["incr_decode_ms_per_eval"]
        df["incr_obj_total_ms"] = df["evals"] * df["incr_obj_ms_per_eval"]

        total_evals = int(df["evals"].sum())

        full_total = float(df["full_total_ms"].sum())
        incr_total = float(df["incr_total_ms"].sum())
        speedup_total = (full_total / incr_total) if incr_total > 0 else float("inf")

        full_decode_avg = float(df["full_decode_total_ms"].sum()) / max(1, total_evals)
        full_obj_avg = float(df["full_obj_total_ms"].sum()) / max(1, total_evals)
        incr_decode_avg = float(df["incr_decode_total_ms"].sum()) / max(1, total_evals)
        incr_obj_avg = float(df["incr_obj_total_ms"].sum()) / max(1, total_evals)

        decode_speedup = (full_decode_avg / incr_decode_avg) if incr_decode_avg > 0 else float("inf")
        obj_speedup = (full_obj_avg / incr_obj_avg) if incr_obj_avg > 0 else float("inf")

        full_share_decode = full_decode_avg / max(1e-12, (full_decode_avg + full_obj_avg))
        incr_share_decode = incr_decode_avg / max(1e-12, (incr_decode_avg + incr_obj_avg))

        ok_total = int(df["ok"].sum())
        bad_total = int(df["bad"].sum())
        ok_rate = ok_total / max(1, (ok_total + bad_total))

        full_total_avg = full_total / max(1, total_evals)
        incr_total_avg = incr_total / max(1, total_evals)
    else:
        total_evals = 0
        speedup_total = 0.0
        decode_speedup = obj_speedup = 0.0
        full_share_decode = incr_share_decode = 0.0
        ok_total = bad_total = 0
        ok_rate = 0.0
        full_total_avg = incr_total_avg = 0.0
        full_decode_avg = full_obj_avg = incr_decode_avg = incr_obj_avg = 0.0

    rep_lines: List[str] = []
    rep_lines.append("End-to-end full vs incremental decode benchmark (PSGS)\n")
    rep_lines.append(f"- runs: {len(df)}")
    rep_lines.append(f"- ok(exact schedule+lists+value): {ok_total} ({ok_rate*100:.1f}%)")
    rep_lines.append(f"- bad: {bad_total}")
    rep_lines.append(f"- evals(total): {total_evals}")
    rep_lines.append("")

    rep_lines.append("Mean speedups (overall, weighted by evals):")
    rep_lines.append(
        f"- total={speedup_total:.3f}x, decode={decode_speedup:.3f}x, objective={obj_speedup:.3f}x, "
        f"full_share_decode={full_share_decode*100:.1f}%, incr_share_decode={incr_share_decode*100:.1f}%"
    )
    rep_lines.append("")
    rep_lines.append("Timing (ms per eval, weighted):")
    rep_lines.append(f"- full: total={full_total_avg:.4f}, decode={full_decode_avg:.4f}, obj={full_obj_avg:.4f}")
    rep_lines.append(f"- incr: total={incr_total_avg:.4f}, decode={incr_decode_avg:.4f}, obj={incr_obj_avg:.4f}")
    rep_lines.append("")
    rep_lines.append("Files:")
    rep_lines.append(f"- {runs_path.name}")
    rep_lines.append(f"- {sum_path.name}")

    rep_path = out_dir / "decode_incremental_psgs_report.txt"
    rep_path.write_text("\n".join(rep_lines), encoding="utf-8")
    print("[saved]", rep_path)


# =========================
# MAIN (edit params here)
# =========================

def main():
    MODE = "dir"  # "single" | "dir"

    SINGLE_FILE = Path(r"data/extended/j30.sm/j301_1_with_metrics.sm")
    IN_DIR = Path(r"data/extended/j30.sm")
    PATTERN = "*_with_metrics.sm"
    MAX_INSTANCES = 10  # 0 = no limit

    OUT_DIR = Path(r"data/experiments/decode_incremental_compare_psgs")

    T = 50
    SEEDS = list(range(1, 11))

    NEIGHBOR = "swap"          # "swap" | "insert"
    STEPS = 2000
    TRIES_PER_STEP = 50
    INSERT_MAX_SHIFT = 10

    T0_MODE = "changed_segment_min_start"  # "moved_start" | "changed_segment_min_start"

    OBJECTIVE = "fast"
    INCLUDE_DUMMY_COSTS = False
    ABS_TOL = 1e-9
    COMMIT_PROB = 0.15

    if MODE == "single":
        files = [SINGLE_FILE]
    elif MODE == "dir":
        files = [p for p in sorted(IN_DIR.rglob(PATTERN)) if p.is_file()]
        if MAX_INSTANCES and MAX_INSTANCES > 0:
            files = files[:MAX_INSTANCES]
    else:
        raise ValueError("MODE must be 'single' or 'dir'")

    print(
        f"[info] mode={MODE} instances={len(files)} neighbor={NEIGHBOR} steps={STEPS} T={T} "
        f"objective={OBJECTIVE} T0_MODE={T0_MODE} commit_prob={COMMIT_PROB}"
    )

    run_experiment(
        files=files,
        out_dir=OUT_DIR,
        T=T,
        seeds=SEEDS,

        neighbor=NEIGHBOR,
        steps=STEPS,
        tries_per_step=TRIES_PER_STEP,
        insert_max_shift=INSERT_MAX_SHIFT,

        t0_mode=T0_MODE,

        objective=OBJECTIVE,
        include_dummy_costs=INCLUDE_DUMMY_COSTS,
        abs_tol=ABS_TOL,
        commit_prob=COMMIT_PROB,
    )


if __name__ == "__main__":
    t0 = perf_counter()
    main()
    dt = perf_counter() - t0
    print(f"[done] total time: {dt:.2f} sec ({dt/60:.2f} min)")


# === src\rcpsp_marketing\experiments\decoder_compare.py ===
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd

from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends, topo_sort
from rcpsp_marketing.core.scheduling import serial_sgs_selective

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective
except Exception:
    parallel_sgs_selective = None

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective_greedy
except Exception:
    parallel_sgs_selective_greedy = None


DecoderFn = Callable[..., Any]


def _run_one(
    name: str,
    decoder_fn: DecoderFn,
    proj: Any,
    order: list[int],
    T: int,
    **kwargs,
) -> dict[str, Any]:
    t0 = perf_counter()
    res = decoder_fn(proj, order, T=T, include_dummies=True, **kwargs)
    dt = perf_counter() - t0

    obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

    return {
        "decoder": name,
        "value": obj.value,
        "revenue": obj.revenue,
        "cost": obj.cost,
        "selected": len(res.selected),
        "skipped": len(res.skipped),
        "makespan": res.schedule.makespan,
        "time_sec": dt,
    }


def compare_decoders(
    proj: Any,
    *,
    T: int,
    order: list[int] | None = None,
    order_mode: str = "random_fixed_ends",   # "random_fixed_ends" | "topo_smallest"
    seed: int = 42,
    greedy_min_score: float = -1e18,
    greedy_unlock_weight: float = 0.0,
    save_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Сравнивает декодеры на одном и том же priority list (без поисковых методов/улучшателей).
    Возвращает DataFrame, можно сохранить в CSV.
    """

    if order is None:
        if order_mode == "random_fixed_ends":
            order = random_topo_sort_fixed_ends(proj, seed=seed)
        elif order_mode == "topo_smallest":
            order = topo_sort(proj, prefer="smallest")
        else:
            raise ValueError(f"Unknown order_mode={order_mode}")

    rows: list[dict[str, Any]] = []
    rows.append(_run_one("SSGS selective", serial_sgs_selective, proj, order, T))

    if parallel_sgs_selective is not None:
        rows.append(_run_one("PSGS selective", parallel_sgs_selective, proj, order, T, include_sink=False))

    if parallel_sgs_selective_greedy is not None:
        rows.append(_run_one(
            "PSGS greedy-score",
            parallel_sgs_selective_greedy,
            proj, order, T,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        ))

    df = pd.DataFrame(rows).sort_values("value", ascending=False)

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv_path, index=False)

    return df


# === src\rcpsp_marketing\experiments\decoder_visualize.py ===
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html
from rcpsp_marketing.core.scheduling import serial_sgs_selective

# optional decoders
try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective
except Exception:
    parallel_sgs_selective = None

try:
    from rcpsp_marketing.core.scheduling import parallel_sgs_selective_greedy
except Exception:
    parallel_sgs_selective_greedy = None


DecoderFn = Callable[..., Any]


def _run_and_save(
    *,
    name: str,
    decoder_fn: DecoderFn,
    proj: Any,
    order: list[int],
    T: int,
    out_dir: Path,
    **kwargs,
) -> Path:
    t0 = perf_counter()
    res = decoder_fn(proj, order, T=T, include_dummies=True, **kwargs)
    dt = perf_counter() - t0

    obj = evaluate_profit_over_horizon(proj, res.schedule, selected_jobs=res.selected, T=T)

    title = (
        f"{proj.name} | {name} | T={T} | "
        f"value={obj.value:_.2f} | sel={len(res.selected)} skip={len(res.skipped)} | "
        f"time={dt:.4f}s"
    )
    fig = plot_schedule_gantt(proj, res.schedule, selected=res.selected, title=title, T=T)

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )
    out_path = out_dir / f"{proj.name}_T{T}_{safe_name}.html"
    return save_schedule_html(fig, out_path)


def visualize_decoder_packings(
    proj: Any,
    *,
    order: list[int],
    T: int,
    out_dir: str | Path = "data/experiments/viz",
    greedy_min_score: float = -1e18,
    greedy_unlock_weight: float = 0.0,
) -> dict[str, Path]:
    """
    Сохраняет Gantt-упаковку (schedule packing) для каждого доступного декодера.
    Возвращает: {decoder_name: html_path}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    # 1) SSGS
    saved["SSGS selective"] = _run_and_save(
        name="SSGS selective",
        decoder_fn=serial_sgs_selective,
        proj=proj,
        order=order,
        T=T,
        out_dir=out_dir,
    )

    # 2) PSGS (если есть)
    if parallel_sgs_selective is not None:
        saved["PSGS selective"] = _run_and_save(
            name="PSGS selective",
            decoder_fn=parallel_sgs_selective,
            proj=proj,
            order=order,
            T=T,
            out_dir=out_dir,
            include_sink=False,
        )

    # 3) PSGS greedy-score (если есть)
    if parallel_sgs_selective_greedy is not None:
        saved["PSGS greedy-score"] = _run_and_save(
            name="PSGS greedy-score",
            decoder_fn=parallel_sgs_selective_greedy,
            proj=proj,
            order=order,
            T=T,
            out_dir=out_dir,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    return saved


# === src\rcpsp_marketing\experiments\sa_batch.py ===
"""Заглушка батч-запуска SA по набору инстансов."""

from pathlib import Path
from typing import Any, Iterable


def run_sa_batch(instances: Iterable[Path]) -> list[Any]:
    raise NotImplementedError("run_sa_batch() ещё не реализована")














# === src\rcpsp_marketing\io\__init__.py ===
"""
Модули ввода/вывода:
- парсинг PSPLib
- генерация расширенных инстансов
- экспорт расписаний.
"""













# === src\rcpsp_marketing\io\psplib_base.py ===
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


# === src\rcpsp_marketing\io\psplib_extended.py ===
"""
Парсер расширенного формата с маркетинговыми метриками.

План:
- на базе результата `load_psplib_base` добавлять поля из *_with_metrics.sm;
- мэппить их в модели из `rcpsp_marketing.data.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import re

from rcpsp_marketing.data.models import Project, ServiceMetrics, MetricEffectsPct
from rcpsp_marketing.io.psplib_base import PSPLibBaseParser, _find_block, _clean_line, _re_int, _re_float


def parse_service_metrics(content: str) -> Optional[ServiceMetrics]:
    block = _find_block(
        content,
        r"SERVICE\s+METRICS\s*\(INITIAL\)",
        r"(RESOURCE\s+COSTS|TASK\s+COSTS|METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    if not block:
        return None

    def find_float(name: str, default: float) -> float:
        m = re.search(rf"\b{name}\s*=\s*({_re_float.pattern})", block, flags=re.I)
        return float(m.group(1)) if m else default

    def find_int(name: str, default: int) -> int:
        m = re.search(rf"\b{name}\s*=\s*({_re_int.pattern})", block, flags=re.I)
        return int(m.group(1)) if m else default

    return ServiceMetrics(
        LT_0=find_float("LT_0", 0.0),
        AC_0=find_int("AC_0", 0),
        CPC_0=find_float("CPC_0", 0.0),
        LCR_0=find_float("LCR_0", 0.0),
        PCR_0=find_float("PCR_0", 0.0),
        CAC_0=find_float("CAC_0", 0.0),
        MARGIN0=find_float("MARGIN0", 0.0),
    )


def parse_resource_costs(content: str) -> Dict[int, float]:
    block = _find_block(
        content,
        r"RESOURCE\s+COSTS\s*\(per\s+unit\s+per\s+time\)",
        r"(TASK\s+COSTS|METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    costs: Dict[int, float] = {}
    if not block:
        return costs
    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln:
            continue
        m = re.search(r"R(\d+)\s*=\s*(" + _re_float.pattern + r")", ln, flags=re.I)
        if m:
            costs[int(m.group(1))] = float(m.group(2))
    return costs


def parse_task_costs(content: str) -> Dict[int, float]:
    block = _find_block(
        content,
        r"TASK\s+COSTS\s*\(total\)",
        r"(METRIC\s+CHANGES|^\*{3,}|END\s+OF\s+FILE)"
    )
    costs: Dict[int, float] = {}
    if not block:
        return costs
    for raw in block.splitlines():
        ln = _clean_line(raw)
        if not ln:
            continue
        m = re.search(r"^\s*(\d+)\s*:\s*(" + _re_float.pattern + r")", ln)
        if m:
            costs[int(m.group(1))] = float(m.group(2))
    return costs


def parse_metric_changes(content: str) -> Dict[int, tuple[str, MetricEffectsPct]]:
    block = _find_block(
        content,
        r"METRIC\s+CHANGES\s*\(per\s+job.*?\)",
        r"(END\s+OF\s+FILE|^\*{3,})"
    )
    out: Dict[int, tuple[str, MetricEffectsPct]] = {}
    if not block:
        return out

    # формат генератора: "job : type  AC% LT% CPC% LCR% PCR%" :contentReference[oaicite:6]{index=6}
    line_re = re.compile(
        r"^\s*(\d+)\s*:\s*([A-Za-z_]+)\s*"
        r"([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)\s+([+\-]?\d+(?:\.\d+)?)",
        flags=re.I | re.M
    )

    for m in line_re.finditer(block):
        j = int(m.group(1))
        jt = m.group(2).lower()
        eff = MetricEffectsPct(
            AC=float(m.group(3)),
            LT=float(m.group(4)),
            CPC=float(m.group(5)),
            LCR=float(m.group(6)),
            PCR=float(m.group(7)),
        )
        out[j] = (jt, eff)

    return out


@dataclass(slots=True)
class PSPLibExtendedParser:
    """
    Парсер extended-файлов: raw PSPLIB + блоки SERVICE/RESOURCE COSTS/TASK COSTS/METRIC CHANGES.
    """

    base: PSPLibBaseParser = field(default_factory=PSPLibBaseParser)

    def parse(self, path: str | Path, *, name: str = "") -> Project:
        path = Path(path)

        # 1) базовый проект
        proj = self.base.parse(path, name=name or path.stem)

        # 2) читаем всё содержимое для расширенных блоков
        content = path.read_text(encoding="utf-8", errors="replace")

        proj.metrics0 = parse_service_metrics(content)
        proj.resource_costs = parse_resource_costs(content)

        task_costs = parse_task_costs(content)
        changes = parse_metric_changes(content)

        # 3) приклеиваем к задачам
        for j, t in proj.tasks.items():
            if j in task_costs:
                t.total_cost = float(task_costs[j])
            if j in changes:
                jt, eff = changes[j]
                t.job_type = jt
                t.effects_pct = eff

        return proj


# === src\rcpsp_marketing\io\schedule_export.py ===
"""
Экспорт расписаний и истории запусков в CSV/JSON.
"""

from pathlib import Path
from typing import Any


def export_schedule_to_csv(schedule: Any, path: Path) -> None:
    """Заглушка экспорта расписания в CSV."""
    raise NotImplementedError("export_schedule_to_csv() ещё не реализована")














# === src\rcpsp_marketing\viz\__init__.py ===
"""Визуализации расписаний и истории поиска."""













# === src\rcpsp_marketing\viz\dag.py ===
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import plotly.graph_objects as go


def _edges_from_project(proj) -> List[Tuple[int, int]]:
    return [(u, v) for u, succs in proj.successors.items() for v in succs]


def _levels_lr_layout(proj) -> Dict[int, Tuple[float, float]]:
    """
    Fallback-layout без graphviz:
    x = уровень (max depth по предшественникам),
    y = индекс внутри уровня.
    """
    # топологический порядок (простая Kahn)
    indeg = {i: len(proj.predecessors.get(i, [])) for i in proj.tasks}
    q = [i for i, d in indeg.items() if d == 0]
    order = []
    while q:
        v = q.pop()
        order.append(v)
        for u in proj.successors.get(v, []):
            indeg[u] -= 1
            if indeg[u] == 0:
                q.append(u)

    # уровни
    level: Dict[int, int] = {}
    for v in order:
        preds = proj.predecessors.get(v, [])
        if not preds:
            level[v] = 0
        else:
            level[v] = 1 + max(level[p] for p in preds)

    # группировка по уровням → y
    buckets: Dict[int, List[int]] = {}
    for v, lv in level.items():
        buckets.setdefault(lv, []).append(v)
    for lv in buckets:
        buckets[lv].sort()

    pos: Dict[int, Tuple[float, float]] = {}
    for lv, nodes in sorted(buckets.items()):
        for idx, v in enumerate(nodes):
            pos[v] = (float(lv), float(-idx))  # минус, чтобы сверху вниз
    return pos


def _graphviz_dot_lr_layout(proj) -> Dict[int, Tuple[float, float]]:
    """
    Graphviz dot layout слева→направо.
    Требует установленный graphviz + python-пакеты networkx и pydot.
    """
    import networkx as nx

    edges = _edges_from_project(proj)
    G = nx.DiGraph(edges)
    G.graph["graph"] = {"rankdir": "LR"}  # left-to-right

    # pydot/graphviz
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    # graphviz_layout возвращает int coords; приводим к float
    return {k: (float(x), float(y)) for k, (x, y) in pos.items()}


def plot_dag(
    proj,
    *,
    layout: str = "dot_lr",
    show_labels: bool = True,
    max_edges: Optional[int] = None,
    title: Optional[str] = None,
) -> go.Figure:
    """
    layout:
      - "dot_lr": Graphviz dot слева→направо (красиво), нужен graphviz
      - "levels_lr": fallback без graphviz (уровни по предшественникам)
    max_edges: если граф большой, можно ограничить количество рёбер (для скорости)
    """
    edges = _edges_from_project(proj)
    if max_edges is not None and len(edges) > max_edges:
        edges = edges[:max_edges]

    if layout == "dot_lr":
        try:
            pos = _graphviz_dot_lr_layout(proj)
        except Exception:
            # fallback
            pos = _levels_lr_layout(proj)
            layout = "levels_lr"
    elif layout == "levels_lr":
        pos = _levels_lr_layout(proj)
    else:
        raise ValueError(f"Unknown layout='{layout}'")

    # рёбра
    edge_x, edge_y = [], []
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none"
    )

    # узлы
    node_ids = sorted(proj.tasks.keys())
    node_x, node_y = [], []
    hover = []
    labels = []
    for n in node_ids:
        if n not in pos:
            continue
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        t = proj.tasks[n]
        hover.append(f"job {n}<br>dur={t.duration}<br>type={getattr(t, 'job_type', 'n/a')}")
        labels.append(str(n))

    node_mode = "markers+text" if show_labels else "markers"
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode=node_mode,
        text=labels if show_labels else None,
        textposition="middle right",
        hovertext=hover,
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title or f"Precedence DAG ({layout})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def save_dag_html(
    proj,
    out_path: str | Path,
    *,
    layout: str = "dot_lr",
    show_labels: bool = True,
    max_edges: Optional[int] = None,
    title: Optional[str] = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_dag(
        proj,
        layout=layout,
        show_labels=show_labels,
        max_edges=max_edges,
        title=title,
    )
    fig.write_html(str(out_path))
    return out_path


# === src\rcpsp_marketing\viz\packing.py ===
"""Заглушка Plotly-графиков по ресурсам (пакующие диаграммы и т.п.)."""

from typing import Any


def plot_resource_packing(schedule: Any) -> None:
    raise NotImplementedError("plot_resource_packing() ещё не реализована")














# === src\rcpsp_marketing\viz\schedule.py ===
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import plotly.graph_objects as go

from rcpsp_marketing.data.models import Project, Schedule


def plot_schedule_gantt(
    project: Project,
    schedule: Schedule,
    *,
    selected: Optional[Iterable[int]] = None,
    hide_dummies: bool = True,
    title: str = "Schedule",
    T: Optional[int] = None,
):
    if selected is None:
        selected = schedule.start.keys()

    rows = []
    for j in selected:
        if j not in schedule.start:
            continue
        if hide_dummies and j in (project.source_id, project.sink_id):
            continue
        t = project.tasks[j]
        rows.append({
            "job": str(j),
            "start": int(schedule.start[j]),
            "finish": int(schedule.finish[j]),
            "type": getattr(t, "job_type", "unknown"),
            "dur": int(schedule.finish[j] - schedule.start[j]),
            "cost": float(getattr(t, "total_cost", 0.0)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure().update_layout(title=title)

    df = df.sort_values(["start", "job"])

    fig = go.Figure()

    for ttype, g in df.groupby("type"):
        fig.add_trace(go.Bar(
            name=ttype,
            y=g["job"],
            x=g["dur"],        # длина бара
            base=g["start"],   # начало
            orientation="h",
            hovertext=[
                f"job={job}<br>start={s}<br>finish={f}<br>dur={d}<br>type={tt}<br>cost={c:.2f}"
                for job, s, f, d, tt, c in zip(g["job"], g["start"], g["finish"], g["dur"], g["type"], g["cost"])
            ],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis=dict(title="time (units)", type="linear"),
        yaxis=dict(title="job", autorange="reversed"),
    )

    if T is not None:
        fig.add_vline(x=T, line_dash="dash")

    return fig


def save_schedule_html(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    return out_path


# === src\rcpsp_marketing\viz\search_history.py ===
"""Заглушка визуализации истории поиска (текущее vs лучшее, температура SA и т.п.)."""

from typing import Sequence


def plot_search_history(best_values: Sequence[float], current_values: Sequence[float] | None = None) -> None:
    raise NotImplementedError("plot_search_history() ещё не реализована")














# === test.py ===
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import plotly.graph_objects as go

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.core.precedence import random_topo_sort_fixed_ends
from rcpsp_marketing.core.scheduling import serial_sgs_selective, parallel_sgs_selective_greedy, parallel_sgs_selective
from rcpsp_marketing.core.objective import evaluate_profit_over_horizon
from rcpsp_marketing.core.improvement import left_shift
from rcpsp_marketing.viz.schedule import plot_schedule_gantt, save_schedule_html
from rcpsp_marketing.algorithms.local_search.simulated_annealing import simulated_annealing

def fmt_obj(obj) -> str:
    return f"value={obj.value:_.2f} revenue={obj.revenue:_.2f} cost={obj.cost:_.2f} makespan={obj.makespan}"


def save_sa_history_html(best_hist: List[float], cur_hist: List[float], out_path: Path, title: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=best_hist, mode="lines", name="best_value"))
    fig.add_trace(go.Scatter(y=cur_hist, mode="lines", name="current_value"))
    fig.update_layout(
        title=title,
        xaxis=dict(title="iteration"),
        yaxis=dict(title="value"),
        showlegend=True,
    )
    fig.write_html(str(out_path))
    return out_path


def main():


    # ====== параметры ======
    instance_path = r"data\extended\j120.sm\j1201_2_with_metrics.sm"
    T = 80
    seed_order = 42
    seed_sa = 123

    # PSGS-greedy (если хочешь финально упаковать этим декодером)
    greedy_min_score = -1e18
    greedy_unlock_weight = 0.0

    # SA параметры
    sa_iters = 50_000          # 500k на j120 может быть очень долго
    sa_T0 = 2e4
    sa_alpha = 0.9998
    sa_Tmin = 1.0
    sa_neighbor = "insert"      # "swap" быстрее, "insert" сильнее
    sa_tries = 30

    # ====== декодеры ======

    # SSGS selective 
    def decode_ssgs(proj_, order_, T_):
        return serial_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
        ) 

    # PSGS selective
    def decode_psgs(proj_, order_, T_):
        return parallel_sgs_selective(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
        )   

    # PSGS with greedy packing rule
    def decode_psgs_greedy(proj_, order_, T_):
        return parallel_sgs_selective_greedy(
            proj_,
            order_,
            T=T_,
            include_dummies=True,
            include_sink=False,
            min_score=greedy_min_score,
            unlock_weight=greedy_unlock_weight,
        )

    # ====== стартовый порядок ======
    order0 = random_topo_sort_fixed_ends(proj, seed=seed_order)
    print("[info] start order len =", len(order0), "first10 =", order0[:10], "last5 =", order0[-5:])



    out_dir = Path("data/experiments/viz/sa_run")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== загрузка проекта ======
    proj = PSPLibExtendedParser().parse(instance_path)
    print("[ok] parsed:", proj.name)
    print("[info] T =", T)



if __name__ == "__main__":
    main()

