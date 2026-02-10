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
    