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
