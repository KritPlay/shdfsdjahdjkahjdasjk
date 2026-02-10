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
