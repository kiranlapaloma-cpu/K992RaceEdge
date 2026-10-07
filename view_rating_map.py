from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd
import streamlit as st


_SCENARIOS = {
    "Recent Best": "Recent Best Projection",
    "Latest Form": "Latest Projection",
    "Peak Ability": "Peak Projection",
}


def _number(value) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
        return str(int(round(float(value))))
    except Exception:
        return ""


def _fmt_points(value: float) -> str:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def _assign_lanes(work: pd.DataFrame, min_gap_pct: float = 17.0, lanes: int = 4) -> list[int]:
    """Greedy label staggering so nearby markers do not print on top of each other."""
    last_x = [-999.0] * lanes
    assigned = []
    # Work is displayed left -> right.
    for x in work["_x"].tolist():
        choices = [i for i in range(lanes) if x - last_x[i] >= min_gap_pct]
        lane = choices[0] if choices else min(range(lanes), key=lambda i: last_x[i])
        last_x[lane] = x
        assigned.append(lane)
    return assigned


def render_rating_map(prediction: dict | None):
    """
    Optional Race Edge race-day placement visual.

    Uses the existing projected ratings only; it does not recalculate or alter
    Race Edge prediction maths. Right = better placed. The best projected runner
    is the zero-point benchmark; every other runner shows the improvement needed
    to match that benchmark on today's terms.
    """
    st.markdown("### Race Edge Rating Map")
    st.caption(
        "Left = least well in • Right = most well in. "
        "The right-most runner is the 0-point benchmark; others show the points needed to match it."
    )

    if not prediction:
        st.info("No Race Edge prediction is available for this race.")
        return

    rows = prediction.get("rows")
    if rows is None or rows.empty:
        st.info("No runners have enough saved Race Edge history for the Rating Map.")
        return

    scenario = st.radio(
        "Rating view",
        list(_SCENARIOS.keys()),
        horizontal=True,
        index=0,
        key=f"rating_map_scenario_{prediction.get('race_date')}_{prediction.get('distance')}",
    )
    projection_col = _SCENARIOS[scenario]
    if projection_col not in rows.columns:
        st.info(f"{scenario} ratings are not available for this race.")
        return

    work = rows[[c for c in ["No.", "Horse", projection_col] if c in rows.columns]].copy()
    work[projection_col] = pd.to_numeric(work[projection_col], errors="coerce")
    work = work.dropna(subset=[projection_col])
    if work.empty:
        st.info(f"No runners have a usable {scenario} rating.")
        return

    leader = float(work[projection_col].max())
    work["Points Needed"] = (leader - work[projection_col]).clip(lower=0.0)

    # The axis is relative placement, not raw MR. Preserve meaningful spacing while
    # giving a one-runner or tied field a stable visual position.
    max_needed = float(work["Points Needed"].max())
    axis_max = max(2.0, math.ceil(max_needed / 2.0) * 2.0)
    work["_x"] = 100.0 * (1.0 - work["Points Needed"] / axis_max)
    work["_x"] = work["_x"].clip(lower=0.0, upper=100.0)
    work = work.sort_values(["_x", "No.", "Horse"], ascending=[True, True, True]).reset_index(drop=True)
    work["_lane"] = _assign_lanes(work)

    lane_height = 58
    top_pad = 22
    axis_y = top_pad + 4 * lane_height + 18
    total_height = axis_y + 58

    labels = []
    markers = []
    for _, row in work.iterrows():
        x = float(row["_x"])
        lane = int(row["_lane"])
        y = top_pad + lane * lane_height
        horse = html.escape(str(row.get("Horse") or ""))
        no = _number(row.get("No."))
        title = f"{('#' + no + ' ') if no else ''}{horse}"
        needed = float(row["Points Needed"])
        status = "BEST PLACED • 0 pts" if needed < 1e-9 else f"NEEDS {_fmt_points(needed)} pts"
        # Keep edge labels inside the card while preserving their marker position.
        if x < 12:
            transform = "translateX(0)"
            align = "left"
        elif x > 88:
            transform = "translateX(-100%)"
            align = "right"
        else:
            transform = "translateX(-50%)"
            align = "center"
        labels.append(
            f'<div style="position:absolute;left:{x:.2f}%;top:{y}px;transform:{transform};'
            f'text-align:{align};white-space:nowrap;line-height:1.15;">'
            f'<div style="font-weight:700;font-size:13px;">{title}</div>'
            f'<div style="font-size:11px;opacity:.72;margin-top:3px;">{status}</div></div>'
        )
        marker_top = y + 39
        line_height = max(8, axis_y - marker_top)
        markers.append(
            f'<div style="position:absolute;left:{x:.2f}%;top:{marker_top}px;height:{line_height}px;'
            f'border-left:1px solid rgba(128,128,128,.45);"></div>'
            f'<div style="position:absolute;left:calc({x:.2f}% - 5px);top:{axis_y-5}px;width:10px;height:10px;'
            f'border-radius:50%;background:#C7A34A;border:2px solid #17243A;box-sizing:border-box;"></div>'
        )

    chart = f'''\n<div style="position:relative;height:{total_height}px;margin:8px 8px 4px 8px;">\n  {''.join(labels)}\n  {''.join(markers)}\n  <div style="position:absolute;left:0;right:0;top:{axis_y}px;height:3px;background:#17243A;border-radius:3px;"></div>\n  <div style="position:absolute;left:0;top:{axis_y+16}px;font-size:12px;font-weight:700;letter-spacing:.02em;">← LEAST WELL IN</div>\n  <div style="position:absolute;right:0;top:{axis_y+16}px;font-size:12px;font-weight:700;letter-spacing:.02em;text-align:right;">MOST WELL IN →</div>\n</div>\n'''
    st.markdown(chart, unsafe_allow_html=True)

    # Compact numerical audit beneath the visual.
    audit = work[[c for c in ["No.", "Horse", projection_col, "Points Needed"] if c in work.columns]].copy()
    audit = audit.sort_values("Points Needed", ascending=True).reset_index(drop=True)
    audit[projection_col] = pd.to_numeric(audit[projection_col], errors="coerce").round(1)
    audit["Points Needed"] = pd.to_numeric(audit["Points Needed"], errors="coerce").round(1)
    audit = audit.rename(columns={projection_col: "Race Edge Rating", "Points Needed": "Needs (pts)"})
    if "No." in audit.columns:
        audit["No."] = pd.to_numeric(audit["No."], errors="coerce").round().astype("Int64")
    st.dataframe(audit, width="stretch", hide_index=True)
