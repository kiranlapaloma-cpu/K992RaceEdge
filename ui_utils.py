"""Streamlit safety guards and plotting helpers used across Race Edge."""
import math
import numpy as np
import pandas as pd
import streamlit as st

# ======================= Global NaN/Inf → None guard (JSON-safe, index-safe) =======================

def _is_nanlike(x):
    try:
        return (
            x is None
            or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
            or (isinstance(x, np.floating) and (np.isnan(x) or np.isinf(x)))
        )
    except Exception:
        return False

def _san_df(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.replace([np.inf, -np.inf], np.nan).where(lambda d: d.notna(), None)
    clean.index = [None if _is_nanlike(v) else v for v in clean.index.tolist()]
    clean.columns = [None if _is_nanlike(v) else v for v in clean.columns.tolist()]
    return clean.astype("object")

def _san_ser(s: pd.Series) -> pd.Series:
    ss = s.replace([np.inf, -np.inf], np.nan)
    ss = ss.where(ss.notna(), None)
    ss.index = [None if _is_nanlike(v) else v for v in ss.index.tolist()]
    return ss.astype("object")

def _sanitize(obj):
    if isinstance(obj, pd.DataFrame):
        return _san_df(obj).reset_index(drop=True)
    if isinstance(obj, pd.Series):
        return _san_ser(obj).reset_index(drop=True)

    try:
        from pandas.io.formats.style import Styler
        if isinstance(obj, Styler):
            sty = obj
            sty.data = _san_df(sty.data).reset_index(drop=True)
            return sty
    except Exception:
        pass

    if isinstance(obj, np.ndarray):
        return [_sanitize(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_sanitize(v) for v in obj)

    return None if _is_nanlike(obj) else obj

# ---- Patch common emitters (incl. data_editor) ----
_orig_write = st.write
def _safe_write(*args, **kwargs):
    return _orig_write(
        *(_sanitize(a) for a in args),
        **{k: _sanitize(v) for k, v in kwargs.items()}
    )
st.write = _safe_write

_orig_json = st.json
st.json = lambda obj, *a, **k: _orig_json(_sanitize(obj), *a, **k)

_orig_metric = st.metric
def _safe_metric(label, value, delta=None, *a, **k):
    v = _sanitize(value)
    d = _sanitize(delta)
    return _orig_metric(
        label,
        "-" if v is None else v,
        "-" if (delta is not None and d is None) else d,
        *a, **k
    )
st.metric = _safe_metric

_orig_dataframe = st.dataframe
def _safe_dataframe(data=None, *a, **k):
    data = _sanitize(data)
    try:
        if isinstance(data, pd.DataFrame):
            data = data.reset_index(drop=True)
    except Exception:
        pass
    return _orig_dataframe(data, *a, **k)
st.dataframe = _safe_dataframe

_orig_table = st.table
st.table = lambda data=None, *a, **k: _orig_table(_sanitize(data), *a, **k)

if hasattr(st, "data_editor"):
    _orig_editor = st.data_editor
    st.data_editor = lambda data=None, *a, **k: _orig_editor(_sanitize(data), *a, **k)

_orig_download_button = st.download_button
def _safe_download_button(*a, **k):
    if "data" in k:
        k["data"] = _sanitize(k["data"])
        if isinstance(k["data"], (pd.DataFrame, pd.Series)):
            k["data"] = k["data"].to_csv(index=False).encode("utf-8")
    return _orig_download_button(*a, **k)
st.download_button = _safe_download_button
# ======================= /Global guard =======================
# ----------------------- Global plot helpers -----------------------
def _repel_labels_builtin(ax, x, y, labels, *, init_shift=0.18, k_attract=0.006, k_repel=0.012, max_iter=250):
    trans = ax.transData
    renderer = ax.figure.canvas.get_renderer()
    xy = np.column_stack([x, y]).astype(float)
    offs = np.zeros_like(xy)
    for i, (xi, yi) in enumerate(xy):
        offs[i] = [init_shift if xi >= 0 else -init_shift, init_shift if yi >= 0 else -init_shift]

    texts, lines = [], []
    for (xi, yi), (dx, dy), lab in zip(xy, offs, labels):
        t = ax.text(xi + dx, yi + dy, lab, fontsize=8.4, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75),
                    zorder=5)
        texts.append(t)
        lines.append(ax.plot([xi, xi + dx], [yi, yi + dy], color="gray", lw=0.45, alpha=0.45, zorder=4)[0])

    ax.figure.canvas.draw()
    for _ in range(max_iter):
        moved = 0.0
        bbs = [t.get_window_extent(renderer=renderer).expanded(1.06, 1.12) for t in texts]
        centers = []
        for bb in bbs:
            centers.append(((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2))

        for i, t in enumerate(texts):
            xi, yi = xy[i]
            tx, ty = t.get_position()
            px, py = trans.transform((tx, ty))
            ox, oy = trans.transform((xi, yi))

            fx = (ox - px) * k_attract
            fy = (oy - py) * k_attract

            for j in range(len(texts)):
                if i == j:
                    continue
                bb_i, bb_j = bbs[i], bbs[j]
                if not bb_i.overlaps(bb_j):
                    continue
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                dxp = (cx_i - cx_j) if (cx_i - cx_j) != 0 else (1.0 if i < j else -1.0)
                dyp = (cy_i - cy_j) if (cy_i - cy_j) != 0 else (1.0 if i < j else -1.0)
                dist = max((dxp * dxp + dyp * dyp) ** 0.5, 1.0)
                fx += (dxp / dist) * k_repel * 12.0
                fy += (dyp / dist) * k_repel * 12.0

            new_tx, new_ty = trans.inverted().transform((px + fx, py + fy))
            moved += abs(new_tx - tx) + abs(new_ty - ty)
            t.set_position((new_tx, new_ty))
            lines[i].set_data([xi, new_tx], [yi, new_ty])

        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        if moved < 1e-3:
            break


def label_points_neatly(ax, x, y, labels):
    try:
        _repel_labels_builtin(ax, x, y, labels)
    except Exception:
        for xi, yi, lab in zip(x, y, labels):
            ax.text(xi, yi, str(lab), fontsize=8.2, va="center", ha="left", zorder=5)


