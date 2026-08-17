import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DISTANCE_STD_100M = {
    1000: 5.13,
    1160: 5.28,
    1200: 5.35,
    1400: 5.65,
    1450: 5.75,
    1600: 5.90,
    1750: 5.92,
    1800: 5.93,
    1900: 5.94,
    1950: 5.95,
    2000: 5.95,
    2200: 6.02,
    2400: 6.10,
    2450: 6.12,
    2600: 6.20,
    2800: 6.30,
    3000: 6.40,
    3200: 6.50,
}


def mad_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad

def winsorize(s, p_lo=0.10, p_hi=0.90):
    try:
        lo = s.quantile(p_lo); hi = s.quantile(p_hi)
        return s.clip(lower=lo, upper=hi)
    except Exception:
        return s

def color_cycle(n):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'])
    out, i = [], 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out

def safe_num(x, default=0.0):
    """Return a finite float; else default."""
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)

def sanitize_jsonable(obj, ndigits=3):
    """
    Recursively convert NaN/Inf -> None and round floats.
    Use before st.json/vega/altair or writing dicts to state.
    """
    if obj is None:
        return None
    if isinstance(obj, (float, np.floating)):
        if not np.isfinite(obj): return None
        return round(float(obj), ndigits)
    if isinstance(obj, (int, np.integer, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_jsonable(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_jsonable(v, ndigits) for v in obj]
    if isinstance(obj, pd.Series):
        return [sanitize_jsonable(v, ndigits) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return [sanitize_jsonable(r, ndigits) for _, r in obj.iterrows()]
    # fallback
    try:
        return sanitize_jsonable(float(obj), ndigits)
    except Exception:
        return None

def _safe_bal_norm(series, center=100.0, pad=0.5):
    """Return a TwoSlopeNorm that always satisfies vmin < center < vmax.
    Falls back to a tiny padded range around the data if it's one-sided or flat."""
    arr = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        vmin, vmax = center - 5.0, center + 5.0
        return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    # If all values are the same, make a tiny symmetric band around the value
    if vmin == vmax:
        vmin = vmin - max(pad, 0.1)
        vmax = vmax + max(pad, 0.1)

    # Ensure the center sits strictly inside (vmin, vmax)
    if vmax <= center:
        vmax = center + max(pad, (center - vmin) * 0.05 + 0.1)
    if vmin >= center:
        vmin = center - max(pad, (vmax - center) * 0.05 + 0.1)

    # Final guard: if still touching, nudge a hair
    if not (vmin < center < vmax):
        if vmin >= center:
            vmin = center - 0.1
        if vmax <= center:
            vmax = center + 0.1

    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)

def _benchmark_std_100m(distance_m: float) -> float:
    """Practical trip benchmark ladder anchored at 5.50s per 100m for 1000m races."""
    knots = sorted((float(k), float(v)) for k, v in DISTANCE_STD_100M.items())
    d = float(distance_m)
    if d <= knots[0][0]:
        return float(knots[0][1])
    if d >= knots[-1][0]:
        return float(knots[-1][1])
    for (a, va), (b, vb) in zip(knots, knots[1:]):
        if a <= d <= b:
            if a == b:
                return float(vb)
            t = (d - a) / (b - a)
            return float(va + (vb - va) * t)
    return 5.70

def _benchmark_split_time(distance_m: float, split_step: int) -> float:
    std100 = _benchmark_std_100m(distance_m)
    step = int(split_step)
    if step == 100:
        return float(std100)
    if step == 200:
        return float(std100 * 2.0)
    return float(std100 * (step / 100.0))

def _robust_center(series: pd.Series, p_lo: float = 0.10, p_hi: float = 0.90) -> float:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return np.nan
    try:
        return float(winsorize(s, p_lo=p_lo, p_hi=p_hi).mean())
    except Exception:
        return float(s.median())

def _phase_avg_split(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.DataFrame:
    vals = df[cols].apply(pd.to_numeric, errors="coerce").replace(0, np.nan) if cols else pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_avg_split_time"] = vals.mean(axis=1, skipna=True) if cols else np.nan
    out[f"{prefix}_valid_splits"] = vals.notna().sum(axis=1) if cols else 0
    return out

def compute_rpss(metrics_df: pd.DataFrame, distance_m: float, split_step: int, seg_markers: list[int]):
    """
    RPSS = Race Pace Strength Score.
    Uses the tsSPI section only for the headline score, benchmarked to a distance standard.
    Also reports race-level Acceleration and Grind phase split averages to describe shape change.
    """
    if metrics_df is None or len(metrics_df) == 0:
        return None
    step = int(split_step)
    D = float(distance_m)
    tssp_start = _adaptive_tssp_start(D, step, seg_markers)
    mid_cols = [c for c in _make_range_cols(D, tssp_start, 600, step) if c in metrics_df.columns]
    if not mid_cols:
        return None

    # Pace-shape companion windows using the user's segment-time definitions.
    # 200m files: 400_Time = 600→400, 200_Time = 400→200, Finish_Time = 200→FIN
    # 100m files: 400_Time = 500→400, 300_Time = 400→300, 200_Time = 300→200,
    #             100_Time = 200→100, Finish_Time = 100→FIN
    if step == 100:
        accel_cols = [c for c in ["500_Time", "400_Time", "300_Time", "200_Time"] if c in metrics_df.columns]
        grind_cols = [c for c in ["100_Time", "Finish_Time"] if c in metrics_df.columns]
    else:
        accel_cols = [c for c in ["400_Time", "200_Time"] if c in metrics_df.columns]
        grind_cols = [c for c in ["Finish_Time"] if c in metrics_df.columns]

    df = metrics_df.copy()
    mid_stats = _phase_avg_split(df, mid_cols, "_RPSS")
    acc_stats = _phase_avg_split(df, accel_cols, "_ACC")
    grd_stats = _phase_avg_split(df, grind_cols, "_GRD")
    for part in [mid_stats, acc_stats, grd_stats]:
        for c in part.columns:
            df[c] = part[c]

    usable = df[(pd.to_numeric(df["_RPSS_avg_split_time"], errors="coerce").notna()) & (df["_RPSS_valid_splits"] > 0)].copy()
    if usable.empty:
        return None

    benchmark_std_100m = _benchmark_std_100m(D)
    benchmark_split_time = _benchmark_split_time(D, step)

    usable["_RPSS_vs_std_pct"] = 100.0 * (benchmark_split_time / pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce"))
    usable["_RPSS_margin"] = benchmark_split_time - pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce")

    finish_col = "Finish_Pos" if "Finish_Pos" in usable.columns else None
    if finish_col:
        usable["_FinishSort"] = pd.to_numeric(usable[finish_col], errors="coerce").fillna(1e9)
        usable = usable.sort_values(["_FinishSort", "Horse" if "Horse" in usable.columns else usable.columns[0]])
    top_n = max(1, int(np.ceil(len(usable) / 2.0)))
    top_half = usable.head(top_n).copy() if finish_col else usable.nsmallest(top_n, columns="_RPSS_avg_split_time")

    race_avg = _robust_center(usable["_RPSS_avg_split_time"])
    top_half_avg = _robust_center(top_half["_RPSS_avg_split_time"])
    rpss = float(100.0 * (benchmark_split_time / race_avg)) if np.isfinite(race_avg) and race_avg > 0 else np.nan
    top_half_rpss = float(100.0 * (benchmark_split_time / top_half_avg)) if np.isfinite(top_half_avg) and top_half_avg > 0 else np.nan
    beaters_pct = float((pd.to_numeric(usable["_RPSS_avg_split_time"], errors="coerce") <= benchmark_split_time).mean() * 100.0)
    top_half_beaters_pct = float((pd.to_numeric(top_half["_RPSS_avg_split_time"], errors="coerce") <= benchmark_split_time).mean() * 100.0)

    accel_avg = _robust_center(usable["_ACC_avg_split_time"])
    grind_avg = _robust_center(usable["_GRD_avg_split_time"])
    accel_vs_std = float(100.0 * (benchmark_split_time / accel_avg)) if np.isfinite(accel_avg) and accel_avg > 0 else np.nan
    grind_vs_std = float(100.0 * (benchmark_split_time / grind_avg)) if np.isfinite(grind_avg) and grind_avg > 0 else np.nan
    accel_lift_pct = float((race_avg / accel_avg) * 100.0) if np.isfinite(race_avg) and np.isfinite(accel_avg) and accel_avg > 0 else np.nan
    grind_hold_pct = float((accel_avg / grind_avg) * 100.0) if np.isfinite(accel_avg) and np.isfinite(grind_avg) and grind_avg > 0 else np.nan

    if not np.isfinite(rpss):
        verdict = "Unavailable"
    elif rpss < 97.0:
        verdict = "Weak sustained pace"
    elif rpss < 98.5:
        verdict = "Fair sustained pace"
    elif rpss < 99.5:
        verdict = "Slightly below strong standard"
    elif rpss <= 100.5:
        verdict = "Genuine / strong"
    elif rpss <= 102.0:
        verdict = "Very strong"
    else:
        verdict = "Exceptional"

    out_cols = [c for c in ["Horse", "Finish_Pos"] if c in usable.columns]
    out = usable[out_cols + ["_RPSS_avg_split_time", "_RPSS_vs_std_pct", "_RPSS_margin", "_RPSS_valid_splits"]].copy()
    out = out.rename(columns={
        "_RPSS_avg_split_time": "Avg tsSPI split time",
        "_RPSS_vs_std_pct": "tsSPI vs std (%)",
        "_RPSS_margin": "Margin vs std (s)",
        "_RPSS_valid_splits": "tsSPI splits used",
    })

    winner_row = None
    if "Finish_Pos" in usable.columns:
        _fp = pd.to_numeric(usable["Finish_Pos"], errors="coerce")
        _w = usable.loc[_fp == _fp.min()] if _fp.notna().any() else usable.iloc[:1]
        if len(_w):
            winner_row = _w.iloc[0]
    elif len(usable):
        winner_row = usable.iloc[0]

    def _phase_best_name_and_vals(df, avg_col):
        if avg_col not in df.columns or "Horse" not in df.columns:
            return None, np.nan, np.nan
        _vals = pd.to_numeric(df[avg_col], errors="coerce")
        _tmp = df.loc[_vals.notna()].copy()
        if _tmp.empty:
            return None, np.nan, np.nan
        _tmp[avg_col] = pd.to_numeric(_tmp[avg_col], errors="coerce")
        _best = _tmp.loc[_tmp[avg_col].idxmin()]
        _best_avg = float(_best[avg_col])
        _best_vs = float(100.0 * (benchmark_split_time / _best_avg)) if np.isfinite(_best_avg) and _best_avg > 0 else np.nan
        return _best.get("Horse"), _best_avg, _best_vs

    win_tsspi = float(winner_row["_RPSS_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_RPSS_avg_split_time")) else np.nan
    win_tsspi_vs = float(100.0 * (benchmark_split_time / win_tsspi)) if np.isfinite(win_tsspi) and win_tsspi > 0 else np.nan
    win_accel = float(winner_row["_ACC_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_ACC_avg_split_time")) else np.nan
    win_accel_vs = float(100.0 * (benchmark_split_time / win_accel)) if np.isfinite(win_accel) and win_accel > 0 else np.nan
    win_grind = float(winner_row["_GRD_avg_split_time"]) if winner_row is not None and pd.notna(winner_row.get("_GRD_avg_split_time")) else np.nan
    win_grind_vs = float(100.0 * (benchmark_split_time / win_grind)) if np.isfinite(win_grind) and win_grind > 0 else np.nan
    winner_name = winner_row.get("Horse") if winner_row is not None and "Horse" in winner_row.index else None

    best_tsspi_name, best_tsspi, best_tsspi_vs = _phase_best_name_and_vals(usable, "_RPSS_avg_split_time")
    best_acc_name, best_acc, best_acc_vs = _phase_best_name_and_vals(usable, "_ACC_avg_split_time")
    best_grd_name, best_grd, best_grd_vs = _phase_best_name_and_vals(usable, "_GRD_avg_split_time")

    phase_table = pd.DataFrame([
        {
            "Phase": "tsSPI",
            "Avg split (s)": race_avg,
            "Vs std (%)": rpss,
            "Change vs prior (%)": np.nan,
            "Winner": winner_name,
            "Winner split (s)": win_tsspi,
            "Winner vs std (%)": win_tsspi_vs,
            "Best horse": best_tsspi_name,
            "Best split (s)": best_tsspi,
            "Best vs std (%)": best_tsspi_vs,
        },
        {
            "Phase": "Acceleration",
            "Avg split (s)": accel_avg,
            "Vs std (%)": accel_vs_std,
            "Change vs prior (%)": accel_lift_pct,
            "Winner": winner_name,
            "Winner split (s)": win_accel,
            "Winner vs std (%)": win_accel_vs,
            "Best horse": best_acc_name,
            "Best split (s)": best_acc,
            "Best vs std (%)": best_acc_vs,
        },
        {
            "Phase": "Grind",
            "Avg split (s)": grind_avg,
            "Vs std (%)": grind_vs_std,
            "Change vs prior (%)": grind_hold_pct,
            "Winner": winner_name,
            "Winner split (s)": win_grind,
            "Winner vs std (%)": win_grind_vs,
            "Best horse": best_grd_name,
            "Best split (s)": best_grd,
            "Best vs std (%)": best_grd_vs,
        },
    ])

    return {
        "distance_m": D,
        "split_step": step,
        "mid_cols": mid_cols,
        "accel_cols": accel_cols,
        "grind_cols": grind_cols,
        "benchmark_std_100m": benchmark_std_100m,
        "benchmark_split_time": benchmark_split_time,
        "race_avg_split_time": race_avg,
        "top_half_avg_split_time": top_half_avg,
        "accel_avg_split_time": accel_avg,
        "grind_avg_split_time": grind_avg,
        "accel_vs_std": accel_vs_std,
        "grind_vs_std": grind_vs_std,
        "accel_lift_pct": accel_lift_pct,
        "grind_hold_pct": grind_hold_pct,
        "rpss": rpss,
        "top_half_rpss": top_half_rpss,
        "beaters_pct": beaters_pct,
        "top_half_beaters_pct": top_half_beaters_pct,
        "verdict": verdict,
        "phase_table": phase_table,
        "table": out.reset_index(drop=True),
    }

def render_rpss_section(rpss_info: dict | None):
    st.markdown("## Race Pace Strength Score (RPSS)")
    st.caption("Race-level sustained pressure score using the tsSPI section only, benchmarked to a standard split time for the trip.")
    if not rpss_info:
        st.info("RPSS could not be calculated for this race.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RPSS", f"{rpss_info['rpss']:.2f}" if np.isfinite(rpss_info['rpss']) else "-")
    c2.metric("Std split", f"{rpss_info['benchmark_split_time']:.2f}s")
    c3.metric("Race avg tsSPI", f"{rpss_info['race_avg_split_time']:.2f}s" if np.isfinite(rpss_info['race_avg_split_time']) else "-")
    c4.metric("Beaters", f"{rpss_info['beaters_pct']:.1f}%")

    st.markdown(f"**Verdict:** {rpss_info['verdict']}  ")
    _top_half_txt = f"{rpss_info['top_half_rpss']:.2f}" if np.isfinite(rpss_info['top_half_rpss']) else "-"
    st.caption(
        f"Benchmark: {rpss_info['benchmark_std_100m']:.2f}s per 100m • Top-half RPSS: {_top_half_txt}"
    )

    phase_df = rpss_info.get("phase_table")
    if phase_df is not None and len(phase_df):
        phase_df = phase_df.copy()
        for _col, _dp in {
            "Avg split (s)": 3,
            "Vs std (%)": 2,
            "Change vs prior (%)": 2,
            "Winner split (s)": 3,
            "Winner vs std (%)": 2,
            "Best split (s)": 3,
            "Best vs std (%)": 2,
        }.items():
            if _col in phase_df.columns:
                phase_df[_col] = pd.to_numeric(phase_df[_col], errors="coerce").round(_dp)
        st.dataframe(phase_df, use_container_width=True, hide_index=True)

def _pri_robust_z(series: pd.Series) -> pd.Series:
    """Race-relative robust z-score with a standard-deviation fallback."""
    s = pd.to_numeric(series, errors="coerce")
    med = float(np.nanmedian(s)) if s.notna().any() else np.nan
    mad = float(np.nanmedian(np.abs(s - med))) if np.isfinite(med) else np.nan
    if np.isfinite(mad) and mad > 1e-9:
        z = (s - med) / (1.4826 * mad)
    else:
        std = float(np.nanstd(s, ddof=0))
        mean = float(np.nanmean(s)) if s.notna().any() else np.nan
        z = (s - mean) / std if np.isfinite(std) and std > 1e-9 else pd.Series(0.0, index=s.index)
    return pd.to_numeric(z, errors="coerce").clip(-3.5, 3.5)

def _phase_speed_from_raw(row: pd.Series, distance_m: float, phase_low: float, phase_high: float):
    """
    Compute speed inside a phase expressed as distance remaining.
    Segment times are apportioned by overlap, which supports 100m, 200m,
    and odd opening segments without changing the source interpretation.
    """
    D = float(distance_m)
    low = max(0.0, float(phase_low))
    high = min(D, float(phase_high))
    if high <= low:
        return np.nan, 0.0

    markers = []
    for c in row.index:
        if isinstance(c, str) and c.endswith("_Time") and c != "Finish_Time":
            try:
                markers.append(int(c.split("_")[0]))
            except Exception:
                pass
    markers = sorted({m for m in markers if 0 < m < D}, reverse=True)

    segments = []
    prev = D
    for marker in markers:
        col = f"{marker}_Time"
        if marker >= prev:
            continue
        segments.append((float(prev), float(marker), col))
        prev = float(marker)
    if "Finish_Time" in row.index and prev > 0:
        segments.append((float(prev), 0.0, "Finish_Time"))

    total_dist = 0.0
    total_time = 0.0
    for seg_high, seg_low, col in segments:
        overlap = max(0.0, min(seg_high, high) - max(seg_low, low))
        if overlap <= 0:
            continue
        t = pd.to_numeric(row.get(col), errors="coerce")
        seg_dist = seg_high - seg_low
        if pd.isna(t) or float(t) <= 0 or seg_dist <= 0:
            continue
        total_dist += overlap
        total_time += float(t) * (overlap / seg_dist)

    if total_dist <= 0 or total_time <= 0:
        return np.nan, total_dist
    return total_dist / total_time, total_dist

def build_pri_table(raw_df: pd.DataFrame, metrics_df: pd.DataFrame, distance_m: float) -> pd.DataFrame:
    """
    Pressure phase: after the opening 200m until 400m from home.
    Retention phase: final 400m.
    PRI is independent of PI. Positive pressure only earns full credit when
    the horse retains that effort through the final 400m.
    """
    D = float(distance_m)
    out = pd.DataFrame(index=raw_df.index)
    out["Horse"] = raw_df["Horse"].astype(str) if "Horse" in raw_df.columns else metrics_df["Horse"].astype(str)
    if "Finish_Pos" in metrics_df.columns:
        out["Finish_Pos"] = pd.to_numeric(metrics_df["Finish_Pos"], errors="coerce").to_numpy()
    if "PI" in metrics_df.columns:
        out["PI"] = pd.to_numeric(metrics_df["PI"], errors="coerce").to_numpy()

    pressure_low = 400.0
    pressure_high = max(pressure_low, D - 200.0)
    middle_speeds, middle_coverage = [], []
    late_speeds, late_coverage = [], []

    for _, row in raw_df.iterrows():
        ms, mc = _phase_speed_from_raw(row, D, pressure_low, pressure_high)
        ls, lc = _phase_speed_from_raw(row, D, 0.0, min(400.0, D))
        middle_speeds.append(ms); middle_coverage.append(mc)
        late_speeds.append(ls); late_coverage.append(lc)

    out["Pressure_Speed"] = pd.to_numeric(pd.Series(middle_speeds, index=out.index), errors="coerce")
    out["Late_Speed"] = pd.to_numeric(pd.Series(late_speeds, index=out.index), errors="coerce")
    out["Pressure_Coverage_m"] = middle_coverage
    out["Late_Coverage_m"] = late_coverage

    field_middle = float(np.nanmedian(out["Pressure_Speed"])) if out["Pressure_Speed"].notna().any() else np.nan
    out["Pressure_Delta_pct"] = np.where(
        np.isfinite(field_middle) & (field_middle > 0),
        100.0 * (out["Pressure_Speed"] / field_middle - 1.0),
        np.nan,
    )
    out["Retention_pct"] = np.where(
        (out["Pressure_Speed"] > 0) & out["Late_Speed"].notna(),
        100.0 * out["Late_Speed"] / out["Pressure_Speed"],
        np.nan,
    )

    out["Pressure_z"] = _pri_robust_z(out["Pressure_Delta_pct"])
    out["Retention_z"] = _pri_robust_z(out["Retention_pct"])

    # Retained-pressure model:
    # - Only above-field pressure creates positive pressure credit.
    # - That credit is gated by how much speed the horse retained late.
    # - 90% retention gives no retained-pressure credit; 100% gives full credit.
    #   Values outside that band are clipped for stability.
    out["Positive_Pressure_z"] = pd.to_numeric(out["Pressure_z"], errors="coerce").clip(lower=0.0)
    out["Retention_Gate"] = (
        (pd.to_numeric(out["Retention_pct"], errors="coerce") - 90.0) / 10.0
    ).clip(lower=0.0, upper=1.0)
    out["Retained_Pressure"] = out["Positive_Pressure_z"] * out["Retention_Gate"]

    # Pressure gate prevents low-pressure closers from topping PRI purely on late retention.
    # - Clearly negative pressure receives little or no access to a high PRI.
    # - Neutral pressure receives partial access.
    # - Strong positive pressure receives full access.
    out["Pressure_Gate"] = (
        (pd.to_numeric(out["Pressure_Delta_pct"], errors="coerce") + 0.25) / 1.0
    ).clip(lower=0.0, upper=1.0)

    out["PRI_Core"] = (
        0.75 * out["Retained_Pressure"]
        + 0.25 * out["Retention_z"] * out["Pressure_Gate"]
    )
    out["PRI"] = (5.0 + 2.5 * np.tanh(out["PRI_Core"] / 1.35)).clip(0.0, 10.0)

    retention_median = float(np.nanmedian(out["Retention_pct"])) if out["Retention_pct"].notna().any() else np.nan
    def _pri_profile(row):
        p = row.get("Pressure_Delta_pct")
        r = row.get("Retention_pct")
        if not (pd.notna(p) and pd.notna(r) and np.isfinite(retention_median)):
            return "Insufficient data"
        high_p = float(p) >= 0.0
        high_r = float(r) >= retention_median
        if high_p and high_r:
            return "Pressure resistant"
        if high_p and not high_r:
            return "Brave but faded"
        if not high_p and high_r:
            return "Pace-assisted closer"
        return "Low-pressure performer"

    out["Profile"] = out.apply(_pri_profile, axis=1)
    out["PRI_Rank"] = out["PRI"].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    out.attrs["pressure_phase"] = f"after first 200m to 400m remaining"
    out.attrs["retention_phase"] = "final 400m"
    out.attrs["field_middle_speed"] = field_middle
    out.attrs["retention_median"] = retention_median
    return out

def compute_race_test_profile(metrics_df: pd.DataFrame, rpss_info=None, grind_col: str = "Grind") -> dict:
    """Interpret what the fitted Race Plane tested without changing any ratings.

    The profile uses the existing regression relationship:
        Grind = intercept + b(tsSPI) + c(Accel)
    Positive coefficient shares describe positive reward only. Negative
    coefficients are retained as inverse relationships rather than being turned
    into misleading positive percentages.
    """
    out = {
        "valid": False,
        "label": "Inconclusive race test",
        "mode": "Inconclusive",
        "confidence": "Low",
        "summary": "The race did not produce a sufficiently stable relationship to identify one clear performance test.",
        "r2": np.nan,
        "intercept": np.nan,
        "travel_coef": np.nan,
        "accel_coef": np.nan,
        "travel_reward_share": np.nan,
        "accel_reward_share": np.nan,
        "rpss": np.nan,
        "tempo": "Unknown",
        "runners": 0,
        "rank": 0,
    }
    if metrics_df is None or metrics_df.empty:
        return out
    if grind_col not in metrics_df.columns:
        grind_col = "Grind" if "Grind" in metrics_df.columns else grind_col
    required = ["tsSPI", "Accel", grind_col]
    if any(c not in metrics_df.columns for c in required):
        return out

    d = metrics_df[required].copy()
    for c in required:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    out["runners"] = int(len(d))
    if len(d) < 4:
        out["summary"] = "Fewer than four complete runners were available, so the Race Plane test is not considered reliable."
        return out

    x = (d["tsSPI"] - 100.0).to_numpy(dtype=float)
    y = (d["Accel"] - 100.0).to_numpy(dtype=float)
    z = (d[grind_col] - 100.0).to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d)), x, y])
    coef, _, rank, _ = np.linalg.lstsq(X, z, rcond=None)
    intercept, b_tsspi, c_accel = [float(v) for v in coef]
    expected = X @ coef
    ss_res = float(np.nansum((z - expected) ** 2))
    ss_tot = float(np.nansum((z - float(np.nanmean(z))) ** 2))
    r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

    rpss = np.nan
    if isinstance(rpss_info, dict):
        rpss = pd.to_numeric(rpss_info.get("rpss"), errors="coerce")
    if np.isfinite(rpss):
        tempo = "Slow" if float(rpss) < 94.0 else "Even" if float(rpss) < 98.0 else "Fast"
    else:
        tempo = "Unknown"

    pos_travel = max(b_tsspi, 0.0)
    pos_accel = max(c_accel, 0.0)
    positive_total = pos_travel + pos_accel
    if positive_total > 1e-12:
        travel_share = pos_travel / positive_total
        accel_share = pos_accel / positive_total
    else:
        travel_share = np.nan
        accel_share = np.nan

    if not np.isfinite(r2) or r2 < 0.30 or rank < 3 or positive_total <= 1e-12:
        mode = "Inconclusive"
    elif accel_share >= 0.65:
        mode = "Acceleration"
    elif travel_share >= 0.65:
        mode = "Sustained Pressure"
    else:
        mode = "Balanced"

    if not np.isfinite(r2) or r2 < 0.30 or rank < 3:
        confidence = "Low"
    elif r2 < 0.50:
        confidence = "Tentative"
    elif r2 < 0.70:
        confidence = "Meaningful"
    else:
        confidence = "Strong"

    if mode == "Inconclusive":
        label = "Inconclusive race test"
        summary = (
            "The fitted plane did not explain enough of the field to identify one dependable race requirement. "
            "Horse-level residuals can still be useful, but the slope should not drive a strong race-shape conclusion."
        )
    elif mode == "Acceleration":
        label = "Tactical sprint test" if tempo == "Slow" else "Acceleration-led test"
        summary = (
            f"This {tempo.lower() if tempo != 'Unknown' else 'race'} contest linked finishing strength more closely to acceleration than sustained travelling speed. "
            "A decisive change of speed was the clearest positive requirement produced by the field."
        )
    elif mode == "Sustained Pressure":
        label = "Sustained-pressure test"
        summary = (
            f"This {tempo.lower() if tempo != 'Unknown' else 'race'} contest linked finishing strength most strongly to sustained travelling speed. "
            "Horses able to carry pressure before the finish were positively rewarded."
        )
    else:
        label = "Balanced all-round test"
        summary = (
            f"This {tempo.lower() if tempo != 'Unknown' else 'race'} contest required a combination of sustained travelling speed and acceleration. "
            "No single earlier phase dominated the positive relationship with finishing strength."
        )

    out.update({
        "valid": True,
        "label": label,
        "mode": mode,
        "confidence": confidence,
        "summary": summary,
        "r2": r2,
        "intercept": intercept,
        "travel_coef": b_tsspi,
        "accel_coef": c_accel,
        "travel_reward_share": travel_share,
        "accel_reward_share": accel_share,
        "rpss": rpss,
        "tempo": tempo,
        "runners": int(len(d)),
        "rank": int(rank),
    })
    return out

def compute_race_shape_verdict(metrics_df: pd.DataFrame, rpss_info=None, race_test_profile=None) -> pd.DataFrame:
    """Create RPSS-aware, time-only narrative verdicts for each runner.

    The module interprets the existing Race Edge phase indices within the shared
    Race Test Profile. It does not alter PI and deliberately ignores position,
    draw and positional gains.
    """
    if metrics_df is None or metrics_df.empty or "Horse" not in metrics_df.columns:
        return pd.DataFrame()

    df = metrics_df.copy()
    grind_col = "Grind_CG" if "Grind_CG" in df.columns else "Grind"
    required = ["tsSPI", "Accel", grind_col]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()

    for c in required + ["PI", "Grind"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def _rz(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        med = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
        mad = float(np.nanmedian(np.abs(x - med))) if np.isfinite(x).any() else 0.0
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 0.20:
            scale = float(np.nanstd(x, ddof=0))
        if not np.isfinite(scale) or scale < 0.20:
            scale = 1.0
        return ((x - med) / scale).clip(-3.0, 3.0)

    z_ts = _rz(df["tsSPI"])
    z_ac = _rz(df["Accel"])
    z_gr = _rz(df[grind_col])
    z_pi = _rz(df["PI"]) if "PI" in df.columns else pd.Series(0.0, index=df.index)

    rpss = np.nan
    rpss_label = "Unknown"
    if isinstance(rpss_info, dict):
        rpss = pd.to_numeric(rpss_info.get("rpss"), errors="coerce")
        rpss_label = str(rpss_info.get("verdict") or "Unknown")
    if np.isfinite(rpss):
        if float(rpss) < 94.0:
            context = "Slow"
        elif float(rpss) < 98.0:
            context = "Even"
        else:
            context = "Fast"
    else:
        context = "Unknown"

    if not isinstance(race_test_profile, dict):
        race_test_profile = compute_race_test_profile(df, rpss_info, "Grind")
    test_mode = str(race_test_profile.get("mode", "Inconclusive"))
    test_label = str(race_test_profile.get("label", "Inconclusive race test"))
    test_confidence = str(race_test_profile.get("confidence", "Low"))

    rows = []
    for i, r in df.iterrows():
        horse = str(r.get("Horse", "Unknown"))
        ts, ac, gr, pi = [float(v) if np.isfinite(v) else 0.0 for v in (z_ts.loc[i], z_ac.loc[i], z_gr.loc[i], z_pi.loc[i])]
        raw_ts = pd.to_numeric(r.get("tsSPI"), errors="coerce")
        raw_ac = pd.to_numeric(r.get("Accel"), errors="coerce")
        raw_gr = pd.to_numeric(r.get(grind_col), errors="coerce")
        raw_pi = pd.to_numeric(r.get("PI"), errors="coerce")

        balanced = max(ts, ac, gr) - min(ts, ac, gr) <= 0.85
        complete = ts >= 0.55 and ac >= 0.55 and gr >= 0.55
        cruise_led = ts >= 0.65 and ts >= ac + 0.45
        burst_led = ac >= 0.75 and ac >= ts + 0.55
        controlled_finish = gr > -0.85
        collapse = gr <= -1.20
        strong_quality = pi >= 0.20 or max(ts, ac, gr) >= 0.80

        if context == "Slow":
            if strong_quality and cruise_led and controlled_finish:
                label = "Hidden Improver"
                verdict = (
                    "This slow-run race placed a premium on a sharp late sprint, which did not fully suit this horse's profile. "
                    "Its strongest evidence came through sustained travelling speed rather than a short burst, while the finish weakened without becoming a complete collapse. "
                    "A more genuinely run race should allow that cruising strength to play a greater role, and additional distance may also suit."
                )
                action = "Follow in a truer-run race or over further."
                confidence = "High" if ts >= 1.0 and controlled_finish and pi >= 0 else "Moderate"
            elif burst_led and gr >= -0.35:
                label = "Tactical Specialist"
                verdict = (
                    "This tactical race played directly to the horse's sharp acceleration. The strongest part of the performance was concentrated in the sprint phase after energy had been conserved earlier. "
                    "The run deserves respect, but it offers less proof that the same level will be reproduced when pressure is sustained from further out."
                )
                action = "Best suited by another tactical setup; seek confirmation in a true-run race."
                confidence = "High" if ac >= 1.15 else "Moderate"
            elif burst_led and collapse:
                label = "Short-Burst Profile"
                verdict = (
                    "The horse produced a notable acceleration in this slow-run contest but could not sustain it through the finish. "
                    "That points to short-speed ability rather than clear evidence that a stronger tempo or extra distance will help."
                )
                action = "Prefer a tactical race at the same or shorter trip."
                confidence = "Moderate"
            elif complete or (balanced and pi >= 0.35):
                label = "Race-Shape Versatile"
                verdict = (
                    "Although the race was tactical, this horse produced a well-distributed performance across travelling speed, acceleration and finishing strength. "
                    "The evidence is not dependent on one isolated phase, suggesting the performance can transfer to a wider range of race shapes."
                )
                action = "Respect under both tactical and more genuine conditions."
                confidence = "High" if complete else "Moderate"
            elif not strong_quality and collapse:
                label = "Limited Evidence"
                verdict = (
                    "The tactical race did not produce enough sustained time evidence to support a positive projection. "
                    "The horse was unable to convert the conserved-energy setup into a competitive late performance."
                )
                action = "No upgrade from this run."
                confidence = "High"
            else:
                label = "Shape Inconclusive"
                verdict = (
                    "This slow-run race did not provide a complete test of the horse's ability, but the sectional profile is mixed rather than clearly positive or tactically dependent. "
                    "More evidence is required before projecting a meaningful improvement or regression under a different tempo."
                )
                action = "Treat the run cautiously until tested in a clearer race shape."
                confidence = "Moderate"

        elif context == "Fast":
            if complete and pi >= 0.45:
                label = "Elite Confirmation"
                verdict = (
                    "This genuinely run race provided a full test of sustained ability, and the horse performed strongly through every major phase. "
                    "It travelled, accelerated and maintained its effort under pressure, confirming that the performance was not created by a favourable tactical setup."
                )
                action = "Treat as fully proven under sustained pressure."
                confidence = "High"
            elif strong_quality and gr >= 0.35 and ts >= 0.0:
                label = "Genuine Performer"
                verdict = (
                    "The strong race tempo tested the horse's ability to absorb sustained pressure, and its sectional profile held together well. "
                    "This is credible, repeatable evidence rather than a performance dependent on a short sprint."
                )
                action = "Expect the form to remain reliable in another true-run race."
                confidence = "High" if pi >= 0.5 else "Moderate"
            elif burst_led and gr < -0.55:
                label = "Pressure Vulnerable"
                verdict = (
                    "The horse showed acceleration but could not maintain that effort once the genuine tempo took full effect. "
                    "The profile suggests it may be more effective when the race develops into a shorter tactical sprint."
                )
                action = "Prefer an easier tempo or shorter pressure phase."
                confidence = "Moderate"
            elif collapse:
                label = "True-Run Exposed"
                verdict = (
                    "The genuine tempo exposed a clear weakness in sustaining speed through the finish. "
                    "This run provides little support for stepping up in distance unless the horse can distribute its effort more efficiently."
                )
                action = "Be cautious over further or in another strongly run race."
                confidence = "High"
            else:
                label = "Credible Under Pressure"
                verdict = (
                    "This race provided a genuine examination, and the horse's performance was broadly supported by the sectional times. "
                    "It was not dominant across every phase, but the run carries more weight than form achieved in a tactical contest."
                )
                action = "Use as reliable evidence under similar conditions."
                confidence = "Moderate"

        else:  # Even or RPSS unavailable
            if complete or (balanced and pi >= 0.45):
                label = "Versatile Confirmation"
                verdict = (
                    "The race provided a balanced test, and this horse performed consistently across the major phases. "
                    "Its ability was not concentrated in one short section, supporting a reliable and adaptable performance profile."
                )
                action = "Respect across a range of normal race shapes."
                confidence = "High" if complete else "Moderate"
            elif burst_led:
                label = "Acceleration-Led"
                verdict = (
                    "The performance was driven primarily by acceleration rather than evenly sustained strength. "
                    "That sharp speed is a genuine asset, although a stronger tempo may place greater pressure on the finishing phase."
                )
                action = "Most appealing when a decisive turn of foot is likely to matter."
                confidence = "Moderate"
            elif cruise_led and controlled_finish and strong_quality:
                label = "Further Potential"
                verdict = (
                    "The horse's strongest evidence came through sustained travelling speed, with enough finishing integrity to suggest the effort was not exhausted. "
                    "A slightly stronger tempo or additional distance may allow this profile to become more effective."
                )
                action = "Consider over further or when the pace is more sustained."
                confidence = "Moderate"
            elif collapse:
                label = "Finish Concern"
                verdict = (
                    "The horse was unable to sustain its effort through the final phase of this balanced contest. "
                    "That weakens the case for extra distance and raises a question about finishing durability."
                )
                action = "Prefer the same or shorter trip until finishing strength improves."
                confidence = "High"
            else:
                label = "Honest Run"
                verdict = (
                    "The sectional profile broadly matches the balanced nature of the race. "
                    "There is no strong evidence that the horse was either substantially helped or hindered by the tempo."
                )
                action = "No major race-shape upgrade or downgrade."
                confidence = "Moderate"

        # Interpret the horse relative to what the Race Plane says the race tested.
        if test_mode == "Acceleration":
            if cruise_led and controlled_finish:
                verdict = (
                    f"The Race Plane identifies this as an {test_label.lower()}, which placed greater positive emphasis on acceleration than sustained travelling speed. "
                    "This horse's stronger evidence came through travelling strength, so the race did not fully test its preferred profile. " + verdict
                )
                action = "Upgrade in a truer sustained-pressure race; " + action[:1].lower() + action[1:]
            elif burst_led:
                verdict = (
                    f"The Race Plane identifies this as an {test_label.lower()}, and this horse's acceleration-led profile matched that requirement. " + verdict
                )
                action = "The setup suited; seek confirmation when the test changes. " + action
        elif test_mode == "Sustained Pressure":
            if cruise_led or (ts >= 0.35 and controlled_finish):
                verdict = (
                    f"The Race Plane identifies this as a {test_label.lower()}, and this horse's travelling strength was aligned with the main positive requirement. " + verdict
                )
                action = "Treat the performance as supported by the race test. " + action
            elif burst_led:
                verdict = (
                    f"The Race Plane identifies this as a {test_label.lower()}, while this horse relied more heavily on acceleration. "
                    "Its strongest attribute was not the principal quality rewarded by the race. " + verdict
                )
                action = "Consider a more tactical setup; " + action[:1].lower() + action[1:]
        elif test_mode == "Balanced":
            if balanced or complete:
                verdict = (
                    f"The Race Plane identifies this as a {test_label.lower()}, and this horse produced an appropriately balanced sectional profile. " + verdict
                )
            elif burst_led or cruise_led:
                verdict = (
                    f"The Race Plane identifies this as a {test_label.lower()}, but this horse's performance was concentrated more heavily in one phase. " + verdict
                )
        else:
            verdict = (
                f"The Race Test Profile is inconclusive ({test_confidence.lower()} confidence), so the horse-level phase evidence carries more weight than the plane slope. " + verdict
            )

        # Race files used by Race Edge normally store the result as Finish_Pos,
        # while some older imports used Finish Position/Finish. Resolve all known
        # formats so the verdict table never loses finishing-position context.
        finish_value = np.nan
        for finish_col in ("Finish_Pos", "Finish Position", "Finish", "Position", "Pos"):
            if finish_col in df.columns:
                candidate = pd.to_numeric(r.get(finish_col), errors="coerce")
                if pd.notna(candidate):
                    finish_value = candidate
                    break

        rows.append({
            "Horse": horse,
            "Finish": finish_value,
            "PI": raw_pi,
            "Verdict": label,
            "Confidence": confidence,
            "Narrative": verdict,
            "Action": action,
            "tsSPI": raw_ts,
            "Accel": raw_ac,
            "Grind": raw_gr,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["Finish"] = pd.to_numeric(out["Finish"], errors="coerce").astype("Int64")
    for c in ["PI", "tsSPI", "Accel", "Grind"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out.attrs["rpss"] = rpss
    out.attrs["rpss_label"] = rpss_label
    out.attrs["context"] = context
    out.attrs["race_test_profile"] = race_test_profile
    return out.sort_values(["PI", "Finish"], ascending=[False, True], na_position="last").reset_index(drop=True)
