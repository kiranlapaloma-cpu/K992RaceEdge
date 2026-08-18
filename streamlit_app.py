# ======================= Batch 1 — Core + UI + I/O + DB bootstrap =======================
import io, math, re, os, hashlib, json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from metrics_engine import (
    build_metrics_and_shape,
    pi_weights_distance_and_context,
    _collect_markers,
)


# ======================= Modular Race Edge components =======================
from common import canon_horse
from wfa import _WFA_BAND_LABELS, get_wfa_lb, wfa_distance_band
from database import (
    _supabase_configured, get_supabase_client, _db_num, _db_round_mr,
    database_sustain_verdict, load_saved_horses, load_horse_history,
    save_horse_runs, load_rating_improver_runs, render_rating_improvers,
    render_horse_search, render_horse_compare, build_database_plane,
    build_database_phase_notes, build_database_handicap,
)
from race_card import render_race_card
from data_tools import (
    to_num, _first_present_value, _parse_race_date_value, _normalise_going_for_pi,
    _uploaded_file_preview, _horse_metadata_frame, normalize_headers, detect_step,
    normalize_200m_columns, expected_segments_from_df, integrity_scan,
)
from performance import (
    mad_std, winsorize, color_cycle, safe_num, sanitize_jsonable, _safe_bal_norm,
    compute_rpss, render_rpss_section, build_pri_table,
    compute_race_test_profile, compute_race_shape_verdict,
)
from metrics_engine import build_metrics_and_shape, pi_weights_distance_and_context

from view_core_metrics import render_core_metrics
from view_pace_curve import render_pace_curve
from view_ability_radar import render_ability_radar
from view_pressure_retention import render_pressure_retention
from view_race_plane import render_race_plane
from view_advanced_models import render_advanced_models
from view_save_race import render_save_race
from view_race_card import render_race_card_view
from view_horse_database import render_horse_database_view

# Streamlit JSON/NaN safety layer and plot label helper moved to ui_utils.py
from ui_utils import _sanitize, label_points_neatly

# ----------------------- Page config -----------------------
st.set_page_config(
    page_title="Race Edge — PI v3.2 + Hidden v2 + Ability v2 + CG + Race Shape + DB",
    layout="wide"
)

# ----------------------- Globals ---------------------------
APP_VERSION = "3.7"

# ----------------------- Small helpers ---------------------
def as_num(x):
    return pd.to_numeric(x, errors="coerce")

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))



def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()





# ----------------------- Supabase Horse Database -----------------------


































# ----------------------- Race Card -----------------------




















# ---------- Safety helpers (drop-in) ----------



from matplotlib.colors import TwoSlopeNorm


def _view_is(*names: str) -> bool:
    try:
        return APP_VIEW in names
    except Exception:
        return False



# ----------------------- RPSS helpers -----------------------












    # Runner-level tsSPI detail table intentionally removed to keep the app lighter and faster.


# ----------------------- CSV metadata helpers ---------------------------









# ----------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown(f"### Race Edge v{APP_VERSION}")
    st.caption("Focused sectional analysis")

    APP_VIEW = st.radio(
        "App View",
        [
            "Core Metrics",
            "Pressure Retention",
            "Pace Curve",
            "Ability Radar",
            "Race Plane Analysis",
            "Advanced Models",
            "Save Race to Database",
            "Race Card",
            "Horse Database",
        ],
        index=0,
    )

    if APP_VIEW not in ("Race Card", "Horse Database"):
        st.markdown("#### Race Setup")
        up = st.file_uploader(
            "Upload CSV/XLSX with 100 m or 200 m splits",
            type=["csv", "xlsx", "xls"]
        )
        _upload_preview = _uploaded_file_preview(up)
        _csv_distance = _first_present_value(_upload_preview, ["Distance", "Race Distance", "Distance (m)"], 1600)
        try:
            _distance_default = int(round(float(_csv_distance)))
        except Exception:
            _distance_default = 1600
        _distance_default = int(np.clip(_distance_default, 800, 4000))
        race_distance_input = st.number_input(
            "Race Distance (m)", min_value=800, max_value=4000, step=50,
            value=_distance_default,
            help="Auto-filled from the CSV when a Distance column is present; otherwise enter it manually.",
        )

        with st.expander("Advanced settings", expanded=False):
            USE_CG = st.toggle("Use Corrected Grind (CG)", value=True, help="Adjust Grind when the field finish collapses; preserves finisher credit.")
            DAMPEN_CG = st.toggle("Dampen Grind weight if collapsed", value=True, help="Shift a little weight Grind→Accel/tsSPI on collapse races.")
            # Race-shape calculations remain enabled internally for advanced models,
            # but their controls and summary outputs are intentionally hidden.
            USE_RACE_SHAPE = True
            USE_GOING_ADJUST = st.toggle(
                "Use Going Adjustment",
                value=True,
                help="Adjust PI weighting based on track going (Firm/Good/Soft/Heavy)"
            )
            _csv_going = _first_present_value(_upload_preview, ["Going", "Track Going", "Condition"], "Good")
            _going_default = _normalise_going_for_pi(_csv_going)
            _going_options = ["Good", "Firm", "Soft", "Heavy"]
            GOING_TYPE = st.selectbox(
                "Track Going",
                options=_going_options,
                index=_going_options.index(_going_default),
                help="Auto-filled from the CSV when Going is present. Only affects PI weights; sectional indices stay unchanged."
            ) if USE_GOING_ADJUST else "Good"
            WIND_AFFECTED = st.toggle("Wind affected race?", value=False, help="Purely informational (disclaimer only).")
            WIND_TAG = st.selectbox(
                "Wind note",
                options=["Headwind", "Tailwind", "Crosswind", "Negligible"],
                index=0,
                disabled=not WIND_AFFECTED,
            ) if WIND_AFFECTED else "None"
            SHOW_WARNINGS = st.toggle("Show data warnings", value=True)
            DEBUG = st.toggle("Debug info", value=False)
    else:
        # Database/reference views do not need a sectional upload.
        up = None
        _upload_preview = pd.DataFrame()
        race_distance_input = 1600
        USE_CG = True
        DAMPEN_CG = True
        USE_RACE_SHAPE = True
        USE_GOING_ADJUST = True
        GOING_TYPE = "Good"
        WIND_AFFECTED = False
        WIND_TAG = "None"
        SHOW_WARNINGS = True
        DEBUG = False

# ----------------------- Horse Database works without a race upload -------
if not up:
    if _view_is("Race Card"):
        render_race_card()
    elif _view_is("Horse Database"):
        st.title("Horse Database")
        st.caption("Research saved horse histories or compare multiple horses. No race file is required.")
        search_tab, improver_tab, compare_tab = st.tabs(["Horse Search", "Rating Improvers", "Compare Horses"])
        with search_tab:
            render_horse_search()
        with improver_tab:
            render_rating_improvers()
        with compare_tab:
            render_horse_compare()
    elif _view_is("Save Race to Database"):
        st.title("Save Race to Database")
        st.info("Upload a sectional file in the sidebar to calculate ratings and save the race.")
    else:
        st.info("Upload a sectional file to begin the Race Edge analysis.")
    st.stop()

# ----------------------- Header normalization / Aliases -------------------

# ----------------------- Split-step detection -----------------------------


# ----------------------- File load & preview ------------------------------
try:
    raw = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
    work, alias_notes = normalize_headers(raw.copy())
    st.success("File loaded.")
except Exception as e:
    st.error("Failed to read file.")
    st.exception(e)
    st.stop()

split_step = detect_step(work)
st.markdown(f"**Detected split step:** {split_step} m")
if alias_notes and SHOW_WARNINGS:
    st.info("Header aliases applied: " + "; ".join(alias_notes))

# ----------------------- Integrity helpers (odds-aware) -------------------



# Quick integrity line (display-only; full use comes after metrics)
integrity_text, _miss, _bad = integrity_scan(work, race_distance_input, split_step)
st.caption(f"Integrity: {integrity_text or 'OK'}")
if split_step == 200:
    st.caption("Finish column assumed to be the 200→0 segment (`Finish_Time`).")

# -------------------------------------------------------------------------
# Hand-off: Batch 2 will compute metrics and the Race Shape module (SED/FRA/SCI)
# ======================= Batch 2 — Metrics Engine + Race Shape (SED/SCI/FRA) =======================
# (Self-contained drop-in)

import math
import numpy as np
import pandas as pd

# ---- tiny helpers we rely on from Batch 1 (already defined there) ----
# as_num(x), clamp(v,lo,hi), mad_std(x) must exist above from Batch 1.

# -------- Stage helpers (works for 100m and 200m data) --------

# Core metrics engine moved to metrics_engine.py


# ---- Compute metrics + race shape now ----
try:
    metrics, seg_markers = build_metrics_and_shape(
        work,
        float(race_distance_input),
        int(split_step),
        USE_CG,
        DAMPEN_CG,
        USE_RACE_SHAPE,
        DEBUG,
        going_type=GOING_TYPE
    )
except Exception as e:
    st.error("Metric computation failed.")
    st.exception(e)
    st.stop()

# ======================= Pressure Retention Index (PRI) =======================



PRI_TABLE = build_pri_table(work.reset_index(drop=True), metrics.reset_index(drop=True), float(race_distance_input))

# ----------------------- RPSS (race-level tsSPI benchmark strength) -----------------------
RPSS_INFO = compute_rpss(metrics, float(race_distance_input), int(split_step), seg_markers)
if RPSS_INFO:
    metrics.attrs["RPSS"] = RPSS_INFO.get("rpss")
    metrics.attrs["RPSS_VERDICT"] = RPSS_INFO.get("verdict")
    metrics.attrs["RPSS_BEATERS_PCT"] = RPSS_INFO.get("beaters_pct")
    metrics.attrs["RPSS_STD_SPLIT"] = RPSS_INFO.get("benchmark_split_time")
    metrics.attrs["RPSS_RACE_AVG_SPLIT"] = RPSS_INFO.get("race_avg_split_time")

# ======================= Data Integrity & Header (post compute) ==========================
def _expected_segments(df: pd.DataFrame) -> list[str]:
    """Return the real sectional columns supplied by the file.

    Distance-derived names are unreliable for odd-distance 200m races such as
    1160m, where the first panel can be 160m and the remaining markers stay on
    the normal 200m grid.
    """
    marks = _collect_markers(df)
    cols = [f"{int(m)}_Time" for m in marks if f"{int(m)}_Time" in df.columns]
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def _integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    exp_cols = _expected_segments(df)
    missing = [c for c in exp_cols if c not in df.columns]
    invalid_counts = {}
    for c in exp_cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    if missing: msgs.append("Missing: " + ", ".join(missing))
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    if bads: msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs), missing, invalid_counts

integrity_text, missing_cols, invalid_counts = _integrity_scan(work, race_distance_input, split_step)

# ======================= Clean race header =======================
_hdr = (
    f"## Race Distance: **{int(race_distance_input)}m**  |  "
    f"Split step: **{split_step}m**"
)

metrics.attrs["WIND_AFFECTED"] = bool(WIND_AFFECTED)
metrics.attrs["WIND_TAG"] = str(WIND_TAG)

st.markdown(_hdr)

if SHOW_WARNINGS and (missing_cols or any(v>0 for v in invalid_counts.values())):
    bads = [f"{k} ({v} rows)" for k,v in invalid_counts.items() if v > 0]
    warn = []
    if missing_cols: warn.append("Missing: " + ", ".join(missing_cols))
    if bads: warn.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    if warn: st.markdown(f"*(⚠ {' • '.join(warn)})*")
if split_step == 200:
    st.caption("First panel & F-window adapt to odd 200m distances (e.g., 1160→F160, 1450→F250, 1100→F100). Finish is the 200→0 split.")

render_core_metrics(globals())


render_pace_curve(globals())
    # ======================= /Pace Curve (enhanced detailed version) =======================




render_ability_radar(globals())

# ======================= Pressure Retention module =======================
render_pressure_retention(globals())


# ======================= Shared Race Test Profile =======================

# ======================= Race Plane Analysis — Experimental =======================
render_race_plane(globals())


# ======================= Race Shape Verdict =======================


# ======================= Advanced Models =======================
render_advanced_models(globals())





# ======================= Save Race to Database (Supabase) =======================
render_save_race(globals())


# ======================= Horse Database (Supabase) =======================
render_race_card_view(globals())

render_horse_database_view(globals())
