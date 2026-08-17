import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = object
from .common import canon_horse

def _supabase_configured() -> bool:
    try:
        return bool(st.secrets["supabase"]["url"] and st.secrets["supabase"]["key"])
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def get_supabase_client():
    if create_client is None:
        raise RuntimeError("The `supabase` package is not installed. Add `supabase` to requirements.txt.")
    try:
        url = str(st.secrets["supabase"]["url"]).strip()
        key = str(st.secrets["supabase"]["key"]).strip()
    except Exception as exc:
        raise RuntimeError("Supabase credentials are missing from Streamlit Secrets.") from exc
    if not url or not key:
        raise RuntimeError("Supabase URL or key is blank in Streamlit Secrets.")
    return create_client(url, key)


def _db_num(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _db_round_mr(value):
    """Round a calculated MR to the nearest whole number for database storage."""
    value = _db_num(value)
    if value is None:
        return None
    # Merit ratings are non-negative; this gives conventional .5-up rounding.
    return int(math.floor(value + 0.5))


def database_sustain_verdict(value) -> str:
    value = _db_num(value)
    if value is None:
        return "Unavailable"
    if value >= 3.0:
        return "Major above expectation"
    if value >= 1.5:
        return "Above expectation"
    if value > -1.5:
        return "Around expectation"
    if value > -3.0:
        return "Below expectation"
    return "Significant late underperformance"


def _fetch_all_horse_rows(columns: str, order_col: str | None = None, desc: bool = False) -> list[dict]:
    """Fetch every horse_runs row in pages so Supabase's per-request row cap
    can never make the Horse Database search silently incomplete.
    """
    client = get_supabase_client()
    page_size = 1000
    start = 0
    rows: list[dict] = []
    while True:
        query = client.table("horse_runs").select(columns)
        if order_col:
            query = query.order(order_col, desc=desc)
        response = query.range(start, start + page_size - 1).execute()
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def load_saved_horses() -> list[str]:
    # Do not use .limit(10000) here: Supabase projects commonly impose a
    # server-side 1,000-row maximum per response. Pagination guarantees that
    # every saved horse remains searchable as the Race Edge database grows.
    rows = _fetch_all_horse_rows("horse", order_col="horse")
    return sorted({
        str(row.get("horse", "")).strip()
        for row in rows
        if str(row.get("horse", "")).strip()
    })


def load_horse_history(horse: str) -> pd.DataFrame:
    """Load every saved run for one horse without scanning a capped global set."""
    client = get_supabase_client()
    columns = (
        "id,horse,finish_position,race_date,track,course,race_number,distance,"
        "rpss,race_test,official_mr,mr_achieved,sustain_residual,sustain_verdict,analyst_note"
    )

    target = canon_horse(str(horse or ""))
    if not target:
        return pd.DataFrame()

    # Records saved by this app are canonicalised before insert, so querying
    # the canonical value directly is both faster and immune to the global
    # Supabase row cap. Keep a paginated fallback for legacy/non-canonical rows.
    response = (
        client.table("horse_runs")
        .select(columns)
        .eq("horse", target)
        .order("race_date", desc=True)
        .execute()
    )
    direct_rows = response.data or []
    if direct_rows:
        return pd.DataFrame(direct_rows)

    rows = _fetch_all_horse_rows(columns, order_col="race_date", desc=True)
    matched = [
        row for row in rows
        if canon_horse(str(row.get("horse", ""))) == target
    ]
    return pd.DataFrame(matched)


def save_horse_runs(records: list[dict]) -> int:
    if not records:
        return 0
    client = get_supabase_client()
    (
        client.table("horse_runs")
        .upsert(records, on_conflict="horse,race_date,track,course,race_number")
        .execute()
    )
    load_saved_horses.clear() if hasattr(load_saved_horses, "clear") else None
    load_rating_improver_runs.clear() if hasattr(load_rating_improver_runs, "clear") else None
    return len(records)


@st.cache_data(ttl=60, show_spinner=False)
def load_rating_improver_runs() -> pd.DataFrame:
    """Load database runs that have both Official MR and MR Achieved available."""
    columns = (
        "id,horse,finish_position,race_date,track,course,race_number,distance,"
        "rpss,race_test,official_mr,mr_achieved,sustain_residual,sustain_verdict,analyst_note"
    )
    rows = _fetch_all_horse_rows(columns, order_col="race_date", desc=True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["official_mr"] = pd.to_numeric(df.get("official_mr"), errors="coerce")
    df["mr_achieved"] = pd.to_numeric(df.get("mr_achieved"), errors="coerce")
    df["race_date"] = pd.to_datetime(df.get("race_date"), errors="coerce")
    df["race_number"] = pd.to_numeric(df.get("race_number"), errors="coerce")
    df["distance"] = pd.to_numeric(df.get("distance"), errors="coerce")
    df["sustain_residual"] = pd.to_numeric(df.get("sustain_residual"), errors="coerce")
    df["rpss"] = pd.to_numeric(df.get("rpss"), errors="coerce")
    df = df.dropna(subset=["horse", "official_mr", "mr_achieved"]).copy()
    df["MR Edge"] = df["mr_achieved"] - df["official_mr"]
    return df


def render_rating_improvers():
    st.markdown("### Rating Improvers")
    st.caption(
        "Find horses whose Race Edge MR Achieved exceeded their Official MR. "
        "The default view is +5 points or better from the last 30 days."
    )
    if not _supabase_configured():
        st.warning("Supabase is not configured in Streamlit Secrets.")
        return

    try:
        all_runs = load_rating_improver_runs()
    except Exception as exc:
        st.error(f"Could not load rating improvers: {exc}")
        return

    if all_runs.empty:
        st.info("No saved runs currently contain both Official MR and MR Achieved.")
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        threshold = st.number_input(
            "Minimum MR improvement",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            key="db_rating_improver_threshold",
            help="MR Achieved minus Official MR. Example: Official 80 and Achieved 86 = +6.",
        )
    with c2:
        horse_filter = st.text_input(
            "Filter horses",
            placeholder="Optional: type part of a horse name...",
            key="db_rating_improver_filter",
        ).strip()

    st.markdown("##### Time Period")
    period = st.radio(
        "Show qualifying performances from",
        ["Last 7 days", "Last 30 days", "This month", "This season", "Custom"],
        index=1,
        horizontal=True,
        key="db_rating_improver_period",
        label_visibility="collapsed",
    )

    today = pd.Timestamp.now().normalize()
    if period == "Last 7 days":
        start_date, end_date = today - pd.Timedelta(days=6), today
    elif period == "Last 30 days":
        start_date, end_date = today - pd.Timedelta(days=29), today
    elif period == "This month":
        start_date, end_date = today.replace(day=1), today
    elif period == "This season":
        # South African racing season starts on 1 August.
        season_year = today.year if today.month >= 8 else today.year - 1
        start_date, end_date = pd.Timestamp(season_year, 8, 1), today
    else:
        default_start = today - pd.Timedelta(days=29)
        custom = st.date_input(
            "Custom date range",
            value=(default_start.date(), today.date()),
            max_value=today.date(),
            key="db_rating_improver_custom_dates",
        )
        if isinstance(custom, (tuple, list)) and len(custom) == 2:
            start_date, end_date = pd.Timestamp(custom[0]), pd.Timestamp(custom[1])
        else:
            st.info("Choose both a start date and an end date.")
            return

    qualifying = all_runs.loc[
        (all_runs["MR Edge"] >= float(threshold))
        & (all_runs["race_date"] >= start_date)
        & (all_runs["race_date"] <= end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    ].copy()
    if horse_filter:
        key = canon_horse(horse_filter)
        qualifying = qualifying.loc[
            qualifying["horse"].astype(str).map(canon_horse).str.contains(key, na=False)
        ].copy()

    if qualifying.empty:
        st.info(f"No performances in the selected time period are +{int(threshold)} MR or better.")
        return

    qualifying = qualifying.sort_values(
        ["race_date", "MR Edge", "race_number"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    # Horse-level watchlist: one row per horse, ranked by its strongest qualifying run.
    summary_rows = []
    for horse, group in qualifying.groupby("horse", sort=False):
        group = group.sort_values(
            ["MR Edge", "race_date", "race_number"],
            ascending=[False, False, False],
            na_position="last",
        )
        best = group.iloc[0]
        latest = group.sort_values(
            ["race_date", "race_number"],
            ascending=[False, False],
            na_position="last",
        ).iloc[0]
        summary_rows.append({
            "Horse": str(horse),
            "Best Edge": float(best["MR Edge"]),
            "Qualifying Runs": int(len(group)),
            "Best Official MR": best.get("official_mr"),
            "Best MR Achieved": best.get("mr_achieved"),
            "Best Run Date": best.get("race_date"),
            "Best Run Distance": best.get("distance"),
            "Latest Qualifying Date": latest.get("race_date"),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Latest Qualifying Date", "Best Edge", "Qualifying Runs"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    for c in ["Best Official MR", "Best MR Achieved", "Best Run Distance"]:
        summary[c] = pd.to_numeric(summary[c], errors="coerce").round().astype("Int64")
    summary["Best Edge"] = pd.to_numeric(summary["Best Edge"], errors="coerce").round().astype("Int64")
    summary["Best Edge"] = summary["Best Edge"].map(lambda x: "—" if pd.isna(x) else f"+{int(x)}")
    for c in ["Best Run Date", "Latest Qualifying Date"]:
        summary[c] = pd.to_datetime(summary[c], errors="coerce").dt.strftime("%Y-%m-%d")

    m1, m2, m3 = st.columns(3)
    m1.metric("Horses Found", int(summary["Horse"].nunique()))
    m2.metric("Qualifying Runs", int(len(qualifying)))
    best_edge = float(pd.to_numeric(qualifying["MR Edge"], errors="coerce").max())
    m3.metric("Largest Edge", f"+{_db_round_mr(best_edge)} MR")

    st.markdown("#### Horses Ahead of the Handicap")
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("#### Qualifying Performances")
    runs = qualifying.rename(columns={
        "horse": "Horse",
        "race_date": "Date",
        "track": "Track",
        "course": "Course",
        "race_number": "Race",
        "distance": "Distance",
        "finish_position": "Finish",
        "official_mr": "Official MR",
        "mr_achieved": "MR Achieved",
        "sustain_residual": "Sustain Residual",
        "rpss": "RPSS",
        "race_test": "Race Test",
    }).copy()
    runs["Date"] = pd.to_datetime(runs["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["Race", "Distance", "Official MR", "MR Achieved"]:
        runs[c] = pd.to_numeric(runs[c], errors="coerce").round().astype("Int64")
    runs["MR Edge"] = pd.to_numeric(runs["MR Edge"], errors="coerce").round().astype("Int64")
    runs["MR Edge"] = runs["MR Edge"].map(lambda x: "—" if pd.isna(x) else f"+{int(x)}")
    runs["Sustain Residual"] = pd.to_numeric(runs["Sustain Residual"], errors="coerce").round(2)
    runs["RPSS"] = pd.to_numeric(runs["RPSS"], errors="coerce").round(2)
    run_cols = [
        "Horse", "Date", "Track", "Course", "Race", "Distance", "Finish",
        "Official MR", "MR Achieved", "MR Edge", "RPSS", "Sustain Residual", "Race Test",
    ]
    st.dataframe(runs[run_cols], width="stretch", hide_index=True)

    st.markdown("#### Open Horse History")
    selected = st.selectbox(
        "Select an improver",
        options=summary["Horse"].tolist(),
        index=None,
        placeholder="Choose a horse to inspect every saved run...",
        key="db_rating_improver_selected_horse",
    )
    if not selected:
        return

    try:
        history = load_horse_history(selected)
    except Exception as exc:
        st.error(f"Could not load horse history: {exc}")
        return
    if history.empty:
        st.info("No saved history was found for this horse.")
        return

    history = history.copy()
    history["race_date"] = pd.to_datetime(history.get("race_date"), errors="coerce")
    for c in ["official_mr", "mr_achieved", "race_number", "distance", "rpss", "sustain_residual"]:
        history[c] = pd.to_numeric(history.get(c), errors="coerce")
    history["MR Edge"] = history["mr_achieved"] - history["official_mr"]
    history = history.sort_values(
        ["race_date", "race_number"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)

    full = history.rename(columns={
        "race_date": "Date",
        "track": "Track",
        "course": "Course",
        "race_number": "Race",
        "distance": "Distance",
        "finish_position": "Finish",
        "official_mr": "Official MR",
        "mr_achieved": "MR Achieved",
        "rpss": "RPSS",
        "sustain_residual": "Sustain Residual",
        "race_test": "Race Test",
        "analyst_note": "Analyst Note",
    })
    full["Date"] = full["Date"].dt.strftime("%Y-%m-%d")
    for c in ["Race", "Distance", "Official MR", "MR Achieved"]:
        full[c] = pd.to_numeric(full[c], errors="coerce").round().astype("Int64")
    full["MR Edge"] = pd.to_numeric(full["MR Edge"], errors="coerce").round().astype("Int64")
    full["RPSS"] = pd.to_numeric(full["RPSS"], errors="coerce").round(2)
    full["Sustain Residual"] = pd.to_numeric(full["Sustain Residual"], errors="coerce").round(2)
    st.markdown(f"**{selected} — complete Race Edge history**")
    st.dataframe(
        full[[
            "Date", "Track", "Course", "Race", "Distance", "Finish", "Official MR",
            "MR Achieved", "MR Edge", "RPSS", "Sustain Residual", "Race Test", "Analyst Note",
        ]],
        width="stretch",
        hide_index=True,
    )


def render_horse_search():
    st.markdown("### Search Horse")
    st.caption("Search a horse to view every saved run, ordered by race date with the most recent first.")
    if not _supabase_configured():
        st.warning("Supabase is not configured in Streamlit Secrets.")
        return
    try:
        horses = load_saved_horses()
    except Exception as exc:
        st.error(f"Could not connect to Supabase: {exc}")
        return
    if not horses:
        st.info("The database is connected, but no horse runs have been saved yet.")
        return

    query = st.text_input(
        "Horse name",
        placeholder="Type part of a horse's name...",
        key="db_horse_query",
    ).strip()
    canonical_query = canon_horse(query)
    matches = (
        [h for h in horses if canonical_query in canon_horse(h)]
        if canonical_query
        else horses
    )
    selected = st.selectbox(
        "Select horse",
        matches,
        index=None,
        placeholder="Choose a horse...",
        key="db_horse_select",
    )
    if not selected:
        return

    try:
        history = load_horse_history(selected)
    except Exception as exc:
        st.error(f"Could not load horse history: {exc}")
        return
    if history.empty:
        st.info("No saved runs found for this horse.")
        return

    history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce")
    history["official_mr"] = pd.to_numeric(history.get("official_mr"), errors="coerce")
    history["mr_achieved"] = pd.to_numeric(history["mr_achieved"], errors="coerce")
    history["sustain_residual"] = pd.to_numeric(history["sustain_residual"], errors="coerce")
    history["rpss"] = pd.to_numeric(history.get("rpss"), errors="coerce")
    history["race_number"] = pd.to_numeric(history["race_number"], errors="coerce")
    history = history.sort_values(
        ["race_date", "race_number"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    valid_mr = history.loc[history["mr_achieved"].notna(), "mr_achieved"]
    latest_mr = valid_mr.iloc[0] if not valid_mr.empty else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("Saved Runs", len(history))
    c2.metric("Latest MR", "—" if not np.isfinite(latest_mr) else f"{latest_mr:.0f}")
    c3.metric("Highest MR", "—" if valid_mr.empty else f"{valid_mr.max():.0f}")

    display = history.rename(columns={
        "race_date": "Date",
        "finish_position": "Finish",
        "track": "Track",
        "course": "Course",
        "race_number": "Race",
        "distance": "Distance",
        "rpss": "RPSS",
        "race_test": "Race Test",
        "official_mr": "Official MR",
        "mr_achieved": "MR Achieved",
        "sustain_residual": "Sustain Residual",
        "sustain_verdict": "Sustain Verdict",
        "analyst_note": "Analyst Note",
    })
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    display["Race"] = pd.to_numeric(display["Race"], errors="coerce").astype("Int64")
    display["Official MR"] = pd.to_numeric(
        display["Official MR"], errors="coerce"
    ).round().astype("Int64")
    display["MR Achieved"] = pd.to_numeric(
        display["MR Achieved"], errors="coerce"
    ).round().astype("Int64")
    cols = [
        "Date", "Track", "Course", "Race", "Distance", "Finish", "Official MR", "MR Achieved",
        "RPSS", "Race Test", "Sustain Residual", "Sustain Verdict", "Analyst Note",
    ]
    st.dataframe(display[cols], width="stretch", hide_index=True)

    st.markdown("### Career Notebook")
    for _, run in display.iterrows():
        date_label = run.get("Date") or "Unknown date"
        track_label = str(run.get("Track") or "")
        course_label = str(run.get("Course") or "")
        race_no = run.get("Race")
        distance = run.get("Distance")
        race_text = "—" if pd.isna(race_no) else str(int(race_no))
        dist_text = "—" if pd.isna(distance) else f"{int(distance)}m"
        heading = f"{date_label} · {track_label} {course_label} · Race {race_text} · {dist_text}"
        with st.expander(heading):
            e1, e2, e3 = st.columns(3)
            mr_value = _db_num(run.get("MR Achieved"))
            sr_value = _db_num(run.get("Sustain Residual"))
            rpss_value = _db_num(run.get("RPSS"))
            e1.metric("Finish", str(run.get("Finish") or "—"))
            e2.metric("MR Achieved", "—" if mr_value is None else f"{_db_round_mr(mr_value)}")
            e3.metric("RPSS", "—" if rpss_value is None else f"{rpss_value:.2f}")
            st.metric("Sustain Residual", "—" if sr_value is None else f"{sr_value:+.2f}")
            st.markdown(f"**Race Test:** {run.get('Race Test') or '—'}")
            st.markdown(f"**Sustain Verdict:** {run.get('Sustain Verdict') or '—'}")
            note = str(run.get("Analyst Note") or "").strip()
            st.markdown(f"**Analyst Note:** {note if note else '—'}")

    st.download_button(
        "Download this horse's history (CSV)",
        data=display[cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{canon_horse(selected).replace(' ', '_').lower()}_race_edge_history.csv",
        mime="text/csv",
    )


def render_horse_compare():
    st.markdown("### Compare Horses")
    st.caption(
        "Compare 2–5 horses from the Race Edge database. Filters are applied to each horse's saved history, "
        "so you can compare like-for-like runs rather than whole careers blindly."
    )
    if not _supabase_configured():
        st.warning("Supabase is not configured in Streamlit Secrets.")
        return

    try:
        horses = load_saved_horses()
    except Exception as exc:
        st.error(f"Could not connect to Supabase: {exc}")
        return
    if len(horses) < 2:
        st.info("At least two horses must be saved before a comparison can be made.")
        return

    selected = st.multiselect(
        "Horses to compare",
        horses,
        max_selections=5,
        placeholder="Choose 2–5 horses...",
        key="db_compare_horses",
    )
    if len(selected) < 2:
        st.info("Select at least two horses to compare.")
        return

    histories = {}
    load_errors = []
    for horse in selected:
        try:
            h = load_horse_history(horse)
        except Exception as exc:
            load_errors.append(f"{horse}: {exc}")
            continue
        if h.empty:
            continue
        h = h.copy()
        h["race_date"] = pd.to_datetime(h.get("race_date"), errors="coerce")
        for c in ["official_mr", "mr_achieved", "sustain_residual", "rpss", "distance", "race_number"]:
            h[c] = pd.to_numeric(h.get(c), errors="coerce")
        h["MR +/-"] = h["mr_achieved"] - h["official_mr"]
        h["horse"] = str(horse)
        histories[horse] = h

    if load_errors:
        st.warning("Some histories could not be loaded: " + "; ".join(load_errors))
    if len(histories) < 2:
        st.info("Not enough horse histories could be loaded for a comparison.")
        return

    combined = pd.concat(histories.values(), ignore_index=True)
    valid_dist = pd.to_numeric(combined["distance"], errors="coerce").dropna()
    min_dist = int(valid_dist.min()) if not valid_dist.empty else 800
    max_dist = int(valid_dist.max()) if not valid_dist.empty else 4000

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        run_scope = st.selectbox(
            "Runs",
            ["All runs", "Last 3", "Last 5"],
            key="db_compare_run_scope",
        )
    with f2:
        if min_dist == max_dist:
            distance_range = (min_dist, max_dist)
            st.number_input("Distance", value=min_dist, disabled=True, key="db_compare_fixed_distance")
        else:
            distance_range = st.slider(
                "Distance range (m)",
                min_value=min_dist,
                max_value=max_dist,
                value=(min_dist, max_dist),
                step=50,
                key="db_compare_distance_range",
            )
    tracks = sorted({str(v).strip() for v in combined.get("track", pd.Series(dtype=str)).dropna() if str(v).strip()})
    courses = sorted({str(v).strip() for v in combined.get("course", pd.Series(dtype=str)).dropna() if str(v).strip()})
    with f3:
        track_filter = st.selectbox("Track", ["All"] + tracks, key="db_compare_track")
    with f4:
        course_filter = st.selectbox("Course", ["All"] + courses, key="db_compare_course")

    filtered = {}
    n_last = {"Last 3": 3, "Last 5": 5}.get(run_scope)
    for horse, h in histories.items():
        hh = h.copy()
        hh = hh[
            pd.to_numeric(hh["distance"], errors="coerce").between(distance_range[0], distance_range[1], inclusive="both")
        ]
        if track_filter != "All":
            hh = hh[hh["track"].astype(str) == track_filter]
        if course_filter != "All":
            hh = hh[hh["course"].astype(str) == course_filter]
        hh = hh.sort_values(["race_date", "race_number"], ascending=[False, False], na_position="last")
        if n_last is not None:
            hh = hh.head(n_last)
        filtered[horse] = hh.reset_index(drop=True)

    summary_rows = []
    for horse in selected:
        h = filtered.get(horse, pd.DataFrame())
        full = histories.get(horse, pd.DataFrame())
        if full.empty:
            continue
        full_sorted = full.sort_values(["race_date", "race_number"], ascending=[False, False], na_position="last")
        official = pd.to_numeric(full_sorted.get("official_mr"), errors="coerce").dropna()
        current_official = official.iloc[0] if not official.empty else np.nan
        mr = pd.to_numeric(h.get("mr_achieved"), errors="coerce") if not h.empty else pd.Series(dtype=float)
        diff = pd.to_numeric(h.get("MR +/-"), errors="coerce") if not h.empty else pd.Series(dtype=float)
        sustain = pd.to_numeric(h.get("sustain_residual"), errors="coerce") if not h.empty else pd.Series(dtype=float)
        summary_rows.append({
            "Horse": horse,
            "Runs": int(len(h)),
            "Current Official MR": current_official,
            "Avg MR Achieved": mr.mean() if not mr.empty else np.nan,
            "Best MR Achieved": mr.max() if not mr.empty else np.nan,
            "Avg MR +/-": diff.mean() if not diff.empty else np.nan,
            "Best MR +/-": diff.max() if not diff.empty else np.nan,
            "Avg Sustain": sustain.mean() if not sustain.empty else np.nan,
        })

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        st.info("No runs match the selected filters.")
        return

    st.markdown("### Comparison Summary")
    for c in ["Current Official MR", "Avg MR Achieved", "Best MR Achieved", "Avg MR +/-", "Best MR +/-", "Avg Sustain"]:
        summary[c] = pd.to_numeric(summary[c], errors="coerce").round(2)
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("### Run-by-Run Comparison")
    run_rows = []
    for horse in selected:
        h = filtered.get(horse, pd.DataFrame()).copy()
        if h.empty:
            continue
        for _, row in h.iterrows():
            run_rows.append({
                "Horse": horse,
                "Date": row.get("race_date"),
                "Track": row.get("track"),
                "Course": row.get("course"),
                "Race": row.get("race_number"),
                "Distance": row.get("distance"),
                "Finish": row.get("finish_position"),
                "Official MR": row.get("official_mr"),
                "MR Achieved": row.get("mr_achieved"),
                "MR +/-": row.get("MR +/-"),
                "RPSS": row.get("rpss"),
                "Sustain Residual": row.get("sustain_residual"),
            })
    runs = pd.DataFrame(run_rows)
    if not runs.empty:
        runs["Date"] = pd.to_datetime(runs["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for c in ["Race", "Distance", "Official MR", "MR Achieved"]:
            runs[c] = pd.to_numeric(runs[c], errors="coerce").round().astype("Int64")
        for c in ["MR +/-", "RPSS", "Sustain Residual"]:
            runs[c] = pd.to_numeric(runs[c], errors="coerce").round(2)
        st.dataframe(runs, width="stretch", hide_index=True)

    # Shared-race detection uses the same fields as the database uniqueness key,
    # excluding horse: date + track + course + race number.
    st.markdown("### Head-to-Head Races")
    shared_source = combined[combined["horse"].isin(selected)].copy()
    shared_source["race_key"] = (
        shared_source["race_date"].dt.strftime("%Y-%m-%d").fillna("") + "|" +
        shared_source["track"].astype(str) + "|" +
        shared_source["course"].astype(str) + "|" +
        shared_source["race_number"].astype("Int64").astype(str)
    )
    shared_keys = []
    for key, grp in shared_source.groupby("race_key", dropna=False):
        if grp["horse"].nunique() >= 2:
            shared_keys.append(key)
    if not shared_keys:
        st.caption("No saved races currently contain two or more of the selected horses.")
    else:
        hh = shared_source[shared_source["race_key"].isin(shared_keys)].copy()
        hh = hh.sort_values(["race_date", "race_number", "horse"], ascending=[False, False, True])
        head_rows = pd.DataFrame({
            "Date": hh["race_date"].dt.strftime("%Y-%m-%d"),
            "Track": hh["track"],
            "Course": hh["course"],
            "Race": hh["race_number"].round().astype("Int64"),
            "Distance": hh["distance"].round().astype("Int64"),
            "Horse": hh["horse"],
            "Finish": hh["finish_position"],
            "Official MR": hh["official_mr"].round().astype("Int64"),
            "MR Achieved": hh["mr_achieved"].round().astype("Int64"),
            "MR +/-": hh["MR +/-"].round(2),
            "Sustain Residual": hh["sustain_residual"].round(2),
        })
        st.dataframe(head_rows, width="stretch", hide_index=True)


