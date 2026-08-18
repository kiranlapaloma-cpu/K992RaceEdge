import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from common import canon_horse
from database import _supabase_configured, load_horse_history, _fetch_all_horse_rows
from sahr import (
    get_fields_meeting, get_meetings_for_date, meeting_display_label,
    meeting_race_options, race_to_race_edge_card, SAHRError,
)

def _racecard_official_mr(value):
    """Race-card feeds use MR=0 for unrated horses. Display/store that as missing."""
    try:
        v = int(float(value))
    except Exception:
        return None
    return None if v <= 0 else v


def _racecard_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _racecard_float(value):
    try:
        if value is None or value == "":
            return None
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _racecard_parse(raw_text: str) -> dict:
    raw_text = str(raw_text or "").strip()
    if not raw_text:
        raise ValueError("Paste the race-card JSON first.")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("The race-card JSON must contain one race object.")
    runners = payload.get("runners")
    if not isinstance(runners, list):
        raise ValueError("No runners array was found in this race-card JSON.")
    return payload


def _racecard_db_counts() -> dict[str, int]:
    """Return canonical horse -> number of saved Race Edge runs."""
    if not _supabase_configured():
        return {}
    rows = _fetch_all_horse_rows("horse", order_col="horse")
    counts = {}
    for row in rows:
        key = canon_horse(str(row.get("horse", "")))
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _racecard_runner_frame(card: dict, db_counts: dict[str, int] | None = None) -> pd.DataFrame:
    db_counts = db_counts or {}
    rows = []
    for runner in card.get("runners", []):
        horse = str(runner.get("horseName") or "").strip()
        if not horse:
            continue

        status = str(runner.get("status") or "").strip().upper()
        is_reserve = status == "R" or str(runner.get("jockeyName") or "").strip().lower().startswith("reserve")
        horse_weight = _racecard_int(runner.get("horseWeight"))
        weight_delta = _racecard_int(runner.get("horseWeightDelta"))
        official_mr = _racecard_official_mr(runner.get("MR"))
        draw = _racecard_int(runner.get("draw"))
        if is_reserve and (draw is None or draw <= 0):
            draw = None

        rows.append({
            "No.": _racecard_int(runner.get("saddleNo")),
            "Horse": horse,
            "Draw": draw,
            "Age": _racecard_int(runner.get("age")),
            "Sex": str(runner.get("sex") or "").strip().upper(),
            "Weight": _racecard_float(runner.get("weight")),
            "Official MR": official_mr,
            "Horse Wgt": horse_weight,
            "Wgt Î": weight_delta,
            "Jockey": str(runner.get("jockeyName") or "").strip(),
            "Trainer": str(runner.get("trainerName") or "").strip(),
            "Odds": str(runner.get("odds") or "").strip(),
            "Open": str(runner.get("openBet") or "").strip(),
            "Equipment": str(runner.get("equipment") or "").strip(),
            "Days Since Run": _racecard_int(runner.get("restDays")),
            "Race Edge Runs": int(db_counts.get(canon_horse(horse), 0)),
            "Status": "Reserve" if is_reserve else "Runner",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in ["No.", "Draw", "Age", "Official MR", "Horse Wgt", "Wgt Î", "Days Since Run", "Race Edge Runs"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    return df.sort_values(["Status", "No."], ascending=[True, True], na_position="last").reset_index(drop=True)


def _render_racecard_history_snapshot(selected_horses: list[str]):
    """Compact comparison for runners selected directly from the race card."""
    if len(selected_horses) < 2:
        return
    if not _supabase_configured():
        st.info("Supabase is not configured, so saved Race Edge histories cannot be compared.")
        return

    summary_rows = []
    run_rows = []
    for horse in selected_horses:
        try:
            hist = load_horse_history(horse)
        except Exception as exc:
            st.warning(f"Could not load {horse}: {exc}")
            continue

        if hist.empty:
            summary_rows.append({
                "Horse": horse, "Saved Runs": 0, "Latest Official MR": np.nan,
                "Avg MR Achieved": np.nan, "Best MR Achieved": np.nan,
                "Avg MR +/-": np.nan, "Best MR +/-": np.nan,
                "Avg Sustain": np.nan,
            })
            continue

        h = hist.copy()
        h["race_date"] = pd.to_datetime(h.get("race_date"), errors="coerce")
        for c in ["official_mr", "mr_achieved", "sustain_residual", "rpss", "distance", "race_number"]:
            h[c] = pd.to_numeric(h.get(c), errors="coerce")
        h["MR +/-"] = h["mr_achieved"] - h["official_mr"]
        h = h.sort_values(["race_date", "race_number"], ascending=[False, False], na_position="last")

        official = h["official_mr"].dropna()
        mr = h["mr_achieved"].dropna()
        diff = h["MR +/-"].dropna()
        sustain = h["sustain_residual"].dropna()

        summary_rows.append({
            "Horse": horse,
            "Saved Runs": len(h),
            "Latest Official MR": official.iloc[0] if not official.empty else np.nan,
            "Avg MR Achieved": mr.mean() if not mr.empty else np.nan,
            "Best MR Achieved": mr.max() if not mr.empty else np.nan,
            "Avg MR +/-": diff.mean() if not diff.empty else np.nan,
            "Best MR +/-": diff.max() if not diff.empty else np.nan,
            "Avg Sustain": sustain.mean() if not sustain.empty else np.nan,
        })

        for _, row in h.head(5).iterrows():
            run_rows.append({
                "Horse": horse,
                "Date": row.get("race_date"),
                "Track": row.get("track"),
                "Course": row.get("course"),
                "Distance": row.get("distance"),
                "Finish": row.get("finish_position"),
                "Official MR": row.get("official_mr"),
                "MR Achieved": row.get("mr_achieved"),
                "MR +/-": row.get("MR +/-"),
                "Sustain Residual": row.get("sustain_residual"),
            })

    if summary_rows:
        st.markdown("### Selected Runner Comparison")
        summary = pd.DataFrame(summary_rows)
        for c in ["Latest Official MR", "Avg MR Achieved", "Best MR Achieved", "Avg MR +/-", "Best MR +/-", "Avg Sustain"]:
            summary[c] = pd.to_numeric(summary[c], errors="coerce").round(2)
        st.dataframe(summary, width="stretch", hide_index=True)

    if run_rows:
        st.markdown("#### Last 5 Saved Runs")
        runs = pd.DataFrame(run_rows)
        runs["Date"] = pd.to_datetime(runs["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for c in ["Distance", "Official MR", "MR Achieved"]:
            runs[c] = pd.to_numeric(runs[c], errors="coerce").round().astype("Int64")
        for c in ["MR +/-", "Sustain Residual"]:
            runs[c] = pd.to_numeric(runs[c], errors="coerce").round(2)
        st.dataframe(runs, width="stretch", hide_index=True)


def _render_racecard_runner_profiles(active: pd.DataFrame):
    """
    Show every active runner as an expandable full Race Edge profile.
    Saved runs are ordered by race date, newest first.
    """
    st.markdown("### Race Edge Runner Profiles")
    st.caption(
        "Open any runner to review its complete saved Race Edge history. "
        "Runners with no saved history are still shown."
    )

    if not _supabase_configured():
        st.info("Supabase is not configured, so saved Race Edge profiles cannot be loaded.")
        return

    for _, runner in active.iterrows():
        horse = str(runner.get("Horse") or "").strip()
        if not horse:
            continue

        current_official = pd.to_numeric(
            pd.Series([runner.get("Official MR")]), errors="coerce"
        ).iloc[0]
        current_age = runner.get("Age")
        current_weight = runner.get("Weight")
        current_draw = runner.get("Draw")
        saddle_no = runner.get("No.")

        try:
            hist = load_horse_history(horse)
        except Exception as exc:
            with st.expander(f"{horse} â profile unavailable", expanded=False):
                st.warning(f"Could not load saved history: {exc}")
            continue

        if hist.empty:
            label = f"{horse} â No Race Edge history"
            with st.expander(label, expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Official MR", "â" if pd.isna(current_official) else f"{int(round(float(current_official)))}")
                c2.metric("Age", "â" if pd.isna(current_age) else str(int(current_age)))
                c3.metric("Weight", "â" if pd.isna(current_weight) else f"{float(current_weight):.1f} kg")
                c4.metric("Draw", "â" if pd.isna(current_draw) else str(int(current_draw)))
                st.caption("No saved Race Edge runs for this horse yet.")
            continue

        h = hist.copy()
        h["race_date"] = pd.to_datetime(h.get("race_date"), errors="coerce")
        for c in [
            "official_mr", "mr_achieved", "sustain_residual",
            "rpss", "distance", "race_number"
        ]:
            h[c] = pd.to_numeric(h.get(c), errors="coerce")

        h["MR +/-"] = h["mr_achieved"] - h["official_mr"]
        h = h.sort_values(
            ["race_date", "race_number"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)

        mr_vals = h["mr_achieved"].dropna()
        edge_vals = h["MR +/-"].dropna()

        latest_mr = mr_vals.iloc[0] if not mr_vals.empty else float("nan")
        highest_mr = mr_vals.max() if not mr_vals.empty else float("nan")
        best_edge = edge_vals.max() if not edge_vals.empty else float("nan")

        label_bits = [horse, f"{len(h)} saved run{'s' if len(h) != 1 else ''}"]
        if pd.notna(latest_mr):
            label_bits.append(f"Latest MR {int(round(float(latest_mr)))}")
        if pd.notna(best_edge):
            edge_int = int(round(float(best_edge)))
            label_bits.append(f"Best edge {'+' if edge_int > 0 else ''}{edge_int}")

        with st.expander(" â ".join(label_bits), expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Current Official MR",
                "â" if pd.isna(current_official) else f"{int(round(float(current_official)))}",
            )
            c2.metric(
                "Latest MR Achieved",
                "â" if pd.isna(latest_mr) else f"{int(round(float(latest_mr)))}",
            )
            c3.metric(
                "Highest MR Achieved",
                "â" if pd.isna(highest_mr) else f"{int(round(float(highest_mr)))}",
            )
            c4.metric(
                "Best MR Edge",
                "â" if pd.isna(best_edge)
                else f"{'+' if int(round(float(best_edge))) > 0 else ''}{int(round(float(best_edge)))}",
            )

            current_bits = []
            if pd.notna(saddle_no):
                current_bits.append(f"No. {int(saddle_no)}")
            if pd.notna(current_age):
                current_bits.append(f"Age {int(current_age)}")
            if pd.notna(current_weight):
                current_bits.append(f"{float(current_weight):.1f} kg")
            if pd.notna(current_draw):
                current_bits.append(f"Draw {int(current_draw)}")
            if current_bits:
                st.caption("Current race: " + " Â· ".join(current_bits))

            display = h.rename(columns={
                "race_date": "Date",
                "track": "Track",
                "course": "Course",
                "race_number": "Race",
                "distance": "Distance",
                "finish_position": "Finish",
                "official_mr": "Official MR",
                "mr_achieved": "MR Achieved",
                "rpss": "RPSS",
                "race_test": "Race Test",
                "sustain_residual": "Sustain Residual",
                "sustain_verdict": "Sustain Verdict",
                "analyst_note": "Analyst Note",
            })

            cols = [
                "Date", "Track", "Course", "Race", "Distance", "Finish",
                "Official MR", "MR Achieved", "MR +/-", "RPSS",
                "Race Test", "Sustain Residual", "Sustain Verdict", "Analyst Note",
            ]
            cols = [c for c in cols if c in display.columns]
            profile = display[cols].copy()

            if "Date" in profile.columns:
                profile["Date"] = pd.to_datetime(
                    profile["Date"], errors="coerce"
                ).dt.strftime("%Y-%m-%d")

            for c in ["Race", "Distance", "Official MR", "MR Achieved"]:
                if c in profile.columns:
                    profile[c] = pd.to_numeric(
                        profile[c], errors="coerce"
                    ).round().astype("Int64")

            if "MR +/-" in profile.columns:
                edge = pd.to_numeric(profile["MR +/-"], errors="coerce")
                profile["MR +/-"] = edge.round().astype("Int64")

            for c in ["RPSS", "Sustain Residual"]:
                if c in profile.columns:
                    profile[c] = pd.to_numeric(
                        profile[c], errors="coerce"
                    ).round(2)

            st.dataframe(profile, width="stretch", hide_index=True)


def _render_loaded_race_card(card: dict):
    """Render a loaded Race Edge card regardless of whether it came from SAHR or pasted JSON."""
    # Race header.
    date_label = str(card.get("dateFormat") or card.get("date") or "â")
    track = str(card.get("clubName") or "â")
    race_no = _racecard_int(card.get("race"))
    time_label = str(card.get("time") or "â")
    distance = _racecard_int(card.get("distance"))
    surface = str(card.get("surfaceDescr") or "â")
    direction = str(card.get("direction") or "").strip()
    name = str(card.get("name") or "").strip()
    description = str(card.get("description") or "").strip()
    wfa_text = str(card.get("WFA") or "").strip()
    stake = str(card.get("stake") or "").strip()
    currency = str(card.get("currency") or "").strip()

    st.markdown(f"### {track}")
    st.markdown(
        f"**{date_label} Â· Race {race_no if race_no is not None else 'â'} Â· {time_label} Â· "
        f"{distance if distance is not None else 'â'}m Â· {surface}"
        + (f" Â· {direction}" if direction else "")
        + "**"
    )
    if name:
        st.write(name)
    if description:
        st.caption(description)
    if wfa_text:
        st.caption(wfa_text)

    active_count = sum(
        1 for r in card.get("runners", [])
        if str(r.get("status") or "").strip().upper() != "R"
    )
    reserve_count = sum(
        1 for r in card.get("runners", [])
        if str(r.get("status") or "").strip().upper() == "R"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runners", active_count)
    m2.metric("Reserves", reserve_count)
    m3.metric("Distance", "â" if distance is None else f"{distance}m")
    stake_label = "â"
    if stake:
        try:
            stake_label = f"{currency}{int(float(str(stake).replace(',', ''))):,}"
        except Exception:
            stake_label = f"{currency}{stake}"
    m4.metric("Stake", stake_label)

    db_counts = {}
    db_error = None
    if _supabase_configured():
        try:
            db_counts = _racecard_db_counts()
        except Exception as exc:
            db_error = str(exc)

    field = _racecard_runner_frame(card, db_counts)
    if field.empty:
        st.warning("No valid runners were found in this race card.")
        return

    active = field[field["Status"] == "Runner"].copy()
    reserves = field[field["Status"] == "Reserve"].copy()

    if db_error:
        st.warning(f"Race Edge database status could not be loaded: {db_error}")
    elif _supabase_configured():
        with_history = int((active["Race Edge Runs"].fillna(0) > 0).sum())
        st.caption(f"Race Edge database coverage: {with_history}/{len(active)} active runners have saved history.")

    display_cols = [
        "No.", "Horse", "Draw", "Age", "Weight", "Official MR",
        "Jockey", "Trainer", "Race Edge Runs",
    ]

    st.markdown("### Runners")
    st.dataframe(active[display_cols], width="stretch", hide_index=True)

    if not reserves.empty:
        with st.expander(f"Reserves ({len(reserves)})", expanded=False):
            st.dataframe(reserves[display_cols], width="stretch", hide_index=True)

    _render_racecard_runner_profiles(active)


def render_race_card():
    st.title("Race Card")
    st.caption(
        "Load a meeting directly from SAHorseracing, or paste a race-card JSON object manually. "
        "The Race Card remains independent of sectional CSV analysis."
    )

    if "race_card_payload" not in st.session_state:
        st.session_state["race_card_payload"] = None
    if "race_card_input_version" not in st.session_state:
        st.session_state["race_card_input_version"] = 0
    if "sahr_meeting" not in st.session_state:
        st.session_state["sahr_meeting"] = None

    tab_live, tab_json = st.tabs(["SAHorseracing", "Paste JSON"])

    with tab_live:
        st.markdown("### Select Meeting")
        race_date = st.date_input("Race Date", value=datetime.now().date(), key="sahr_race_date")

        # Clear stale selections when the user changes date.
        date_key = race_date.strftime("%Y%m%d")
        if st.session_state.get("sahr_meeting_date_key") != date_key:
            st.session_state["sahr_meeting_date_key"] = date_key
            st.session_state["sahr_available_meetings"] = None
            st.session_state["sahr_meeting"] = None
            st.session_state["race_card_payload"] = None

        if st.button("Find Meetings", type="primary", width="stretch", key="sahr_find_meetings"):
            try:
                found = get_meetings_for_date(race_date)
                st.session_state["sahr_available_meetings"] = found
                st.session_state["sahr_meeting"] = None
                st.session_state["race_card_payload"] = None
                if found:
                    st.success(f"Found {len(found)} meeting{'s' if len(found) != 1 else ''}.")
                else:
                    st.warning("No South African meetings were returned for that date.")
            except Exception as exc:
                st.session_state["sahr_available_meetings"] = None
                st.error(f"Could not discover meetings: {exc}")

        available = st.session_state.get("sahr_available_meetings")
        if available:
            labels = []
            label_to_item = {}
            for item in available:
                base = meeting_display_label(item)
                label = base
                # Ensure labels remain unique even if a venue has two course IDs.
                if label in label_to_item:
                    label = f"{base} (Club {item.get('club')})"
                labels.append(label)
                label_to_item[label] = item

            selected_meeting_label = st.selectbox("Meeting", labels, key="sahr_meeting_select")
            selected_meeting = label_to_item[selected_meeting_label]

            if st.button("Load Meeting", width="stretch", key="sahr_load_meeting"):
                try:
                    meeting = get_fields_meeting(race_date, int(selected_meeting["club"]))
                    st.session_state["sahr_meeting"] = meeting
                    st.session_state["race_card_payload"] = None
                    st.success(str(meeting.get("heading") or "Meeting loaded."))
                except Exception as exc:
                    st.session_state["sahr_meeting"] = None
                    st.error(f"Could not load SAHorseracing meeting: {exc}")

        meeting = st.session_state.get("sahr_meeting")
        if meeting:
            heading = str(meeting.get("heading") or "").strip()
            if heading:
                st.markdown(f"**{heading}**")
            options = meeting_race_options(meeting)
            if options:
                option_map = {label: key for key, label in options}
                selected_label = st.selectbox("Race", list(option_map.keys()), key="sahr_race_select")
                if st.button("Load Selected Race", width="stretch", key="sahr_load_race"):
                    try:
                        card = race_to_race_edge_card(meeting, option_map[selected_label])
                        st.session_state["race_card_payload"] = card
                        st.success(f"Race {card.get('race')} loaded into Race Edge.")
                    except Exception as exc:
                        st.error(f"Could not load selected race: {exc}")

            st.caption(
                "Source: SAHorseracing / National Horse Racing Bureau data feed. "
                "The feed states that use/display is restricted to private use and commercial use requires a licence."
            )

    with tab_json:
        raw = st.text_area(
            "Race-card JSON",
            height=220,
            placeholder='Paste the JSON containing the race details and "runners" array...',
            key=f"race_card_json_{st.session_state['race_card_input_version']}",
        )

        b1, b2 = st.columns([1, 1])
        with b1:
            load_clicked = st.button("Load Race Card", type="primary", width="stretch", key="json_load_card")
        with b2:
            clear_clicked = st.button("Clear Race Card", width="stretch", key="json_clear_card")

        if clear_clicked:
            st.session_state["race_card_payload"] = None
            st.session_state["race_card_input_version"] += 1
            st.rerun()

        if load_clicked:
            try:
                st.session_state["race_card_payload"] = _racecard_parse(raw)
                st.success("Race card loaded.")
            except Exception as exc:
                st.error(f"Could not load race card: {exc}")

    card = st.session_state.get("race_card_payload")
    if not card:
        return

    st.divider()
    _render_loaded_race_card(card)
