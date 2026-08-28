import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
from common import canon_horse
from database import _supabase_configured, load_horse_history, _fetch_all_horse_rows
from performance_profile import build_performance_profile, render_performance_profile
from race_prediction import build_race_predictions, prediction_display_table
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



def _racecard_mr_achieved_stats(horse: str) -> tuple[int | None, int | None]:
    """Return (latest MR Achieved, highest MR Achieved) from saved Race Edge history."""
    if not horse or not _supabase_configured():
        return None, None
    try:
        hist = load_horse_history(horse)
    except Exception:
        return None, None
    if hist is None or hist.empty or "mr_achieved" not in hist.columns:
        return None, None

    h = hist.copy()
    h["mr_achieved"] = pd.to_numeric(h["mr_achieved"], errors="coerce")
    if "race_date" in h.columns:
        h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce")
    else:
        h["race_date"] = pd.NaT
    if "race_number" in h.columns:
        h["race_number"] = pd.to_numeric(h["race_number"], errors="coerce")
    else:
        h["race_number"] = np.nan

    h = h.sort_values(
        ["race_date", "race_number"],
        ascending=[False, False],
        na_position="last",
    )
    achieved = h["mr_achieved"].dropna()
    if achieved.empty:
        return None, None

    latest = int(round(float(achieved.iloc[0])))
    highest = int(round(float(achieved.max())))
    return latest, highest

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
        latest_mr_achieved, highest_mr_achieved = _racecard_mr_achieved_stats(horse)
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
            "Wgt Change": weight_delta,
            "Jockey": str(runner.get("jockeyName") or "").strip(),
            "Trainer": str(runner.get("trainerName") or "").strip(),
            "Odds": str(runner.get("odds") or "").strip(),
            "Open": str(runner.get("openBet") or "").strip(),
            "Equipment": str(runner.get("equipment") or "").strip(),
            "Days Since Run": _racecard_int(runner.get("restDays")),
            "Race Edge Runs": int(db_counts.get(canon_horse(horse), 0)),
            "Latest MR Achieved": latest_mr_achieved,
            "Highest MR Achieved": highest_mr_achieved,
            "Status": "Reserve" if is_reserve else "Runner",
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for c in [
        "No.", "Draw", "Age", "Official MR", "Horse Wgt", "Wgt Change",
        "Days Since Run", "Race Edge Runs", "Latest MR Achieved", "Highest MR Achieved"
    ]:
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

        performance_profile = build_performance_profile(
            hist,
            current_official_mr=current_official,
        )

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



def _compact_group(value):
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return "-"
    return text.replace("Group ", "")


def _runner_label(number, horse):
    try:
        if number is not None and not pd.isna(number):
            return f"#{int(round(float(number)))} {horse}"
    except Exception:
        pass
    return str(horse)


def _prediction_lookup(prediction: dict | None) -> pd.DataFrame:
    if not prediction:
        return pd.DataFrame()
    rows = prediction.get("rows")
    if rows is None or rows.empty:
        return pd.DataFrame()
    out = rows.copy()
    out["Horse Key"] = out["Horse"].map(canon_horse)
    return out


def _enhanced_racecard_table(active: pd.DataFrame, prediction: dict | None) -> pd.DataFrame:
    """Primary race-day table: today's card plus Race Edge ability snapshot."""
    table = active.copy()
    pred = _prediction_lookup(prediction)

    if not pred.empty:
        merge_cols = [
            c for c in [
                "Horse Key", "Latest MR", "Established MR", "Peak MR",
                "Latest Group", "Established Group", "Peak Group"
            ]
            if c in pred.columns
        ]
        table["Horse Key"] = table["Horse"].map(canon_horse)
        table = table.merge(
            pred[merge_cols].drop_duplicates("Horse Key"),
            on="Horse Key",
            how="left",
        )
    else:
        for c in [
            "Latest MR", "Established MR", "Peak MR",
            "Latest Group", "Established Group", "Peak Group"
        ]:
            table[c] = np.nan

    table["Groups"] = table.apply(
        lambda r: " / ".join([
            _compact_group(r.get("Latest Group")),
            _compact_group(r.get("Established Group")),
            _compact_group(r.get("Peak Group")),
        ]),
        axis=1,
    )

    out = table[[
        c for c in [
            "No.", "Horse", "Draw", "Age", "Weight", "Official MR",
            "Latest MR", "Established MR", "Peak MR", "Groups"
        ]
        if c in table.columns
    ]].copy()

    for c in ["No.", "Draw", "Age", "Official MR"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round().astype("Int64")
    if "Weight" in out.columns:
        out["Weight"] = pd.to_numeric(out["Weight"], errors="coerce").round(1)
    for c in ["Latest MR", "Established MR", "Peak MR"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(1)

    out = out.rename(columns={
        "Weight": "Wgt",
        "Official MR": "MR",
        "Latest MR": "Latest",
        "Established MR": "Est.",
        "Peak MR": "Peak",
    })
    return out


def _format_margin(margin) -> str:
    margin = _racecard_float(margin)
    if margin is None or abs(margin) < 1e-9:
        return "0.0L"
    if abs(margin * 4 - round(margin * 4)) < 1e-9:
        if abs(margin * 2 - round(margin * 2)) < 1e-9:
            return f"{margin:.1f}L"
        return f"{margin:.2f}L"
    return f"{margin:.2f}L"


def _scenario_group_blocks(scenario: pd.DataFrame, group_col: str):
    """Return ordered (group, dataframe) blocks for a scenario."""
    if scenario is None or scenario.empty:
        return []
    work = scenario.copy()
    if group_col not in work.columns:
        work[group_col] = "Group A"
    groups = []
    for group in work[group_col].dropna().drop_duplicates().tolist():
        groups.append((str(group), work.loc[work[group_col] == group].copy()))
    return groups


def _render_racecard_runner_profiles(active: pd.DataFrame, prediction: dict | None = None):
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

    pred_lookup = _prediction_lookup(prediction)
    pred_by_key = {}
    if not pred_lookup.empty:
        pred_by_key = {
            str(row["Horse Key"]): row
            for _, row in pred_lookup.iterrows()
        }

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
            with st.expander(f"{_runner_label(saddle_no, horse)} | profile unavailable", expanded=False):
                st.warning(f"Could not load saved history: {exc}")
            continue

        if hist.empty:
            label = f"{_runner_label(saddle_no, horse)} | No Race Edge history"
            with st.expander(label, expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Official MR", "-" if pd.isna(current_official) else f"{int(round(float(current_official)))}")
                c2.metric("Age", "-" if pd.isna(current_age) else str(int(current_age)))
                c3.metric("Weight", "-" if pd.isna(current_weight) else f"{float(current_weight):.1f} kg")
                c4.metric("Draw", "-" if pd.isna(current_draw) else str(int(current_draw)))
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

        pred_row = pred_by_key.get(canon_horse(horse))
        group_text = "-"
        if pred_row is not None:
            group_text = " / ".join([
                _compact_group(pred_row.get("Latest Group")),
                _compact_group(pred_row.get("Established Group")),
                _compact_group(pred_row.get("Peak Group")),
            ])

        label_bits = [
            _runner_label(saddle_no, horse),
            f"{len(h)} saved run{'s' if len(h) != 1 else ''}",
        ]
        if pd.notna(current_official):
            label_bits.append(f"MR {int(round(float(current_official)))}")
        if pd.notna(current_weight):
            label_bits.append(f"{float(current_weight):.1f}kg")
        if group_text != "- / - / -":
            label_bits.append(group_text)

        # Build the shared Performance Profile from this horse's historical runs.
        # The live Race Card Official MR is passed separately as the current mark;
        # historical MR +/- values continue to use each saved run's own Official MR.
        performance_profile = build_performance_profile(
            h,
            current_official_mr=None if pd.isna(current_official) else float(current_official),
        )

        with st.expander(" | ".join(label_bits), expanded=False):
            render_performance_profile(
                performance_profile,
                show_current_official=True,
                compact=True,
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
                st.caption("Current race: " + " | ".join(current_bits))

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



def _render_race_prediction(
    active: pd.DataFrame,
    card: dict,
    prediction: dict | None = None,
):
    if active is None or active.empty or not _supabase_configured():
        return prediction

    distance = _racecard_float(card.get("distance"))
    if distance is None:
        return prediction

    if prediction is None:
        prediction = build_race_predictions(
            active,
            race_date=card.get("date") or card.get("dateFormat"),
            distance_m=distance,
            history_loader=load_horse_history,
        )

    rows = prediction.get("rows") if prediction else None
    st.markdown("### Race Edge Prediction")
    if rows is None or rows.empty:
        st.caption("No runners have enough saved Race Edge history to build a prediction yet.")
        return prediction

    st.caption(
        "Latest, Established and Peak are independent views of today's race. "
        "Margins are sequential and use 1 rating point = 0.5L. "
        "A 5-point gap starts a new ability group."
    )

    scenarios = prediction.get("scenarios", {})
    scenario_order = [
        ("Latest Form", "Latest Group"),
        ("Established Ability", "Established Group"),
        ("Peak Ability", "Peak Group"),
    ]

    display_columns = st.columns(3)
    for display_col, (scenario_name, group_col) in zip(display_columns, scenario_order):
        with display_col:
            st.markdown(f"#### {scenario_name}")
            scenario = scenarios.get(scenario_name)
            if scenario is None or scenario.empty:
                st.caption("No valid ratings.")
                continue

            for group_name, block in _scenario_group_blocks(scenario, group_col):
                st.caption(group_name.upper())
                for _, row in block.head(4).iterrows():
                    rank = int(row["Rank"])
                    runner = _runner_label(row.get("No."), row["Horse"])
                    if rank == 1:
                        st.markdown(f"**{rank}. {runner}**")
                    else:
                        st.markdown(
                            f"**{rank}. {runner}** - "
                            f"{_format_margin(row.get('Margin Behind Previous (L)'))}"
                        )

    st.markdown("#### Race Edge Consensus Top 4")
    consensus = prediction.get("consensus", [])
    if not consensus:
        st.caption("Not enough comparable Race Edge history for a consensus.")
    else:
        consensus_rows = []
        for item in consensus:
            rank_text = " / ".join(
                "-" if rank is None else str(rank)
                for rank in [
                    item.get("latest_rank"),
                    item.get("established_rank"),
                    item.get("peak_rank"),
                ]
            )
            group_text = " / ".join(
                _compact_group(group)
                for group in [
                    item.get("latest_group"),
                    item.get("established_group"),
                    item.get("peak_group"),
                ]
            )
            consensus_rows.append({
                "#": item["position"],
                "Runner": _runner_label(item.get("no"), item["horse"]),
                "L / E / P": rank_text,
                "Groups": group_text,
            })

        st.dataframe(
            pd.DataFrame(consensus_rows),
            width="stretch",
            hide_index=True,
        )

    with st.expander("Prediction Detail", expanded=False):
        detail_view = st.segmented_control(
            "Prediction View",
            ["Latest Form", "Established Ability", "Peak Ability"],
            default="Latest Form",
            key="race_card_prediction_detail_view",
        )

        detail_table = prediction_display_table(prediction)

        view_map = {
            "Latest Form": {
                "projection": "Latest Projection",
                "group": "Latest Group",
                "rank": "Latest Form Rank",
            },
            "Established Ability": {
                "projection": "Established Projection",
                "group": "Established Group",
                "rank": "Established Ability Rank",
            },
            "Peak Ability": {
                "projection": "Peak Projection",
                "group": "Peak Group",
                "rank": "Peak Ability Rank",
            },
        }

        selected = view_map.get(detail_view or "Latest Form", view_map["Latest Form"])
        projection_col = selected["projection"]
        group_col = selected["group"]
        rank_col = selected["rank"]

        # Show only the selected prediction view while keeping all three
        # scenarios calculated in the underlying prediction engine.
        wanted_cols = [
            c for c in ["No.", "Horse", "Current MR", projection_col, group_col]
            if c in detail_table.columns
        ]
        selected_table = detail_table[wanted_cols].copy()

        # Rank the audit table according to the selected scenario rather than
        # the combined consensus ordering.
        prediction_rows = prediction.get("rows")
        if (
            prediction_rows is not None
            and not prediction_rows.empty
            and "Horse" in selected_table.columns
            and rank_col in prediction_rows.columns
        ):
            rank_lookup = prediction_rows[["Horse", rank_col]].copy()
            selected_table = selected_table.merge(
                rank_lookup,
                on="Horse",
                how="left",
            )
            selected_table = selected_table.sort_values(
                [rank_col, "Horse"],
                ascending=[True, True],
                na_position="last",
            ).drop(columns=[rank_col])

        st.dataframe(
            selected_table.reset_index(drop=True),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"{detail_view or 'Latest Form'} projection shown. "
            "A new group starts at a 5-point gap from the current group leader."
        )

    return prediction


def _render_loaded_race_card(card: dict):
    """Render the Race Card as the race-day decision centre."""
    date_label = str(card.get("dateFormat") or card.get("date") or "-")
    track = str(card.get("clubName") or "-")
    race_no = _racecard_int(card.get("race"))
    time_label = str(card.get("time") or "-")
    distance = _racecard_int(card.get("distance"))
    surface = str(card.get("surfaceDescr") or "-")
    name = str(card.get("name") or "").strip()
    description = str(card.get("description") or "").strip()
    wfa_text = str(card.get("WFA") or "").strip()
    stake = str(card.get("stake") or "").strip()
    currency = str(card.get("currency") or "").strip()

    active_count = sum(
        1 for r in card.get("runners", [])
        if str(r.get("status") or "").strip().upper() != "R"
    )
    reserve_count = sum(
        1 for r in card.get("runners", [])
        if str(r.get("status") or "").strip().upper() == "R"
    )

    # Compact race header.
    st.markdown(
        f"### {track} | R{race_no if race_no is not None else '-'} | "
        f"{distance if distance is not None else '-'}m | {time_label}"
    )
    header_bits = [date_label, surface, f"{active_count} runners"]
    if reserve_count:
        header_bits.append(f"{reserve_count} reserve{'s' if reserve_count != 1 else ''}")
    if stake:
        try:
            header_bits.append(f"{currency}{int(float(str(stake).replace(',', ''))):,}")
        except Exception:
            header_bits.append(f"{currency}{stake}")
    st.caption(" | ".join(header_bits))
    if name:
        st.markdown(f"**{name}**")
    if description:
        st.caption(description)
    if wfa_text:
        st.caption(wfa_text)

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

    # Build prediction once and reuse it across card, prediction and profiles.
    prediction = None
    if _supabase_configured() and distance is not None:
        try:
            prediction = build_race_predictions(
                active,
                race_date=card.get("date") or card.get("dateFormat"),
                distance_m=distance,
                history_loader=load_horse_history,
            )
        except Exception as exc:
            st.warning(f"Race Edge prediction could not be built: {exc}")

    if db_error:
        st.warning(f"Race Edge database status could not be loaded: {db_error}")
    elif _supabase_configured():
        with_history = int((active["Race Edge Runs"].fillna(0) > 0).sum())
        st.caption(
            f"Race Edge coverage: {with_history}/{len(active)} active runners have saved history."
        )

    st.markdown("### Race Card")
    st.dataframe(
        _enhanced_racecard_table(active, prediction),
        width="stretch",
        hide_index=True,
    )
    st.caption("Groups = Latest / Established / Peak. A 5-point gap starts the next group.")

    if not reserves.empty:
        with st.expander(f"Reserves ({len(reserves)})", expanded=False):
            reserve_cols = [
                c for c in ["No.", "Horse", "Draw", "Age", "Weight", "Official MR", "Jockey", "Trainer"]
                if c in reserves.columns
            ]
            st.dataframe(reserves[reserve_cols], width="stretch", hide_index=True)

    _render_race_prediction(active, card, prediction=prediction)
    _render_racecard_runner_profiles(active, prediction=prediction)



def _session_load_meeting_race(meeting: dict, race_key: str):
    """Load one race from an already-fetched SAHR meeting into Race Edge."""
    card = race_to_race_edge_card(meeting, race_key)
    st.session_state["race_card_payload"] = card
    st.session_state["sahr_current_race_key"] = str(race_key)
    return card


def _render_race_strip(meeting: dict):
    """
    Formgrids-style one-tap race navigation.
    Shows a single row (or wrapped rows on narrow screens) of R1, R2, R3...
    and highlights the currently loaded race.
    """
    options = meeting_race_options(meeting)
    if not options:
        return

    race_keys = [str(key) for key, _label in options]

    current_key = str(st.session_state.get("sahr_current_race_key") or "")
    if current_key not in race_keys:
        payload = st.session_state.get("race_card_payload") or {}
        payload_race = payload.get("race")
        if payload_race is not None:
            for key in race_keys:
                try:
                    if int(key) == int(payload_race):
                        current_key = key
                        st.session_state["sahr_current_race_key"] = key
                        break
                except Exception:
                    pass

    st.markdown("**Race:**")

    # Keep the strip compact. Up to 10 races usually fits comfortably on iPad.
    # If there are more, wrap onto a second row.
    per_row = 10
    for start in range(0, len(race_keys), per_row):
        chunk = race_keys[start:start + per_row]
        cols = st.columns(len(chunk))
        for col, key in zip(cols, chunk):
            with col:
                try:
                    race_num = int(key)
                except Exception:
                    race_num = race_keys.index(key) + 1

                is_current = key == current_key
                if st.button(
                    str(race_num),
                    type="primary" if is_current else "secondary",
                    width="stretch",
                    key=f"sahr_race_strip_{key}",
                ):
                    _session_load_meeting_race(meeting, key)
                    st.rerun()


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
    if "sahr_current_race_key" not in st.session_state:
        st.session_state["sahr_current_race_key"] = None

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
            st.session_state["sahr_current_race_key"] = None

        if st.button("Find Meetings", type="primary", width="stretch", key="sahr_find_meetings"):
            try:
                found = get_meetings_for_date(race_date)
                st.session_state["sahr_available_meetings"] = found
                st.session_state["sahr_meeting"] = None
                st.session_state["race_card_payload"] = None
                st.session_state["sahr_current_race_key"] = None
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
                    st.session_state["sahr_current_race_key"] = None
                    st.success(str(meeting.get("heading") or "Meeting loaded."))

                    # Open Race 1 immediately after loading the meeting.
                    options = meeting_race_options(meeting)
                    if options:
                        first_key = str(options[0][0])
                        _session_load_meeting_race(meeting, first_key)
                        st.rerun()
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
                _render_race_strip(meeting)

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
            st.session_state["sahr_current_race_key"] = None
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
