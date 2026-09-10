"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

from database import load_races_by_date, delete_saved_race

def render_horse_database_view(ctx):
    globals().update(ctx)
    if _view_is("Horse Database"):
        st.title("Horse Database")
        st.caption("Research saved horse histories or compare multiple horses. This module works without a race file loaded.")
        search_tab, compare_tab, manage_tab = st.tabs(["Horse Search", "Compare Horses", "Race Management"])
        with search_tab:
            render_horse_search()
        with compare_tab:
            render_horse_compare()
        with manage_tab:
            _render_race_management()


def _render_race_management():
    st.markdown("### Race Management")
    st.caption(
        "Look up every race saved on a date, review its runners, then delete the entire race from the database if needed."
    )

    if not _supabase_configured():
        st.warning("Supabase is not configured in Streamlit Secrets.")
        return

    lookup_date = st.date_input(
        "Race date", value=datetime.now().date(), key="db_manage_race_date"
    )

    try:
        day_df = load_races_by_date(lookup_date)
    except Exception as exc:
        st.error(f"Could not load saved races: {exc}")
        return

    if day_df.empty:
        st.info("No races are saved for this date.")
        return

    race_keys = []
    for (track, course, race_number), group in day_df.groupby(
        ["track", "course", "race_number"], dropna=False, sort=False
    ):
        distance_values = pd.to_numeric(group.get("distance"), errors="coerce").dropna()
        distance = int(distance_values.iloc[0]) if not distance_values.empty else None
        race_no = int(race_number) if pd.notna(race_number) else 0
        label = (
            f"{str(track)} | {str(course)} | Race {race_no}"
            + (f" | {distance}m" if distance is not None else "")
            + f" | {len(group)} runners"
        )
        race_keys.append({"label": label, "track": str(track), "course": str(course), "race_number": race_no})

    selected_label = st.selectbox(
        "Saved race",
        [r["label"] for r in race_keys],
        key=f"db_manage_race_select_{lookup_date.isoformat()}",
    )
    selected_key = next(r for r in race_keys if r["label"] == selected_label)

    race_df = day_df.loc[
        (day_df["track"].astype(str) == selected_key["track"])
        & (day_df["course"].astype(str) == selected_key["course"])
        & (pd.to_numeric(day_df["race_number"], errors="coerce") == selected_key["race_number"])
    ].copy()

    race_df["Official MR"] = pd.to_numeric(race_df.get("official_mr"), errors="coerce").astype("Int64")
    race_df["MR Achieved"] = pd.to_numeric(race_df.get("mr_achieved"), errors="coerce").astype("Int64")
    race_df["MR +/-"] = race_df["MR Achieved"] - race_df["Official MR"]
    display = race_df[["horse", "finish_position", "Official MR", "MR Achieved", "MR +/-"]].rename(
        columns={"horse": "Horse", "finish_position": "Finish"}
    )
    st.dataframe(display, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Delete Race")
    st.warning("Deleting a race permanently removes every saved runner in that race from Supabase.")
    key_base = f"{lookup_date.isoformat()}_{selected_key['track']}_{selected_key['course']}_{selected_key['race_number']}"
    confirm_delete = st.checkbox(
        "I understand this will permanently delete the entire selected race.",
        key=f"db_manage_delete_confirm_{key_base}",
    )
    if st.button(
        "Delete Selected Race", disabled=not confirm_delete, key=f"db_manage_delete_btn_{key_base}"
    ):
        try:
            count = delete_saved_race(
                lookup_date, selected_key["track"], selected_key["course"], selected_key["race_number"]
            )
            st.success(f"Deleted {count} saved horse runs from the selected race.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not delete this race: {exc}")
