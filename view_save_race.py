"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_save_race(ctx):
    globals().update(ctx)
    if _view_is("Save Race to Database"):
        st.title("Save Race to Database")
        st.caption("Calculate WFA-adjusted ratings and save or update the currently loaded race. Horse research is kept in the separate Horse Database module.")

        if not _supabase_configured():
            st.warning("Supabase is not configured in Streamlit Secrets.")
        else:
            db_plane, db_profile = build_database_plane(metrics, RPSS_INFO)
            handicap_df = build_database_handicap(metrics, race_distance_input, work)
            phase_notes_df = build_database_phase_notes(metrics)

            if db_plane.empty:
                st.info("At least four runners with tsSPI, Accel and Grind are required before this race can be saved.")
            elif handicap_df.empty:
                st.info("Horse, PI and weight data are required before ratings can be calculated.")
            else:
                st.markdown("### Race Details")
                _race_meta_source = work if isinstance(work, pd.DataFrame) else metrics
                _csv_date_raw = _first_present_value(_race_meta_source, ["Race Date", "Date", "Race_Date"])
                _csv_date = _parse_race_date_value(_csv_date_raw, datetime.now().date())
                _csv_track = str(_first_present_value(_race_meta_source, ["Track", "Racecourse", "Venue"], "Greyville")).strip().title()
                _csv_course = str(_first_present_value(_race_meta_source, ["Course"], "Turf")).strip().title()
                _csv_race_number = _first_present_value(_race_meta_source, ["Race Number", "Race_Number", "Race"], 1)
                try:
                    _csv_race_number = int(float(_csv_race_number))
                except Exception:
                    _csv_race_number = 1

                _track_options = ["Greyville", "Scottsville", "Turffontein", "Vaal", "Fairview", "Kenilworth", "Durbanville"]
                _course_options = ["Poly", "Turf", "Inside", "Standside", "Main", "Classic"]
                if _csv_track and _csv_track not in _track_options:
                    _track_options = [_csv_track] + _track_options
                if _csv_course and _csv_course not in _course_options:
                    _course_options = [_csv_course] + _course_options

                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    db_race_date = st.date_input(
                        "Race Date", value=_csv_date,
                        key=f"db_race_date_{_csv_date.isoformat()}_{int(race_distance_input)}",
                    )
                with d2:
                    db_track = st.selectbox(
                        "Track", _track_options,
                        index=_track_options.index(_csv_track) if _csv_track in _track_options else 0,
                        key=f"db_track_{canon_horse(_csv_track)}_{int(race_distance_input)}",
                    )
                with d3:
                    db_course = st.selectbox(
                        "Course", _course_options,
                        index=_course_options.index(_csv_course) if _csv_course in _course_options else 0,
                        key=f"db_course_{canon_horse(_csv_course)}_{int(race_distance_input)}",
                    )
                with d4:
                    db_race_number = st.number_input(
                        "Race Number", min_value=1, max_value=20, value=int(np.clip(_csv_race_number, 1, 20)), step=1,
                        key=f"db_race_number_{_csv_date.isoformat()}_{canon_horse(_csv_track)}",
                    )

                _autofilled = []
                if _csv_date_raw is not None: _autofilled.append("date")
                if _first_present_value(_race_meta_source, ["Track", "Racecourse", "Venue"]) is not None: _autofilled.append("track")
                if _first_present_value(_race_meta_source, ["Course"]) is not None: _autofilled.append("course")
                if _first_present_value(_race_meta_source, ["Race Number", "Race_Number", "Race"]) is not None: _autofilled.append("race number")
                if _autofilled:
                    st.caption("Auto-filled from CSV: " + ", ".join(_autofilled) + ". All fields remain editable.")

                rd1, rd2, rd3 = st.columns(3)
                with rd1:
                    st.number_input("Distance (m)", value=int(race_distance_input), disabled=True, key="db_distance")
                rpss_value = _db_num(RPSS_INFO.get("rpss")) if isinstance(RPSS_INFO, dict) else None
                with rd2:
                    st.text_input(
                        "RPSS",
                        value="—" if rpss_value is None else f"{rpss_value:.2f}",
                        disabled=True,
                        key="db_rpss",
                    )
                race_test_label = str(db_profile.get("label", "Inconclusive race test"))
                with rd3:
                    st.text_input("Race Test", value=race_test_label, disabled=True, key="db_race_test")

                st.markdown("### Ahead of the Handicap — WFA Adjusted")
                st.caption(
                    "Enter each horse's age, select the line horse and assign its achieved MR. "
                    "Race Edge reads the race month and distance, applies the South African WFA scale, "
                    "and calculates every MR automatically. **1 lb = 0.5 kg** and **1 kg = 2 MR points**."
                )

                _horse_meta = _horse_metadata_frame(work)
                age_input = handicap_df[["Horse"]].copy()
                age_input["Horse Key"] = age_input["Horse"].map(canon_horse)
                if not _horse_meta.empty:
                    age_input = age_input.merge(
                        _horse_meta[["Horse Key", "Age"]].drop_duplicates("Horse Key"),
                        on="Horse Key", how="left",
                    )
                else:
                    age_input["Age"] = pd.NA
                age_input["Age"] = pd.to_numeric(age_input["Age"], errors="coerce").astype("Int64")
                age_input = age_input[["Horse", "Age"]]
                age_editor_key = (
                    f"db_age_editor_{int(db_race_number)}_{db_race_date.isoformat()}_"
                    f"{int(race_distance_input)}_{int(age_input['Age'].notna().sum())}"
                )
                edited_ages = st.data_editor(
                    age_input,
                    width="stretch",
                    hide_index=True,
                    disabled=["Horse"],
                    column_config={
                        "Age": st.column_config.NumberColumn(
                            "Age", min_value=2, max_value=15, step=1, required=True
                        ),
                    },
                    key=age_editor_key,
                )
                edited_ages["Age"] = pd.to_numeric(edited_ages["Age"], errors="coerce")
                missing_age_horses = edited_ages.loc[edited_ages["Age"].isna(), "Horse"].astype(str).tolist()

                line_options = handicap_df["Horse"].astype(str).tolist()

                # Race Edge Suggested Line Horse:
                # 1) Eligible only when PI is between 4.9 and 5.9 inclusive.
                # 2) Among eligible horses, choose Sustain Residual closest to 0.00.
                # 3) If tied, choose PI closest to 5.00.
                _suggested_line_horse = None
                _suggested_residual = None
                _suggested_pi = None

                if not db_plane.empty and "Sustain_Residual" in db_plane.columns:
                    _line_candidates = db_plane[["Horse", "Sustain_Residual"]].copy()
                    _line_candidates["Sustain_Residual"] = pd.to_numeric(
                        _line_candidates["Sustain_Residual"], errors="coerce"
                    )

                    _pi_lookup = (
                        metrics[["Horse", "PI"]].copy()
                        if "PI" in metrics.columns
                        else pd.DataFrame()
                    )
                    if not _pi_lookup.empty:
                        _pi_lookup["PI"] = pd.to_numeric(
                            _pi_lookup["PI"], errors="coerce"
                        )
                        _line_candidates = _line_candidates.merge(
                            _pi_lookup.drop_duplicates("Horse"),
                            on="Horse",
                            how="left",
                        )
                    else:
                        _line_candidates["PI"] = np.nan

                    _line_candidates = _line_candidates[
                        _line_candidates["Horse"].astype(str).isin(line_options)
                        & _line_candidates["Sustain_Residual"].notna()
                        & _line_candidates["PI"].between(4.9, 5.9, inclusive="both")
                    ].copy()

                    if not _line_candidates.empty:
                        _line_candidates["_ResidualDistance"] = (
                            _line_candidates["Sustain_Residual"].abs()
                        )
                        _line_candidates["_PIDistance"] = (
                            _line_candidates["PI"] - 5.0
                        ).abs()
                        _line_candidates = _line_candidates.sort_values(
                            ["_ResidualDistance", "_PIDistance"],
                            ascending=[True, True],
                            kind="stable",
                        )

                        _suggested_row = _line_candidates.iloc[0]
                        _suggested_line_horse = str(_suggested_row["Horse"])
                        _suggested_residual = float(
                            _suggested_row["Sustain_Residual"]
                        )
                        _suggested_pi = _db_num(_suggested_row.get("PI"))

                _default_line_index = (
                    line_options.index(_suggested_line_horse)
                    if _suggested_line_horse in line_options
                    else 0
                )

                h1, h2 = st.columns(2)
                with h1:
                    line_horse = st.selectbox(
                        "Line Horse",
                        line_options,
                        index=_default_line_index,
                        key=f"db_line_horse_{db_race_date.isoformat()}_{int(db_race_number)}",
                    )
                    if _suggested_line_horse is not None:
                        _pi_text = "N/A" if _suggested_pi is None else f"{_suggested_pi:.2f}"
                        st.caption(
                            f"Suggested: {_suggested_line_horse} | "
                            f"Sustain Residual {_suggested_residual:+.2f} | PI {_pi_text}"
                        )
                    else:
                        st.caption(
                            "No suggested line horse: no runner has PI between 4.9 and 5.9 "
                            "with a valid Sustain Residual."
                        )
                _official_mr_lookup = {}
                if not _horse_meta.empty:
                    for _, _r in _horse_meta.drop_duplicates("Horse Key").iterrows():
                        _mr = _db_num(_r.get("Official MR"))
                        if _mr is not None:
                            _official_mr_lookup[str(_r.get("Horse Key"))] = _db_round_mr(_mr)
                _line_mr_default = _official_mr_lookup.get(canon_horse(line_horse), 100)
                with h2:
                    line_mr = st.number_input(
                        "Line Horse MR Achieved", min_value=0, max_value=200,
                        value=int(_line_mr_default), step=1,
                        key=f"db_line_mr_{db_race_date.isoformat()}_{int(db_race_number)}_{canon_horse(line_horse)}",
                        help="Auto-filled from Official MR when available; otherwise enter the line rating manually.",
                    )

                if missing_age_horses:
                    preview = ", ".join(missing_age_horses[:5])
                    more = "…" if len(missing_age_horses) > 5 else ""
                    st.warning(f"Enter an age for every horse before ratings can be calculated: {preview}{more}")
                    rating_df = pd.DataFrame()
                else:
                    rating_df = handicap_df.merge(edited_ages, on="Horse", how="left")
                    rating_df["Age"] = rating_df["Age"].astype(int)
                    rating_df["WFA (lb)"] = rating_df["Age"].map(
                        lambda age: get_wfa_lb(db_race_date, race_distance_input, int(age))
                    )
                    rating_df["WFA (kg)"] = rating_df["WFA (lb)"] * 0.5
                    rating_df["Effective Weight"] = rating_df["Weight (kg)"] + rating_df["WFA (kg)"]

                    line_row = rating_df.loc[
                        rating_df["Horse"].astype(str) == str(line_horse)
                    ].iloc[0]
                    line_perf_mr = float(line_row["Performance MR"])
                    line_effective_weight = float(line_row["Effective Weight"])

                    rating_df["Performance Difference"] = rating_df["Performance MR"] - line_perf_mr
                    rating_df["Weight + WFA Adjustment"] = 2.0 * (
                        rating_df["Effective Weight"] - line_effective_weight
                    )
                    rating_df["MR Achieved Raw"] = (
                        float(line_mr)
                        + rating_df["Performance Difference"]
                        + rating_df["Weight + WFA Adjustment"]
                    )
                    # Merit Ratings are whole numbers. Use conventional .5-up rounding
                    # for both display and database storage.
                    rating_df["MR Achieved"] = rating_df["MR Achieved Raw"].map(_db_round_mr).astype("Int64")

                    band = wfa_distance_band(race_distance_input)
                    st.info(
                        f"WFA scale applied: **{db_race_date.strftime('%B')} · "
                        f"{_WFA_BAND_LABELS[band]}**. "
                        f"Line horse: **{line_horse}**, age **{int(line_row['Age'])}**, "
                        f"WFA **{line_row['WFA (lb)']:.0f} lb**."
                    )

                    rating_display = rating_df[[
                        "Horse", "Age", "Weight (kg)", "WFA (lb)", "WFA (kg)",
                        "Effective Weight", "PI", "Performance Difference",
                        "Weight + WFA Adjustment", "MR Achieved",
                    ]].copy()
                    for col in [
                        "Weight (kg)", "WFA (lb)", "WFA (kg)", "Effective Weight",
                        "PI", "Performance Difference", "Weight + WFA Adjustment",
                    ]:
                        rating_display[col] = pd.to_numeric(rating_display[col], errors="coerce").round(2)
                    rating_display["MR Achieved"] = pd.to_numeric(
                        rating_display["MR Achieved"], errors="coerce"
                    ).astype("Int64")
                    st.dataframe(rating_display, width="stretch", hide_index=True)

                if rating_df.empty:
                    st.info("Complete the horse ages to unlock the database save table.")
                else:
                    # IMPORTANT: build the database save list from the full rated field,
                    # not from db_plane. A horse can legitimately be missing one of
                    # tsSPI / Accel / Grind and therefore have no Race Plane residual,
                    # but it should still be preserved in the historical database.
                    rating_save = rating_df[[
                        "Horse", "Age", "Weight (kg)", "WFA (lb)",
                        "Effective Weight", "MR Achieved"
                    ]].copy()
                    rating_save["Official MR"] = rating_save["Horse"].map(
                        lambda horse: _official_mr_lookup.get(canon_horse(horse))
                    )
                    save_df = rating_save.copy()

                    # Merge Race Plane information where it exists. Missing values are
                    # retained as NULL / Unavailable rather than dropping the horse.
                    plane_save = db_plane[["Horse", "Sustain_Residual", "Sustain_Verdict"]].copy()
                    save_df = save_df.merge(plane_save, on="Horse", how="left")
                    save_df["Sustain_Verdict"] = save_df["Sustain_Verdict"].fillna("Unavailable")

                    # Store finishing position together with the full rated field size,
                    # e.g. 1/11. This keeps the denominator correct even when a runner
                    # does not have enough phase data for the Race Plane calculation.
                    field_size = int(len(save_df))
                    finish_lookup = pd.DataFrame({"Horse": save_df["Horse"].astype(str)})
                    _finish_source = metrics if "Finish_Pos" in metrics.columns else work
                    _finish_col = next((c for c in ["Finish_Pos", "Finish Position", "Fin", "Position"] if c in _finish_source.columns), None)
                    if _finish_col is not None:
                        finish_lookup = _finish_source[["Horse", _finish_col]].copy().rename(columns={_finish_col: "Finish_Pos"})
                        finish_lookup["Horse"] = finish_lookup["Horse"].astype(str)
                        finish_lookup["Finish_Pos"] = pd.to_numeric(
                            finish_lookup["Finish_Pos"], errors="coerce"
                        ).astype("Int64")
                        finish_lookup["Finish Position"] = finish_lookup["Finish_Pos"].map(
                            lambda p: f"{int(p)}/{field_size}" if pd.notna(p) else f"—/{field_size}"
                        )
                        finish_lookup = finish_lookup[["Horse", "Finish Position"]]
                    else:
                        finish_lookup["Finish Position"] = f"—/{field_size}"

                    save_df = save_df.merge(finish_lookup, on="Horse", how="left")
                    save_df = save_df.merge(phase_notes_df, on="Horse", how="left")
                    save_df["Analyst Note"] = save_df["Phase Note"].fillna(
                        "F — | T — | A — | CG —"
                    ) + "\n\nNotes:\n"
                    save_df = save_df[[
                        "Horse", "Finish Position", "Age", "Weight (kg)", "WFA (lb)",
                        "Effective Weight", "Official MR", "MR Achieved", "Sustain_Residual",
                        "Sustain_Verdict", "Analyst Note",
                    ]].rename(columns={
                        "Sustain_Residual": "Sustain Residual",
                        "Sustain_Verdict": "Sustain Verdict",
                    })

                    editor_key = (
                        f"supabase_race_editor_{canon_horse(line_horse)}_"
                        f"{float(line_mr):.1f}_{int(db_race_number)}_wfa"
                    )
                    edited_db = st.data_editor(
                        save_df,
                        width="stretch",
                        hide_index=True,
                        disabled=[
                            "Horse", "Finish Position", "Age", "Weight (kg)", "WFA (lb)",
                            "Effective Weight", "Official MR", "MR Achieved", "Sustain Residual", "Sustain Verdict",
                        ],
                        column_config={
                            "Age": st.column_config.NumberColumn("Age", format="%d"),
                            "Weight (kg)": st.column_config.NumberColumn("Weight (kg)", format="%.1f"),
                            "WFA (lb)": st.column_config.NumberColumn("WFA (lb)", format="%.0f"),
                            "Effective Weight": st.column_config.NumberColumn(
                                "Effective Weight", format="%.1f"
                            ),
                            "Official MR": st.column_config.NumberColumn("Official MR", format="%d"),
                            "MR Achieved": st.column_config.NumberColumn("MR Achieved", format="%d"),
                            "Sustain Residual": st.column_config.NumberColumn(
                                "Sustain Residual", format="%+.2f"
                            ),
                            "Analyst Note": st.column_config.TextColumn("Analyst Note", width="large"),
                        },
                        key=editor_key,
                    )

                if not rating_df.empty and st.button("Save / Update Race in Database", type="primary", key="save_supabase_race"):
                    errors = []
                    if not str(db_track).strip():
                        errors.append("enter the track")
                    if not str(db_course).strip():
                        errors.append("enter the course")
                    if errors:
                        st.error("Please " + " and ".join(errors) + " before saving.")
                    else:
                        records = []
                        now_iso = datetime.now(timezone.utc).isoformat()
                        for _, row in edited_db.iterrows():
                            horse = canon_horse(str(row.get("Horse", "")))
                            if not horse:
                                continue
                            note = row.get("Analyst Note", "")
                            note = "" if note is None or pd.isna(note) else str(note).strip()
                            records.append({
                                "horse": horse,
                                "finish_position": str(row.get("Finish Position", "")).strip(),
                                "race_date": db_race_date.isoformat(),
                                "track": str(db_track).strip().title(),
                                "course": str(db_course).strip().title(),
                                "race_number": int(db_race_number),
                                "distance": int(race_distance_input),
                                "rpss": rpss_value,
                                "race_test": race_test_label,
                                "official_mr": _db_round_mr(row.get("Official MR")),
                                "mr_achieved": _db_round_mr(row.get("MR Achieved")),
                                "sustain_residual": _db_num(row.get("Sustain Residual")),
                                "sustain_verdict": str(row.get("Sustain Verdict", "")),
                                "analyst_note": note,
                                "updated_at": now_iso,
                            })
                        try:
                            count = save_horse_runs(records)
                            st.success(f"Saved or updated {count} horse runs in Supabase.")
                            if count != len(edited_db):
                                st.warning(
                                    f"Database save contained {count} records from {len(edited_db)} rows in the save table. "
                                    "Check for a blank horse name if those numbers differ."
                                )
                        except Exception as exc:
                            st.error(f"Database save failed: {exc}")
