"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_core_metrics(ctx):
    globals().update(ctx)
    if _view_is("Core Metrics"):
        st.markdown("## Sectional Metrics (PI + Core Sectionals + SRI + TOF)")

    if _view_is("Core Metrics"):
        GR_COL = metrics.attrs.get("GR_COL", "Grind")

        show_cols = [
            "Horse","Finish_Pos","PI",
            "F200_idx","tsSPI","Accel","Grind","Grind_CG",
            "RaceTime_s","TOF","TOF_Profile",
            "EARLY_idx","LATE_idx",
            "Peak_Speed","Peak_Location","SRI","SRI_Profile",
            "GrindAdjPts","DeltaG",
            "Sprint_Conversion_Penalty"
        ]

        # ---- make the column pick robust (no KeyError if some are missing) ----
        tmp = metrics.copy()
        for c in show_cols:
            if c not in tmp.columns:
                tmp[c] = np.nan
        display_df = tmp[show_cols].copy()

        # prefer sorting by finish as secondary key when present
        _finish_sort = pd.to_numeric(display_df["Finish_Pos"], errors="coerce").fillna(1e9)
        display_df = display_df.assign(_FinishSort=_finish_sort).sort_values(
            ["PI","_FinishSort"], ascending=[False, True]
        ).drop(columns=["_FinishSort"])

        st.dataframe(display_df, use_container_width=True)

        # Going note (PI only)
        pi_meta = metrics.attrs.get("PI_GOING_META", {})
        if pi_meta:
            g = str(pi_meta.get("going","Good"))
            n = int(pi_meta.get("field_n", len(display_df)))
            mult = pi_meta.get("multipliers", {})
            # Compact summary (only show components that actually moved)
            moved = [f"{k}x{mult[k]:.3f}" for k in ["Accel","F200_idx","tsSPI","Grind"] if abs(mult.get(k,1.0)-1.0) >= 0.005]
            if moved:
                st.caption(f"Going: {g} - PI weight multipliers: " + ", ".join(moved) + f" (field={n}).")

        render_rpss_section(RPSS_INFO)

        # GCI-based Race Class Summary removed.

        # ======================= Ahead of the Handicap - WFA Adjusted =======================
        st.markdown("## Ahead of the Handicap - WFA Adjusted")
        st.caption(
            "Select a line horse and assign its achieved MR. Race Edge uses each horse's age, "
            "the race date and distance, and the South African WFA scale to calculate the rest of the field. "
            "**1 lb = 0.5 kg** and **1 kg = 2 MR points**."
        )

        handicap_df = build_database_handicap(metrics, race_distance_input, work)
        if handicap_df.empty:
            st.info("Horse, PI and weight data are required before ratings can be calculated.")
        else:
            # Race date is read from the CSV metadata so the correct monthly WFA scale is used.
            _race_meta_source = work if isinstance(work, pd.DataFrame) else metrics
            _csv_date_raw = _first_present_value(_race_meta_source, ["Race Date", "Date", "Race_Date"])
            _race_date = _parse_race_date_value(_csv_date_raw, datetime.now().date())

            # Horse ages and Official MRs are auto-filled from the loaded CSV where available.
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
                key=f"core_aoh_age_editor_{_race_date.isoformat()}_{int(race_distance_input)}",
            )
            edited_ages["Age"] = pd.to_numeric(edited_ages["Age"], errors="coerce")
            missing_age_horses = edited_ages.loc[
                edited_ages["Age"].isna(), "Horse"
            ].astype(str).tolist()

            line_options = handicap_df["Horse"].astype(str).tolist()

            # Race Edge Suggested Line Horse:
            # 1) Sustain Residual closest to 0.00
            # 2) If tied, PI closest to 5.00
            _suggested_line_horse = None
            _suggested_residual = None
            _suggested_pi = None
            _line_plane, _line_profile = build_database_plane(metrics, RPSS_INFO)
            if not _line_plane.empty and "Sustain_Residual" in _line_plane.columns:
                _line_candidates = _line_plane[["Horse", "Sustain_Residual"]].copy()
                _line_candidates["Sustain_Residual"] = pd.to_numeric(
                    _line_candidates["Sustain_Residual"], errors="coerce"
                )

                _pi_lookup = metrics[["Horse", "PI"]].copy() if "PI" in metrics.columns else pd.DataFrame()
                if not _pi_lookup.empty:
                    _pi_lookup["PI"] = pd.to_numeric(_pi_lookup["PI"], errors="coerce")
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
                ].copy()

                if not _line_candidates.empty:
                    _line_candidates["_ResidualDistance"] = _line_candidates["Sustain_Residual"].abs()
                    _line_candidates["_PIDistance"] = (
                        pd.to_numeric(_line_candidates["PI"], errors="coerce") - 5.0
                    ).abs().fillna(np.inf)
                    _line_candidates = _line_candidates.sort_values(
                        ["_ResidualDistance", "_PIDistance"],
                        ascending=[True, True],
                        kind="stable",
                    )
                    _suggested_row = _line_candidates.iloc[0]
                    _suggested_line_horse = str(_suggested_row["Horse"])
                    _suggested_residual = float(_suggested_row["Sustain_Residual"])
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
                    key=f"core_aoh_line_horse_{_race_date.isoformat()}_{int(race_distance_input)}",
                )
                if _suggested_line_horse is not None:
                    _pi_text = "N/A" if _suggested_pi is None else f"{_suggested_pi:.2f}"
                    st.caption(
                        f"Suggested: {_suggested_line_horse} | "
                        f"Sustain Residual {_suggested_residual:+.2f} | PI {_pi_text}"
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
                    "Line Horse MR Achieved",
                    min_value=0,
                    max_value=200,
                    value=int(_line_mr_default),
                    step=1,
                    key=f"core_aoh_line_mr_{_race_date.isoformat()}_{int(race_distance_input)}_{canon_horse(line_horse)}",
                    help="Auto-filled from Official MR when available; otherwise enter the line rating manually.",
                )

            if missing_age_horses:
                preview = ", ".join(missing_age_horses[:5])
                more = "..." if len(missing_age_horses) > 5 else ""
                st.warning(
                    f"Enter an age for every horse before ratings can be calculated: {preview}{more}"
                )
            else:
                rating_df = handicap_df.merge(edited_ages, on="Horse", how="left")
                rating_df["Age"] = rating_df["Age"].astype(int)
                rating_df["WFA (lb)"] = rating_df["Age"].map(
                    lambda age: get_wfa_lb(_race_date, race_distance_input, int(age))
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
                rating_df["MR Achieved"] = rating_df["MR Achieved Raw"].map(_db_round_mr).astype("Int64")

                # Show Official MR alongside achieved MR so improvement is immediately visible.
                rating_df["Official MR"] = rating_df["Horse"].map(
                    lambda horse: _official_mr_lookup.get(canon_horse(horse))
                )
                rating_df["MR +/-"] = (
                    pd.to_numeric(rating_df["MR Achieved"], errors="coerce")
                    - pd.to_numeric(rating_df["Official MR"], errors="coerce")
                )

                band = wfa_distance_band(race_distance_input)
                st.info(
                    f"WFA scale applied: **{_race_date.strftime('%B')} | "
                    f"{_WFA_BAND_LABELS[band]}**. "
                    f"Line horse: **{line_horse}**, age **{int(line_row['Age'])}**, "
                    f"WFA **{line_row['WFA (lb)']:.0f} lb**."
                )

                rating_display = rating_df[[
                    "Horse", "Age", "PI", "Official MR", "MR Achieved", "MR +/-",
                ]].copy()
                for col in ["PI", "MR +/-"]:
                    rating_display[col] = pd.to_numeric(rating_display[col], errors="coerce").round(2)
                for col in ["Official MR", "MR Achieved"]:
                    rating_display[col] = pd.to_numeric(
                        rating_display[col], errors="coerce"
                    ).astype("Int64")

                # Keep the line-horse view easy to compare by ranking achieved MR highest first.
                rating_display = rating_display.sort_values(
                    ["MR Achieved", "PI"], ascending=[False, False], na_position="last"
                )
                st.dataframe(rating_display, width="stretch", hide_index=True)

                csv_bytes = rating_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download WFA-adjusted handicap table (CSV)",
                    csv_bytes,
                    file_name="ahead_of_handicap_wfa_adjusted.csv",
                    mime="text/csv",
                    width="stretch",
                )
        # ======================= /Ahead of the Handicap =======================
        # ======================= End of Batch 2 =======================

        # ======================= Batch 3 - Visuals + Hidden v2 + Ability v2 =======================
        from matplotlib.patches import Rectangle
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.lines import Line2D

        # ----------------------- Label repel (built-in fallback) -----------------------
        def _repel_labels_builtin(ax, x, y, labels, *, init_shift=0.18, k_attract=0.006, k_repel=0.012, max_iter=250):
            trans=ax.transData; renderer=ax.figure.canvas.get_renderer()
            xy=np.column_stack([x,y]).astype(float); offs=np.zeros_like(xy)
            for i,(xi,yi) in enumerate(xy):
                offs[i]=[init_shift if xi>=0 else -init_shift, init_shift if yi>=0 else -init_shift]
            texts,lines=[],[]
            for (xi,yi),(dx,dy),lab in zip(xy,offs,labels):
                t=ax.text(xi+dx, yi+dy, lab, fontsize=8.4, va="center", ha="left",
                          bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
                texts.append(t)
                ln=Line2D([xi,xi+dx],[yi,yi+dy], lw=0.75, color="black", alpha=0.9)
                ax.add_line(ln); lines.append(ln)
            inv=ax.transData.inverted()
            for _ in range(max_iter):
                moved=False
                bbs=[t.get_window_extent(renderer=renderer).expanded(1.02,1.15) for t in texts]
                for i in range(len(texts)):
                    for j in range(i+1,len(texts)):
                        if not bbs[i].overlaps(bbs[j]): continue
                        ci=((bbs[i].x0+bbs[i].x1)/2,(bbs[i].y0+bbs[i].y1)/2)
                        cj=((bbs[j].x0+bbs[j].x1)/2,(bbs[j].y0+bbs[j].y1)/2)
                        vx,vy=ci[0]-cj[0],ci[1]-cj[1]
                        if vx==0 and vy==0: vx=1.0
                        n=(vx**2+vy**2)**0.5; dx,dy=(vx/n)*k_repel*72,(vy/n)*k_repel*72
                        for t,s in ((texts[i],+1),(texts[j],-1)):
                            tx,ty=t.get_position()
                            px=trans.transform((tx,ty))+s*np.array([dx,dy])
                            t.set_position(inv.transform(px)); moved=True
                if not moved: break
            for t,ln,(xi,yi) in zip(texts,lines,xy):
                tx,ty=t.get_position(); ln.set_data([xi,tx],[yi,ty])

        def label_points_neatly(ax, x, y, names):
            try:
                from adjustText import adjust_text
                texts=[ax.text(xi,yi,nm,fontsize=8.4,
                               bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
                       for xi,yi,nm in zip(x,y,names)]
                adjust_text(texts, x=x, y=y, ax=ax,
                            only_move={'points':'y','text':'xy'},
                            force_points=0.6, force_text=0.7,
                            expand_text=(1.05,1.15), expand_points=(1.05,1.15),
                            arrowprops=dict(arrowstyle="->", lw=0.75, color="black", alpha=0.9,
                                            shrinkA=0, shrinkB=3))
            except Exception:
                _repel_labels_builtin(ax, x, y, names)
