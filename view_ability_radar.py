"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_ability_radar(ctx):
    globals().update(ctx)
    if _view_is("Ability Radar"):
        st.markdown("## Ability Radar")
        st.caption(
            "Compare horses from the same race using the core Race Edge indexed values. "
            "The radar uses F200, tsSPI, Accel and Grind on an 80–110 scale, where 100 is race-average/par."
        )

        radar_phases = ["F200", "tsSPI", "Accel", "Grind"]

        def _pick_numeric_col(df, candidates):
            """Return the first candidate column that exists and has at least one numeric value."""
            for c in candidates:
                if c in df.columns:
                    vals = pd.to_numeric(df[c], errors="coerce")
                    if vals.notna().any():
                        return c
            # If the column exists but is empty, still return it as a last resort so the warning is precise.
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        gr_attr = metrics.attrs.get("GR_COL", None)
        grind_candidates = []
        if gr_attr:
            grind_candidates.append(gr_attr)
        grind_candidates += ["Grind_CG", "Grind", "Grind_idx", "Corrected_Grind"]

        radar_cols = {
            "F200": _pick_numeric_col(metrics, ["F200_idx", "F200", "F200_Index", "F200 index", "F200_score"]),
            "tsSPI": _pick_numeric_col(metrics, ["tsSPI", "tsSPI_idx", "tsSPI_Index"]),
            "Accel": _pick_numeric_col(metrics, ["Accel", "Accel_idx", "Acceleration", "Acceleration_idx"]),
            "Grind": _pick_numeric_col(metrics, grind_candidates),
        }

        missing = [label for label, col in radar_cols.items() if col is None]
        if missing:
            st.warning(
                "Ability Radar needs usable F200, tsSPI, Accel and Grind columns. "
                f"Missing: {', '.join(missing)}"
            )
        else:
            radar_df = metrics.copy()
            radar_value_cols = []
            for phase, col in radar_cols.items():
                radar_col = f"{phase}_Radar"
                plot_col = f"{phase}_Plot"
                radar_df[radar_col] = pd.to_numeric(radar_df[col], errors="coerce")
                radar_df[plot_col] = radar_df[radar_col].clip(80, 110)
                radar_value_cols.append(radar_col)

            radar_numeric = radar_df[radar_value_cols].apply(pd.to_numeric, errors="coerce")
            valid_rows = radar_numeric.notna().any(axis=1)

            if not valid_rows.any():
                st.warning(
                    "Ability Radar could not find numeric values for F200, tsSPI, Accel or Grind in this race. "
                    "Check that the metrics table has been calculated before opening this module."
                )
            else:
                radar_df["Radar_Avg"] = radar_numeric.mean(axis=1, skipna=True)
                radar_df["Radar_Spread"] = radar_numeric.max(axis=1, skipna=True) - radar_numeric.min(axis=1, skipna=True)

                def _safe_main_weapon(row):
                    vals = row.dropna()
                    if vals.empty:
                        return "Unknown"
                    return str(vals.idxmax()).replace("_Radar", "")

                radar_df["Main Weapon"] = radar_numeric.apply(_safe_main_weapon, axis=1)

                def _radar_profile_100(row):
                    vals = {p: row.get(f"{p}_Radar") for p in radar_phases}
                    vals = {k: float(v) for k, v in vals.items() if pd.notna(v) and np.isfinite(float(v))}
                    if not vals:
                        return "Unknown"
                    f = vals.get("F200", np.nan)
                    t = vals.get("tsSPI", np.nan)
                    a = vals.get("Accel", np.nan)
                    g = vals.get("Grind", np.nan)
                    avg = float(np.mean(list(vals.values())))
                    spread = float(np.nanmax(list(vals.values())) - np.nanmin(list(vals.values())))
                    top_phase = max(vals, key=vals.get)

                    if avg >= 101.0 and min(vals.values()) >= 100.0:
                        return "Complete horse"
                    if np.isfinite(a) and np.isfinite(g) and a >= 101.0 and g >= 101.0:
                        return "Power finisher"
                    if np.isfinite(t) and np.isfinite(g) and t >= 101.0 and g >= 101.0:
                        return "Sustained galloper"
                    if np.isfinite(f) and np.isfinite(a) and f >= 101.0 and a >= 101.0:
                        return "Speed / tactical horse"
                    if top_phase == "Accel" and vals[top_phase] >= 100.5:
                        return "Turn-of-foot horse"
                    if top_phase == "Grind" and vals[top_phase] >= 100.5:
                        return "Grinder / sustainer"
                    if top_phase == "tsSPI" and vals[top_phase] >= 100.5:
                        return "Strong traveller"
                    if top_phase == "F200" and vals[top_phase] >= 100.5:
                        return "Early-speed horse"
                    if spread <= 1.0:
                        return "Balanced / neutral"
                    return "Mixed profile"

                radar_df["Profile"] = radar_df.apply(_radar_profile_100, axis=1)

                # Sensible default: top PI horses with usable radar values, otherwise first few usable horses.
                horse_list = radar_df["Horse"].astype(str).tolist() if "Horse" in radar_df.columns else []
                default_horses = []
                usable_radar_df = radar_df[valid_rows].copy()
                if "PI" in usable_radar_df.columns:
                    tmp = usable_radar_df.copy()
                    tmp["PI"] = pd.to_numeric(tmp["PI"], errors="coerce")
                    default_horses = tmp.sort_values("PI", ascending=False)["Horse"].astype(str).head(3).tolist()
                if not default_horses and "Horse" in usable_radar_df.columns:
                    default_horses = usable_radar_df["Horse"].astype(str).head(3).tolist()

                selected_horses = st.multiselect(
                    "Select horses to compare",
                    options=horse_list,
                    default=default_horses,
                    help="Choose 1–5 horses for a clean radar comparison."
                )
                if len(selected_horses) > 5:
                    st.warning("For readability, the radar chart shows the first 5 selected horses.")
                    selected_horses = selected_horses[:5]

                c1, c2, c3, c4 = st.columns(4)
                leaders = []
                for phase in radar_phases:
                    vals = pd.to_numeric(radar_df[f"{phase}_Radar"], errors="coerce")
                    if vals.notna().any():
                        idx = vals.idxmax()
                        leaders.append((phase, radar_df.loc[idx, "Horse"], float(vals.loc[idx])))
                    else:
                        leaders.append((phase, "No data", np.nan))
                for col, item in zip([c1, c2, c3, c4], leaders):
                    phase, horse, score = item
                    col.metric(f"Best {phase}", str(horse), "—" if not np.isfinite(score) else f"{score:.2f}")

                st.caption("Radar scale: 80–110. The 100 ring is race-average/par. Values outside 80–110 are clipped on the chart only; the table keeps the real values.")

                if not selected_horses:
                    st.info("Select at least one horse to draw the Ability Radar.")
                else:
                    plot_df = radar_df[radar_df["Horse"].astype(str).isin(selected_horses)].copy()
                    order_map = {h: i for i, h in enumerate(selected_horses)}
                    plot_df["_sel_order"] = plot_df["Horse"].astype(str).map(order_map)
                    plot_df = plot_df.sort_values("_sel_order")

                    try:
                        labels = radar_phases
                        n = len(labels)
                        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
                        angles += angles[:1]

                        fig, ax = plt.subplots(figsize=(7.8, 7.8), subplot_kw=dict(polar=True))
                        ax.set_theta_offset(np.pi / 2)
                        ax.set_theta_direction(-1)
                        ax.set_xticks(angles[:-1])
                        ax.set_xticklabels(labels, fontsize=11)
                        ax.set_ylim(80, 110)
                        ax.set_yticks([80, 90, 100, 105, 110])
                        ax.set_yticklabels(["80", "90", "100", "105", "110"], fontsize=8, alpha=0.78)
                        ax.grid(True, linestyle=":", alpha=0.45)

                        theta = np.linspace(0, 2 * np.pi, 240)
                        ax.plot(theta, np.full_like(theta, 100.0), linewidth=1.6, linestyle="--", alpha=0.65)

                        plotted_any = False
                        for _, row in plot_df.iterrows():
                            raw_values = [row.get(f"{p}_Radar", np.nan) for p in radar_phases]
                            if not pd.Series(raw_values).notna().any():
                                continue
                            # Use 100 as a neutral visual placeholder for a missing phase so one blank value doesn't break the chart.
                            values = []
                            for p in radar_phases:
                                v = row.get(f"{p}_Plot", np.nan)
                                values.append(100.0 if pd.isna(v) or not np.isfinite(float(v)) else float(v))
                            values += values[:1]
                            label = str(row.get("Horse", "Horse"))
                            ax.plot(angles, values, linewidth=2.2, marker="o", label=label)
                            ax.fill(angles, values, alpha=0.10)
                            plotted_any = True

                        if plotted_any:
                            ax.set_title("Ability Radar — indexed values", pad=24, fontsize=15)
                            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
                            st.pyplot(fig)
                        else:
                            st.info("The selected horses do not have enough numeric radar values to draw the chart.")
                    except Exception as e:
                        st.info(f"Ability Radar could not be drawn: {e}")

                    st.markdown("### Radar comparison table")
                    show_cols = ["Horse", "Finish_Pos"] if "Finish_Pos" in radar_df.columns else ["Horse"]
                    if "PI" in radar_df.columns:
                        show_cols.append("PI")
                    show_cols += [f"{p}_Radar" for p in radar_phases]
                    show_cols += ["Radar_Avg", "Radar_Spread", "Main Weapon", "Profile"]
                    view = plot_df[[c for c in show_cols if c in plot_df.columns]].copy()
                    rename_map = {f"{p}_Radar": p for p in radar_phases}
                    view = view.rename(columns=rename_map)
                    for c in ["F200", "tsSPI", "Accel", "Grind", "Radar_Avg", "Radar_Spread", "PI"]:
                        if c in view.columns:
                            view[c] = pd.to_numeric(view[c], errors="coerce").round(2)
                    st.dataframe(view, use_container_width=True, hide_index=True)

                st.markdown("### Full Ability Radar table")
                full_cols = ["Horse", "Finish_Pos"] if "Finish_Pos" in radar_df.columns else ["Horse"]
                if "PI" in radar_df.columns:
                    full_cols.append("PI")
                full_cols += [f"{p}_Radar" for p in radar_phases]
                full_cols += ["Radar_Avg", "Radar_Spread", "Main Weapon", "Profile"]
                full_view = radar_df[[c for c in full_cols if c in radar_df.columns]].copy().rename(columns={f"{p}_Radar": p for p in radar_phases})
                for c in ["F200", "tsSPI", "Accel", "Grind", "Radar_Avg", "Radar_Spread", "PI"]:
                    if c in full_view.columns:
                        full_view[c] = pd.to_numeric(full_view[c], errors="coerce").round(2)
                if "Radar_Avg" in full_view.columns:
                    full_view = full_view.sort_values("Radar_Avg", ascending=False, na_position="last").reset_index(drop=True)
                st.dataframe(full_view, use_container_width=True, hide_index=True)

                with st.expander("How to read Ability Radar"):
                    st.markdown(
                        """
    - **F200** = early/first-section ability.
    - **tsSPI** = sustained travelling strength.
    - **Accel** = ability to quicken when the race lifts.
    - **Grind** = ability to keep sustaining after the move.
    - **100** is the race average/par line.
    - **Above 100** means the horse was stronger than the race average in that phase.
    - **Below 100** means the horse was weaker than the race average in that phase.
    - The radar uses raw indexed metrics, so it works consistently with both 100m and 200m split files.
                        """
                    )
