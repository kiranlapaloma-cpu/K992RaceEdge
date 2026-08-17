"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_race_plane(ctx):
    globals().update(ctx)
    if _view_is("Race Plane Analysis", "Class Plane Analysis"):
        st.markdown("## Race Plane Analysis")
        st.caption(
            "PPS identifies the strongest field-relative overall performance across tsSPI, Accel and Grind, using PI's three-phase ratio while excluding the noisy opening block. "
            "Sustain Residual remains a separate signal of above- or below-plane sustain."
        )

        req = {"Horse", "tsSPI", "Accel", "Grind"}
        if not req.issubset(metrics.columns):
            st.warning("Race Plane Analysis needs Horse, tsSPI, Accel and Grind columns.")
        else:
            cpa1, cpa2, cpa3 = st.columns([1.1, 1.0, 1.0])
            with cpa1:
                grind_options = ["Grind"]
                if "Grind_CG" in metrics.columns:
                    grind_options.append("Grind_CG")
                plane_grind_col = st.selectbox(
                    "Grind target",
                    grind_options,
                    index=0,
                    help="Use raw Grind by default. Grind_CG is available if you want the plane to use corrected grind."
                )
            with cpa2:
                centre_values = st.toggle(
                    "Use centred values",
                    value=True,
                    help="Recommended. Fits tsSPI−100, Accel−100 and Grind−100 for cleaner coefficients."
                )
            with cpa3:
                show_3d_plane = st.toggle("Show 3D plane", value=True)

            # PPS and the plane deliberately exclude the noisy opening block.
            # The module uses only tsSPI, Accel and Grind.
            base_plane_cols = ["Horse", "tsSPI", "Accel", plane_grind_col]
            plane_df = metrics.loc[:, base_plane_cols].copy()
            extra_cols = [c for c in ["TOF", "PI", "SRI", "Peak_Location", "Finish_Pos"] if c in metrics.columns]
            for c in extra_cols:
                plane_df[c] = metrics[c]

            for c in ["tsSPI", "Accel", plane_grind_col, "TOF", "PI", "SRI", "Finish_Pos"]:
                if c in plane_df.columns:
                    plane_df[c] = pd.to_numeric(plane_df[c], errors="coerce")

            plane_df = plane_df.dropna(subset=["tsSPI", "Accel", plane_grind_col]).reset_index(drop=True)

            if len(plane_df) < 4:
                st.info("Need at least 4 runners with tsSPI, Accel and Grind to fit a stable plane.")
            else:
                if centre_values:
                    x = (plane_df["tsSPI"] - 100.0).to_numpy(dtype=float)
                    y = (plane_df["Accel"] - 100.0).to_numpy(dtype=float)
                    z = (plane_df[plane_grind_col] - 100.0).to_numpy(dtype=float)
                    x_label = "tsSPI − 100"
                    y_label = "Accel − 100"
                    z_label = f"{plane_grind_col} − 100"
                    formula_target = f"{plane_grind_col}Δ"
                else:
                    x = plane_df["tsSPI"].to_numpy(dtype=float)
                    y = plane_df["Accel"].to_numpy(dtype=float)
                    z = plane_df[plane_grind_col].to_numpy(dtype=float)
                    x_label = "tsSPI"
                    y_label = "Accel"
                    z_label = plane_grind_col
                    formula_target = plane_grind_col

                Xmat = np.column_stack([np.ones(len(plane_df)), x, y])
                coef, residuals, rank, singular_vals = np.linalg.lstsq(Xmat, z, rcond=None)
                intercept, b_tsspi, c_accel = [float(v) for v in coef]
                expected_z = Xmat @ coef

                if centre_values:
                    plane_df["Expected_Grind"] = 100.0 + expected_z
                    plane_df["Sustain_Residual"] = plane_df[plane_grind_col] - plane_df["Expected_Grind"]
                else:
                    plane_df["Expected_Grind"] = expected_z
                    plane_df["Sustain_Residual"] = plane_df[plane_grind_col] - plane_df["Expected_Grind"]

                z_mean = float(np.nanmean(z))
                ss_res = float(np.nansum((z - expected_z) ** 2))
                ss_tot = float(np.nansum((z - z_mean) ** 2))
                r2 = np.nan if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot

                def _cr_profile(v):
                    try:
                        v = float(v)
                    except Exception:
                        return "-"
                    if v >= 3.0:
                        return "🔥 Major above-plane"
                    if v >= 1.5:
                        return "🟢 Above expectation"
                    if v > -1.5:
                        return "⚪ Around expectation"
                    if v > -3.0:
                        return "🟠 Below expectation"
                    return "🔴 Emptied / weak sustain"

                plane_df["Sustain_Profile"] = plane_df["Sustain_Residual"].map(_cr_profile)
                plane_df["Expected_Grind"] = plane_df["Expected_Grind"].round(2)
                plane_df["Sustain_Residual"] = plane_df["Sustain_Residual"].round(2)

                # --- Race Test Profile: positive reward only; negative coefficients remain inverse relationships. ---
                race_test_profile = compute_race_test_profile(metrics, RPSS_INFO, plane_grind_col)
                travel_share = race_test_profile.get("travel_reward_share", np.nan)
                accel_share = race_test_profile.get("accel_reward_share", np.nan)
                residual_std = float(np.nanstd(plane_df["Sustain_Residual"].to_numpy(dtype=float), ddof=1)) if len(plane_df) > 1 else np.nan
                residual_range = float(np.nanmax(plane_df["Sustain_Residual"]) - np.nanmin(plane_df["Sustain_Residual"])) if len(plane_df) else np.nan

                st.markdown("### Race Plane Formula")
                st.code(
                    f"{formula_target} = {intercept:.3f} + ({b_tsspi:.3f} × {x_label}) + ({c_accel:.3f} × {y_label})",
                    language="text"
                )
                st.caption(
                    f"R² = {r2:.3f} · runners used = {len(plane_df)} · rank = {rank}. "
                    "Sustain Residual = Actual Grind − Expected Grind. Positive means the horse sustained better than the race plane predicted."
                )

                st.markdown("### Race Test Profile")
                profile_cols = st.columns(4)
                profile_cols[0].metric("Travel reward", "-" if not np.isfinite(travel_share) else f"{travel_share*100:.0f}%", f"coef {b_tsspi:+.2f}")
                profile_cols[1].metric("Acceleration reward", "-" if not np.isfinite(accel_share) else f"{accel_share*100:.0f}%", f"coef {c_accel:+.2f}")
                profile_cols[2].metric("Explainability", "-" if not np.isfinite(r2) else f"{r2*100:.0f}%", race_test_profile.get("confidence", "Low"))
                profile_cols[3].metric("Residual spread", "-" if not np.isfinite(residual_std) else f"{residual_std:.2f}", "std dev")

                st.info(
                    f"**{race_test_profile.get('label', 'Inconclusive race test')} — {race_test_profile.get('confidence', 'Low')} confidence.**  "
                    f"{race_test_profile.get('summary', '')}"
                )
                inverse_notes = []
                if b_tsspi < 0:
                    inverse_notes.append(f"tsSPI had an inverse relationship with Grind ({b_tsspi:+.3f})")
                if c_accel < 0:
                    inverse_notes.append(f"Accel had an inverse relationship with Grind ({c_accel:+.3f})")
                if inverse_notes:
                    st.caption("Inverse relationship detected: " + "; ".join(inverse_notes) + ". Negative coefficients are not counted as positive reward shares.")
                if rank < 3:
                    st.warning("The plane is not fully stable because the points are close to collinear. Treat residuals cautiously.")

                # --- Plane Position Score (PPS): field-relative overall performance using
                # the exact same phase weighting as PI. Sustain Residual remains separate.
                def _pps_robust_z(series):
                    s = pd.to_numeric(series, errors="coerce").astype(float)
                    med = float(np.nanmedian(s)) if np.isfinite(s).any() else 0.0
                    mad = float(np.nanmedian(np.abs(s - med))) if np.isfinite(s).any() else 0.0
                    if np.isfinite(mad) and mad > 1e-12:
                        out = (s - med) / (1.4826 * mad)
                    else:
                        sd = float(np.nanstd(s, ddof=0)) if np.isfinite(s).any() else 0.0
                        mu = float(np.nanmean(s)) if np.isfinite(s).any() else 0.0
                        out = (s - mu) / sd if np.isfinite(sd) and sd > 1e-12 else pd.Series(0.0, index=s.index)
                    return pd.Series(out, index=s.index, dtype=float).clip(-3.5, 3.5)

                plane_df["PPS_z_tsSPI"] = _pps_robust_z(plane_df["tsSPI"])
                plane_df["PPS_z_Accel"] = _pps_robust_z(plane_df["Accel"])
                plane_df["PPS_z_Grind"] = _pps_robust_z(plane_df[plane_grind_col])

                # PPS follows PI's tsSPI : Accel : Grind relationship, but deliberately
                # excludes F200/the opening block because that section is noisy. The
                # three retained PI weights are therefore normalised back to 100%.
                pi_phase_weights = dict(metrics.attrs.get("PI_PHASE_WEIGHTS", {}) or {})
                default_accel_ratio = 1.0 + 200.0 / max(float(race_distance_input), 1.0)
                raw_pps_weights = {
                    "tsSPI": float(pi_phase_weights.get("tsSPI", 1.0)),
                    "Accel": float(pi_phase_weights.get("Accel", default_accel_ratio)),
                    "Grind": float(pi_phase_weights.get("Grind", 1.0)),
                }
                raw_total = sum(max(0.0, v) for v in raw_pps_weights.values()) or 1.0
                pps_weights = {k: max(0.0, v) / raw_total for k, v in raw_pps_weights.items()}
                available_pps = {
                    "tsSPI": "PPS_z_tsSPI",
                    "Accel": "PPS_z_Accel",
                    "Grind": "PPS_z_Grind",
                }
                plane_df["PPS_Core"] = 0.0
                for phase, z_col in available_pps.items():
                    plane_df["PPS_Core"] += pps_weights[phase] * plane_df[z_col]
                plane_df["PPS"] = np.clip(
                    5.0 + 2.75 * np.tanh(plane_df["PPS_Core"] / 1.35),
                    0.0,
                    10.0,
                )
                plane_df["PPS"] = plane_df["PPS"].round(2)
                plane_df["PPS_Rank"] = plane_df["PPS"].rank(method="min", ascending=False).astype(int)

                pps_hi = float(np.nanquantile(plane_df["PPS"], 0.67))
                pps_lo = float(np.nanquantile(plane_df["PPS"], 0.33))
                sr_spread = float(np.nanstd(plane_df["Sustain_Residual"], ddof=1)) if len(plane_df) > 1 else 0.0
                sr_cut = max(1.0, 0.65 * sr_spread) if np.isfinite(sr_spread) else 1.0

                def _performance_architecture(row):
                    pps = float(row.get("PPS", np.nan))
                    sr = float(row.get("Sustain_Residual", np.nan))
                    if not (np.isfinite(pps) and np.isfinite(sr)):
                        return "Unclear profile"
                    if pps >= pps_hi and sr >= sr_cut:
                        return "Complete performance"
                    if pps >= pps_hi and sr <= -sr_cut:
                        return "Strong but incomplete"
                    if pps >= pps_hi:
                        return "Strong balanced run"
                    if pps <= pps_lo and sr >= sr_cut:
                        return "Honest sustain, limited level"
                    if pps <= pps_lo and sr <= -sr_cut:
                        return "Below benchmark"
                    if sr >= sr_cut:
                        return "Hidden finishing strength"
                    if sr <= -sr_cut:
                        return "Acceleration-led, weak sustain"
                    return "Balanced / expected"

                plane_df["Performance_Architecture"] = plane_df.apply(_performance_architecture, axis=1)

                st.markdown("### Performance Plane Rankings (PPS)")
                weight_note = (
                    f"tsSPI {pps_weights['tsSPI']*100:.1f}% · "
                    f"Accel {pps_weights['Accel']*100:.1f}% · Grind {pps_weights['Grind']*100:.1f}%"
                )
                st.caption(
                    f"PPS excludes the noisy opening block and uses PI's normalised three-phase ratio: {weight_note}. "
                    "Sustain Residual remains separate: it shows whether the horse sustained better or worse than its earlier effort predicted."
                )

                top_pps_row = plane_df.sort_values(["PPS", "PI" if "PI" in plane_df.columns else "PPS"], ascending=False).iloc[0]
                high_cr_row = plane_df.sort_values("Sustain_Residual", ascending=False).iloc[0]
                incomplete_pool = plane_df[
                    (plane_df["Sustain_Residual"] < -sr_cut) & (plane_df["PPS"] >= pps_hi)
                ].copy()
                pps_cards = st.columns(3)
                pps_cards[0].metric("Best Overall Performance", str(top_pps_row["Horse"]), f"PPS {float(top_pps_row['PPS']):.2f}")
                pps_cards[1].metric("Best Sustain Relative to Effort", str(high_cr_row["Horse"]), f"SR {float(high_cr_row['Sustain_Residual']):+.2f}")
                if len(incomplete_pool):
                    incomplete_row = incomplete_pool.sort_values("PPS", ascending=False).iloc[0]
                    pps_cards[2].metric(
                        "Strong but Incomplete",
                        str(incomplete_row["Horse"]),
                        f"PPS {float(incomplete_row['PPS']):.2f} · SR {float(incomplete_row['Sustain_Residual']):+.2f}",
                    )
                else:
                    pps_cards[2].metric("Strong but Incomplete", "None flagged", "No high-PPS negative sustain")

                out_cols = [
                    "PPS_Rank", "Horse", "Finish_Pos", "PPS", "PI",
                    "tsSPI", "Accel", plane_grind_col,
                    "Expected_Grind", "Sustain_Residual", "Sustain_Profile", "Performance_Architecture",
                ]
                out_cols = [c for c in out_cols if c in plane_df.columns]
                out_cols += [c for c in ["TOF", "SRI", "Peak_Location"] if c in plane_df.columns and c not in out_cols]
                rank_df = plane_df.sort_values(["PPS", "Sustain_Residual"], ascending=[False, False]).reset_index(drop=True)
                st.dataframe(rank_df[out_cols], use_container_width=True, hide_index=True)

                csv = rank_df[out_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Performance Plane table (CSV)",
                    data=csv,
                    file_name="performance_plane_pps.csv",
                    mime="text/csv"
                )

                if show_3d_plane:
                    st.markdown("### Interactive 3D Performance Plane")
                    try:
                        import plotly.graph_objects as go

                        view_col, horse_col, label_col = st.columns([1.15, 1.5, 1.0])
                        with view_col:
                            plane_view = st.selectbox(
                                "Plane view",
                                ["Performance View", "Sustain View", "Race Test View", "Top View"],
                                index=0,
                                help="Preset camera angles. You can still rotate the chart manually.",
                            )
                        with horse_col:
                            horse_choices = ["None"] + plane_df.sort_values("PPS", ascending=False)["Horse"].astype(str).tolist()
                            highlighted_horse = st.selectbox(
                                "Highlight horse",
                                horse_choices,
                                index=0,
                                help="Focus on one horse while keeping the full field visible.",
                            )
                        with label_col:
                            label_mode = st.selectbox(
                                "Labels",
                                ["Key horses", "All horses", "Hover only"],
                                index=0,
                            )

                        # Visual encodings only. The underlying Race Plane calculations are unchanged.
                        cr_vals = plane_df["Sustain_Residual"].to_numpy(dtype=float)
                        pps_vals = plane_df["PPS"].to_numpy(dtype=float)
                        horse_names = plane_df["Horse"].astype(str).to_numpy()
                        expected_plot_z = expected_z
                        actual_plot_z = z

                        sr_abs = float(np.nanmax(np.abs(cr_vals))) if np.isfinite(cr_vals).any() else 1.0
                        sr_abs = max(sr_abs, 1.0)
                        marker_sizes = 9.0 + 13.0 * np.clip((pps_vals - 3.0) / 5.0, 0.0, 1.0)

                        # Key labels: best PPS, best positive sustain, weakest sustain, winner, and selected horse.
                        key_names = set()
                        if len(plane_df):
                            key_names.add(str(plane_df.loc[plane_df["PPS"].idxmax(), "Horse"]))
                            key_names.add(str(plane_df.loc[plane_df["Sustain_Residual"].idxmax(), "Horse"]))
                            key_names.add(str(plane_df.loc[plane_df["Sustain_Residual"].idxmin(), "Horse"]))
                        if "Finish_Pos" in plane_df.columns:
                            winners = plane_df[pd.to_numeric(plane_df["Finish_Pos"], errors="coerce") == 1]
                            key_names.update(winners["Horse"].astype(str).tolist())
                        if highlighted_horse != "None":
                            key_names.add(highlighted_horse)

                        if label_mode == "All horses":
                            text_labels = horse_names.tolist()
                        elif label_mode == "Hover only":
                            text_labels = [""] * len(plane_df)
                        else:
                            text_labels = [name if name in key_names else "" for name in horse_names]

                        selected_mask = np.zeros(len(plane_df), dtype=bool)
                        line_widths = np.full(len(plane_df), 1.1, dtype=float)
                        if highlighted_horse != "None":
                            selected_mask = horse_names == highlighted_horse
                            marker_sizes[selected_mask] = marker_sizes[selected_mask] * 1.45
                            line_widths[selected_mask] = 3.0

                        fig3d = go.Figure()

                        # Neutral fitted plane.
                        x_grid = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 22)
                        y_grid = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 22)
                        Xg, Yg = np.meshgrid(x_grid, y_grid)
                        Zg = intercept + b_tsspi * Xg + c_accel * Yg
                        fig3d.add_trace(go.Surface(
                            x=Xg,
                            y=Yg,
                            z=Zg,
                            colorscale=[[0, "#41546d"], [1, "#7387a0"]],
                            showscale=False,
                            opacity=0.23,
                            hoverinfo="skip",
                            name="Expected Grind plane",
                        ))

                        # Residual lines and expected-position markers.
                        for i, row in plane_df.iterrows():
                            positive = float(row["Sustain_Residual"]) >= 0
                            line_colour = "#d5b45d" if positive else "#a95f5f"
                            line_alpha = 0.92 if highlighted_horse in ("None", str(row["Horse"])) else 0.22
                            fig3d.add_trace(go.Scatter3d(
                                x=[x[i], x[i]],
                                y=[y[i], y[i]],
                                z=[expected_plot_z[i], actual_plot_z[i]],
                                mode="lines",
                                line=dict(color=line_colour, width=float(line_widths[i]) + 1.0),
                                opacity=line_alpha,
                                hoverinfo="skip",
                                showlegend=False,
                            ))

                        fig3d.add_trace(go.Scatter3d(
                            x=x,
                            y=y,
                            z=expected_plot_z,
                            mode="markers",
                            marker=dict(size=4, color="#8392a6", opacity=0.48, symbol="circle"),
                            customdata=np.column_stack([horse_names, plane_df["Expected_Grind"].to_numpy(dtype=float)]),
                            hovertemplate="<b>%{customdata[0]}</b><br>Expected Grind: %{customdata[1]:.2f}<extra></extra>",
                            name="Expected Grind",
                        ))

                        finish_vals = plane_df["Finish_Pos"].fillna("").astype(str).to_numpy() if "Finish_Pos" in plane_df.columns else np.array([""] * len(plane_df))
                        pi_vals = plane_df["PI"].to_numpy(dtype=float) if "PI" in plane_df.columns else np.full(len(plane_df), np.nan)
                        architecture_vals = plane_df["Performance_Architecture"].astype(str).to_numpy()
                        customdata = list(zip(
                            horse_names.tolist(),
                            finish_vals.tolist(),
                            pi_vals.tolist(),
                            pps_vals.tolist(),
                            plane_df["tsSPI"].to_numpy(dtype=float).tolist(),
                            plane_df["Accel"].to_numpy(dtype=float).tolist(),
                            plane_df[plane_grind_col].to_numpy(dtype=float).tolist(),
                            plane_df["Expected_Grind"].to_numpy(dtype=float).tolist(),
                            cr_vals.tolist(),
                            architecture_vals.tolist(),
                        ))

                        def add_actual_performance_trace(indices, *, opacity, show_scale, trace_name):
                            """Add one supported Scatter3d trace with scalar opacity.

                            Plotly 3D markers do not accept per-point opacity arrays, so highlighted
                            and non-highlighted runners are deliberately rendered as separate traces.
                            """
                            indices = np.asarray(indices, dtype=int)
                            if indices.size == 0:
                                return
                            fig3d.add_trace(go.Scatter3d(
                                x=x[indices],
                                y=y[indices],
                                z=actual_plot_z[indices],
                                mode="markers+text",
                                text=[text_labels[i] for i in indices],
                                textposition="top center",
                                textfont=dict(color="#f4f6fa", size=11),
                                marker=dict(
                                    size=marker_sizes[indices],
                                    color=cr_vals[indices],
                                    colorscale=[
                                        [0.00, "#8e4e4e"],
                                        [0.42, "#c4cbd4"],
                                        [0.50, "#f2f4f7"],
                                        [0.58, "#d8c58a"],
                                        [1.00, "#c49a32"],
                                    ],
                                    cmin=-sr_abs,
                                    cmax=sr_abs,
                                    showscale=show_scale,
                                    colorbar=dict(
                                        title=dict(
                                            text="Sustain<br>Residual",
                                            font=dict(color="#e9edf3"),
                                        ),
                                        thickness=14,
                                        len=0.65,
                                        tickfont=dict(color="#c9d2de"),
                                        outlinecolor="#526174",
                                    ) if show_scale else None,
                                    line=dict(color="#f4f6fa", width=1.0),
                                    opacity=float(opacity),
                                ),
                                customdata=[customdata[i] for i in indices],
                                hovertemplate=(
                                    "<b>%{customdata[0]}</b><br>"
                                    "Finish: %{customdata[1]}<br>"
                                    "PI: %{customdata[2]:.2f}<br>"
                                    "PPS: %{customdata[3]:.2f}<br><br>"
                                    "tsSPI: %{customdata[4]:.2f}<br>"
                                    "Accel: %{customdata[5]:.2f}<br>"
                                    "Grind: %{customdata[6]:.2f}<br>"
                                    "Expected Grind: %{customdata[7]:.2f}<br>"
                                    "Sustain Residual: %{customdata[8]:+.2f}<br>"
                                    "Architecture: %{customdata[9]}"
                                    "<extra></extra>"
                                ),
                                name=trace_name,
                                showlegend=False,
                            ))

                        all_indices = np.arange(len(plane_df), dtype=int)
                        if highlighted_horse == "None":
                            add_actual_performance_trace(
                                all_indices,
                                opacity=1.0,
                                show_scale=True,
                                trace_name="Actual performance",
                            )
                        else:
                            dim_indices = all_indices[~selected_mask]
                            selected_indices = all_indices[selected_mask]
                            add_actual_performance_trace(
                                dim_indices,
                                opacity=0.30,
                                show_scale=True,
                                trace_name="Other runners",
                            )
                            add_actual_performance_trace(
                                selected_indices,
                                opacity=1.0,
                                show_scale=False,
                                trace_name="Highlighted runner",
                            )

                        cameras = {
                            "Performance View": dict(eye=dict(x=1.55, y=-1.65, z=1.15)),
                            "Sustain View": dict(eye=dict(x=1.10, y=-1.05, z=1.85)),
                            "Race Test View": dict(
                                eye=dict(
                                    x=1.85 if abs(b_tsspi) >= abs(c_accel) else 0.85,
                                    y=-0.85 if abs(b_tsspi) >= abs(c_accel) else -1.85,
                                    z=1.10,
                                )
                            ),
                            "Top View": dict(eye=dict(x=0.01, y=0.01, z=2.75)),
                        }

                        fig3d.update_layout(
                            height=760,
                            margin=dict(l=0, r=0, t=42, b=0),
                            paper_bgcolor="#0b1220",
                            plot_bgcolor="#0b1220",
                            font=dict(color="#e8edf4"),
                            title=dict(
                                text=f"{race_test_profile.get('label', 'Race Plane')} · R² {r2:.2f}",
                                x=0.02,
                                font=dict(size=18, color="#f6f7f9"),
                            ),
                            legend=dict(
                                orientation="h",
                                x=0.0,
                                y=1.02,
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#d6dee8"),
                            ),
                            scene=dict(
                                xaxis=dict(
                                    title="tsSPI — Travelling Strength" + (" (centred)" if centre_values else ""),
                                    backgroundcolor="#0b1220",
                                    gridcolor="rgba(160,175,195,0.16)",
                                    zerolinecolor="rgba(160,175,195,0.20)",
                                    color="#cfd7e2",
                                ),
                                yaxis=dict(
                                    title="Accel — Change of Speed" + (" (centred)" if centre_values else ""),
                                    backgroundcolor="#0b1220",
                                    gridcolor="rgba(160,175,195,0.16)",
                                    zerolinecolor="rgba(160,175,195,0.20)",
                                    color="#cfd7e2",
                                ),
                                zaxis=dict(
                                    title="Grind — Finishing Sustain" + (" (centred)" if centre_values else ""),
                                    backgroundcolor="#0b1220",
                                    gridcolor="rgba(160,175,195,0.16)",
                                    zerolinecolor="rgba(160,175,195,0.20)",
                                    color="#cfd7e2",
                                ),
                                camera=cameras[plane_view],
                                aspectmode="auto",
                            ),
                        )
                        st.plotly_chart(fig3d, width="stretch", config={"displaylogo": False, "scrollZoom": True})
                        st.caption(
                            "Marker size = PPS · Marker colour = Sustain Residual · Vertical line = Expected Grind to Actual Grind. "
                            "Gold indicates above-expected sustain; muted red indicates below-expected sustain."
                        )

                        if highlighted_horse != "None":
                            selected = plane_df[plane_df["Horse"].astype(str) == highlighted_horse].iloc[0]
                            selected_sr = float(selected["Sustain_Residual"])
                            selected_direction = "above" if selected_sr > 0 else "below" if selected_sr < 0 else "in line with"
                            st.info(
                                f"**{highlighted_horse}:** PPS {float(selected['PPS']):.2f} · "
                                f"Sustain Residual {selected_sr:+.2f}. The horse finished {selected_direction} the Grind level "
                                f"predicted from its travelling speed and acceleration. Profile: {selected['Performance_Architecture']}."
                            )

                        st.markdown("### Performance Architecture Map")
                        median_pps = float(np.nanmedian(plane_df["PPS"]))
                        fig2d = go.Figure()
                        fig2d.add_vline(x=median_pps, line_width=1, line_dash="dash", line_color="rgba(210,220,232,0.38)")
                        fig2d.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(210,220,232,0.38)")
                        def add_architecture_trace(indices, *, opacity, trace_name):
                            indices = np.asarray(indices, dtype=int)
                            if indices.size == 0:
                                return
                            fig2d.add_trace(go.Scatter(
                                x=plane_df["PPS"].to_numpy(dtype=float)[indices],
                                y=plane_df["Sustain_Residual"].to_numpy(dtype=float)[indices],
                                mode="markers+text",
                                text=[
                                    horse_names[i] if label_mode != "Hover only" and horse_names[i] in key_names else ""
                                    for i in indices
                                ],
                                textposition="top center",
                                marker=dict(
                                    size=(marker_sizes * 0.9)[indices],
                                    color=cr_vals[indices],
                                    colorscale=[[0.00, "#8e4e4e"], [0.50, "#f2f4f7"], [1.00, "#c49a32"]],
                                    cmin=-sr_abs,
                                    cmax=sr_abs,
                                    line=dict(color="#f4f6fa", width=1),
                                    showscale=False,
                                    opacity=float(opacity),
                                ),
                                customdata=[
                                    (horse_names[i], architecture_vals[i], finish_vals[i], pi_vals[i])
                                    for i in indices
                                ],
                                hovertemplate=(
                                    "<b>%{customdata[0]}</b><br>"
                                    "Finish: %{customdata[2]}<br>PI: %{customdata[3]:.2f}<br>"
                                    "PPS: %{x:.2f}<br>Sustain Residual: %{y:+.2f}<br>"
                                    "%{customdata[1]}<extra></extra>"
                                ),
                                name=trace_name,
                                showlegend=False,
                            ))

                        if highlighted_horse == "None":
                            add_architecture_trace(all_indices, opacity=1.0, trace_name="Field")
                        else:
                            add_architecture_trace(all_indices[~selected_mask], opacity=0.30, trace_name="Other runners")
                            add_architecture_trace(all_indices[selected_mask], opacity=1.0, trace_name="Highlighted runner")
                        fig2d.add_annotation(x=0.99, y=0.98, xref="paper", yref="paper", text="Complete / high-level sustain", showarrow=False, font=dict(color="#d8c58a", size=11), xanchor="right")
                        fig2d.add_annotation(x=0.99, y=0.04, xref="paper", yref="paper", text="Strong but incomplete", showarrow=False, font=dict(color="#b97676", size=11), xanchor="right")
                        fig2d.add_annotation(x=0.01, y=0.98, xref="paper", yref="paper", text="Hidden sustainer", showarrow=False, font=dict(color="#d8c58a", size=11), xanchor="left")
                        fig2d.add_annotation(x=0.01, y=0.04, xref="paper", yref="paper", text="Below race structure", showarrow=False, font=dict(color="#b97676", size=11), xanchor="left")
                        fig2d.update_layout(
                            height=470,
                            margin=dict(l=25, r=20, t=25, b=30),
                            paper_bgcolor="#0b1220",
                            plot_bgcolor="#0b1220",
                            font=dict(color="#e8edf4"),
                            xaxis=dict(title="PPS — Overall Three-Phase Performance", gridcolor="rgba(160,175,195,0.15)", zeroline=False),
                            yaxis=dict(title="Sustain Residual", gridcolor="rgba(160,175,195,0.15)", zeroline=False),
                        )
                        st.plotly_chart(fig2d, width="stretch", config={"displaylogo": False})
                    except Exception as e:
                        st.info(f"Interactive Race Plane could not be rendered: {e}")

                with st.expander("How to read this module"):
                    st.markdown(
                        """
    - **PPS:** field-relative overall performance strength using only tsSPI, Accel and Grind. It follows PI's ratio across those three phases, with the noisy opening block excluded.
    - **PPS Rank:** where the horse sits from strongest to weakest overall plane position.
    - **Race Plane Formula:** the race-specific expected relationship between sustained speed, acceleration and Grind.
    - **Expected Grind:** what the model predicts a horse should have produced from its tsSPI and Accel.
    - **Sustain Residual:** actual Grind minus expected Grind. It measures how well the horse completed the performance relative to its earlier travelling speed and acceleration.
    - **Performance Architecture:** combines PPS and Sustain Residual to describe whether the run was complete, balanced, hidden, acceleration-led or incomplete.
    - **Positive Sustain Residual:** the horse sustained better than expected.
    - **Negative Sustain Residual:** the horse did less late than its travel/acceleration profile suggested.

    - **Race Test Profile:** what the fitted plane positively rewarded. Reward shares use only positive coefficients; negative coefficients are shown separately as inverse relationships.
    - **Confidence:** based on the plane’s R² and regression stability. A low-confidence profile should not drive a strong conclusion.

    This is experimental. In small fields or unusual race shapes, use it as a guide rather than a final rating.
                        """
                    )
