"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_pace_curve(ctx):
    globals().update(ctx)
    if _view_is("Pace Curve"):
        # ======================= Pace Curve — enhanced detailed version =======================
        st.markdown("## Pace Curve")

        cpc1, cpc2, cpc3 = st.columns([1.15, 1.0, 1.0])
        with cpc1:
            pace_mode = st.selectbox("Curve mode", ["Raw Pace (m/s)", "Vs Field Average"], index=0, key="pace_curve_mode")
        with cpc2:
            runner_set = st.selectbox("Runners shown", ["Winner vs Field", "Top 4", "Top 8", "Top 10", "Whole field"], index=2, key="pace_curve_runner_set")
        with cpc3:
            show_phase_shading = st.toggle("Phase shading", value=True, key="pace_curve_phase_shading")

        show_end_labels = st.toggle("Line-end labels", value=True, key="pace_curve_end_labels")

        step = int(metrics.attrs.get("STEP", 100))
        D = float(race_distance_input)
        marks = _collect_markers(work)

        def _pace_x_label(v):
            try:
                vv = int(round(float(v)))
                return f"{vv}m" if vv > 0 else "FIN"
            except Exception:
                return str(v)

        # ---- Build segments in race order (early -> finish) ----
        segs = []  # (x_value, seg_len, time_col)
        if marks:
            m1 = int(marks[0])
            L0 = max(1.0, D - m1)
            if f"{m1}_Time" in work.columns:
                segs.append((m1, float(L0), f"{m1}_Time"))
            for a, b in zip(marks, marks[1:]):
                src = f"{int(b)}_Time"
                if src in work.columns:
                    segs.append((int(b), float(a - b), src))
        if "Finish_Time" in work.columns:
            segs.append((0, float(step), "Finish_Time"))

        if not segs:
            st.info("Not enough *_Time columns to draw the pace curve.")
        else:
            # ---- Compute per-horse segment speeds ----
            nseg = len(segs)
            arr = np.full((len(work), nseg), np.nan, dtype="float32")
            for j, (_, L, col) in enumerate(segs):
                if col in work.columns:
                    t = pd.to_numeric(work[col], errors="coerce").astype("float32")
                    t = np.where((t > 0) & np.isfinite(t), t, np.nan)
                    arr[:, j] = L / t

            field_avg = np.nanmean(arr, axis=0)
            if not np.isfinite(np.nanmean(field_avg)):
                st.info("Pace curve: all segments missing/invalid.")
            else:
                # ---- Choose runners ----
                if "Finish_Pos" in metrics.columns and metrics["Finish_Pos"].notna().any():
                    ranked = metrics.sort_values("Finish_Pos", ascending=True).copy()
                    ranking_rule = "Finish_Pos"
                elif "PI" in metrics.columns and metrics["PI"].notna().any():
                    ranked = metrics.sort_values("PI", ascending=False).copy()
                    ranking_rule = "PI"
                else:
                    ranked = metrics.copy()
                    ranking_rule = "table order"

                n_lookup = {"Winner vs Field": 1, "Top 4": 4, "Top 8": 8, "Top 10": 10, "Whole field": len(ranked)}
                topn = n_lookup.get(runner_set, 8)
                picked = ranked.head(topn).copy()
                if picked.empty:
                    st.info("No runners available for pace curve.")
                else:
                    x_vals = np.arange(nseg)
                    x_labels = [_pace_x_label(xv) for (xv, _, _) in segs]

                    picked_names = [str(x) for x in picked.get("Horse", pd.Series(dtype=str)).tolist()]
                    speed_map = {}
                    for _, r in picked.iterrows():
                        name = str(r.get("Horse", ""))
                        speeds = np.full(nseg, np.nan, dtype="float32")
                        for j, (_, L, col) in enumerate(segs):
                            t = pd.to_numeric(r.get(col, np.nan), errors="coerce")
                            if np.isfinite(t) and t > 0:
                                speeds[j] = L / float(t)
                        speed_map[name] = speeds

                    # ---- Display transform ----
                    disp_map = {}
                    if pace_mode == "Vs Field Average":
                        for nm, spd in speed_map.items():
                            disp_map[nm] = spd - field_avg
                        field_line = np.zeros_like(field_avg)
                        ylab = "Speed vs field avg (m/s)"
                        title_tail = "relative to field average"
                    else:
                        disp_map = speed_map
                        field_line = field_avg
                        ylab = "Speed (m/s)"
                        title_tail = "raw pace by segment"

                    # ---- Interactive Plotly chart (no Streamlit image endpoint) ----
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    # High-contrast runner colours: never use black or near-black.
                    palette = [
                        "#E63946", "#0077B6", "#16A34A", "#D97706",
                        "#9333EA", "#DB2777", "#0891B2", "#65A30D",
                        "#EA580C", "#4F46E5", "#0D9488", "#C026D3",
                        "#B45309", "#2563EB", "#DC2626", "#7C3AED",
                        "#15803D", "#BE185D", "#0369A1", "#A16207",
                        "#6D28D9", "#0F766E", "#C2410C", "#4338CA",
                    ]
                    winner_name = picked_names[0] if picked_names else None
                    top4_names = set(picked_names[:min(4, len(picked_names))])

                    # Shade the same early/sustain/acceleration/grind intervals.
                    if show_phase_shading and nseg >= 2:
                        seg_marks = [int(xv) for (xv, _, _) in segs]
                        idx_600 = seg_marks.index(600) if 600 in seg_marks else None
                        idx_200 = seg_marks.index(200) if 200 in seg_marks else None
                        idx_fin = seg_marks.index(0) if 0 in seg_marks else (nseg - 1)
                        bands = []
                        if idx_600 is not None and idx_600 > 0:
                            if idx_600 >= 2:
                                mid_cut = min(max(0, int(round(idx_600 * 0.55))), idx_600)
                                if mid_cut > 0:
                                    bands.append((-0.5, mid_cut - 0.5, "Early"))
                                if idx_600 > mid_cut:
                                    bands.append((mid_cut - 0.5, idx_600 - 0.5, "Sustain"))
                            else:
                                bands.append((-0.5, idx_600 - 0.5, "Early"))
                        elif idx_fin > 0:
                            bands.append((-0.5, idx_fin - 0.5, "Race"))
                        if idx_600 is not None and idx_200 is not None and idx_200 >= idx_600:
                            bands.append((idx_600 - 0.5, idx_200 + 0.5, "Acceleration"))
                        if idx_fin is not None:
                            bands.append((idx_fin - 0.5, idx_fin + 0.5, "Grind"))
                        for ib, (x0, x1, label) in enumerate(bands):
                            x0, x1 = max(-0.5, x0), min(nseg - 0.5, x1)
                            if x1 > x0:
                                fig.add_vrect(x0=x0, x1=x1,
                                    fillcolor="rgba(100,116,139,0.07)" if ib % 2 == 0 else "rgba(100,116,139,0.13)",
                                    line_width=0, layer="below")
                                fig.add_annotation(x=(x0+x1)/2, y=1, yref="paper", text=label,
                                    showarrow=False, yanchor="top", font=dict(size=10, color="#64748b"))

                    fig.add_trace(go.Scatter(
                        x=x_vals.tolist(), y=np.asarray(field_line, dtype=float).tolist(),
                        mode="lines+markers", name="Field average",
                        line=dict(color="#64748B", width=3, dash="dash"),
                        marker=dict(size=6), hovertemplate="%{y:.3f} m/s<extra>Field average</extra>"
                    ))
                    end_specs = []
                    for i, name in enumerate(picked_names):
                        y = disp_map.get(name)
                        if y is None or not np.any(np.isfinite(y)):
                            continue
                        is_winner = name == winner_name
                        is_top4 = name in top4_names
                        color = palette[i % len(palette)]
                        yplot = np.asarray(y, dtype=float)
                        finite = np.flatnonzero(np.isfinite(yplot))
                        # Plotly naturally leaves gaps at missing segments.
                        fig.add_trace(go.Scatter(
                            x=x_vals.tolist(), y=[float(v) if np.isfinite(v) else None for v in yplot],
                            mode="lines+markers", name=name,
                            line=dict(color=color, width=3.5 if is_winner else (2.5 if is_top4 else 1.5)),
                            marker=dict(size=7 if is_winner else (6 if is_top4 else 4)),
                            opacity=1 if is_winner else (0.9 if is_top4 else 0.65),
                            showlegend=is_top4,
                            hovertemplate="%{fullData.name}<br>%{x}: %{y:.3f} m/s<extra></extra>"
                        ))
                        if len(finite):
                            peak_idx = finite[np.argmax(yplot[finite])]
                            fig.add_trace(go.Scatter(
                                x=[int(peak_idx)], y=[float(yplot[peak_idx])], mode="markers",
                                marker=dict(size=10 if is_top4 else 8, color=color,
                                            line=dict(color="#FFFFFF", width=1)),
                                showlegend=False, hoverinfo="skip"
                            ))
                            if show_end_labels:
                                last = finite[-1]
                                end_specs.append((name, float(last), float(yplot[last]), color, is_winner))

                    # Non-overlapping labels placed to the right, with connector lines.
                    if show_end_labels and end_specs:
                        all_y = np.concatenate([np.asarray(field_line, dtype=float)] +
                                               [np.asarray(v, dtype=float) for v in disp_map.values()])
                        valid = all_y[np.isfinite(all_y)]
                        if len(valid):
                            lo, hi = float(valid.min()), float(valid.max())
                            span = max(hi-lo, 0.1)
                            gap = span * 0.055
                            ordered = sorted(end_specs, key=lambda t: t[2])
                            placed = []
                            for spec in ordered:
                                placed.append(max(spec[2], placed[-1] + gap) if placed else spec[2])
                            # Allow sufficient vertical space for crowded labels.
                            lower = min(lo - 0.05*span, placed[0])
                            upper = max(hi + 0.05*span, placed[-1] + 0.03*span)
                            fig.update_yaxes(range=[lower, upper])
                            for (name, xend, yend, color, is_winner), yl in zip(ordered, placed):
                                fig.add_shape(type="line", x0=xend, y0=yend,
                                              x1=nseg-1+0.17, y1=yl,
                                              line=dict(color=color, width=1))
                                fig.add_annotation(x=nseg-1+0.20, y=yl, text=name,
                                                   showarrow=False, xanchor="left", yanchor="middle",
                                                   font=dict(size=10, color=color,
                                                             family="Arial Black" if is_winner else "Arial"))

                    fig.update_layout(
                        title=f"Pace Curve — {title_tail}",
                        xaxis=dict(title="Race segment (distance remaining)",
                                   tickmode="array", tickvals=x_vals.tolist(), ticktext=x_labels,
                                   range=[-0.35, nseg-1+(2.3 if show_end_labels else 0.25)],
                                   showgrid=True, gridcolor="rgba(148,163,184,0.18)"),
                        yaxis=dict(title=ylab, showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=True),
                        template="plotly_white", height=610,
                        margin=dict(l=55, r=15, t=70, b=65),
                        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "responsive": True})

                    # caption / interpretation
                    try:
                        if pace_mode == "Raw Pace (m/s)":
                            late_slice = slice(max(0, nseg-3), nseg)
                            early_slice = slice(0, min(2, nseg))
                            field_early = float(np.nanmean(field_avg[early_slice]))
                            field_late = float(np.nanmean(field_avg[late_slice]))
                            race_shape = "quickened late" if field_late > field_early else "strong early, flatter late"
                            winner_line = disp_map.get(winner_name, np.full(nseg, np.nan)) if winner_name else np.full(nseg, np.nan)
                            winner_note = ""
                            if winner_name and np.any(np.isfinite(winner_line)) and np.any(np.isfinite(field_avg)):
                                wlate = float(np.nanmean(winner_line[late_slice]))
                                flate = float(np.nanmean(field_avg[late_slice]))
                                if wlate > flate:
                                    winner_note = f" {winner_name} finished above the field average late."
                                else:
                                    winner_note = f" {winner_name} tracked the race shape rather than clearly separating late."
                            st.caption(f"View: {runner_set}. Ranking basis: {ranking_rule}. Race shape looked {race_shape}.{winner_note}")
                        else:
                            st.caption(f"View: {runner_set}. Ranking basis: {ranking_rule}. Values above zero indicate segments run faster than the field average.")
                    except Exception:
                        st.caption(f"View: {runner_set}. Ranking basis: {ranking_rule}.")
