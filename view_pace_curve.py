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

                    # ---- Plot ----
                    fig, ax = plt.subplots(figsize=(10.8, 6.4))

                    # phase shading: early / sustain / acceleration / grind
                    if show_phase_shading and nseg >= 2:
                        seg_marks = [int(xv) for (xv, _, _) in segs]
                        idx_600 = seg_marks.index(600) if 600 in seg_marks else None
                        idx_200 = seg_marks.index(200) if 200 in seg_marks else None
                        idx_fin = seg_marks.index(0) if 0 in seg_marks else (nseg - 1)

                        bands = []
                        # Everything before the acceleration phase
                        if idx_600 is not None and idx_600 > 0:
                            if idx_600 >= 2:
                                mid_cut = max(0, int(round(idx_600 * 0.55)))
                                mid_cut = min(mid_cut, idx_600)
                                if mid_cut > 0:
                                    bands.append((-0.5, mid_cut - 0.5, "Early"))
                                if idx_600 - 0.5 > mid_cut - 0.5:
                                    bands.append((mid_cut - 0.5, idx_600 - 0.5, "Sustain"))
                            else:
                                bands.append((-0.5, idx_600 - 0.5, "Early"))
                        elif idx_fin > 0:
                            bands.append((-0.5, idx_fin - 0.5, "Race"))

                        # User-defined pace phases
                        if idx_600 is not None and idx_200 is not None and idx_200 >= idx_600:
                            bands.append((idx_600 - 0.5, idx_200 + 0.5, "Acceleration"))
                        if idx_fin is not None:
                            bands.append((idx_fin - 0.5, idx_fin + 0.5, "Grind"))

                        for idx_b, (x0, x1, lab) in enumerate(bands):
                            x0 = max(-0.5, x0)
                            x1 = min(nseg - 0.5, x1)
                            if x1 > x0:
                                ax.axvspan(x0, x1, alpha=0.06 if idx_b % 2 == 0 else 0.10, color="grey")
                                ax.text((x0+x1)/2.0, 0.98, lab, transform=ax.get_xaxis_transform(),
                                        ha="center", va="top", fontsize=8)

                    # field average
                    ax.plot(x_vals, field_line, color="black", lw=2.8, marker="o", ms=4.2, label="Field average", zorder=4)

                    # runner lines
                    palette = color_cycle(len(picked))
                    winner_name = picked_names[0] if picked_names else None
                    top4_names = set(picked_names[:min(4, len(picked_names))])
                    end_label_specs = []

                    for i, name in enumerate(picked_names):
                        y = disp_map.get(name)
                        if y is None or not np.any(np.isfinite(y)):
                            continue
                        is_winner = (name == winner_name)
                        is_top4 = name in top4_names
                        lw = 2.8 if is_winner else (2.0 if is_top4 else 1.15)
                        alpha = 1.0 if is_winner else (0.92 if is_top4 else 0.58)
                        ms = 4.0 if is_winner else (3.2 if is_top4 else 2.4)
                        z = 6 if is_winner else (5 if is_top4 else 3)
                        ax.plot(x_vals, y, color=palette[i], lw=lw, alpha=alpha,
                                marker="o", ms=ms, label=name, zorder=z)

                        # mark strongest segment
                        finite = np.where(np.isfinite(y))[0]
                        if len(finite):
                            jj = finite[np.nanargmax(y[finite])]
                            ax.scatter([x_vals[jj]], [y[jj]], color=palette[i], s=28 if is_top4 else 20,
                                       edgecolor="black", linewidth=0.4, zorder=z+1)

                        # store end labels for all shown runners
                        if show_end_labels and len(finite):
                            jf = finite[-1]
                            end_label_specs.append({
                                "name": name,
                                "x": float(x_vals[jf]),
                                "y": float(y[jf]),
                                "color": palette[i],
                                "is_winner": is_winner,
                                "z": z,
                            })

                    # non-overlapping right-edge labels with leader lines
                    if show_end_labels and end_label_specs:
                        y0, y1 = ax.get_ylim()
                        yr = max(1e-9, y1 - y0)
                        min_gap = 0.055 * yr
                        label_x = len(x_vals) - 1 + 0.10

                        end_label_specs = sorted(end_label_specs, key=lambda d: d["y"])
                        placed_y = []
                        for d in end_label_specs:
                            yy = d["y"]
                            if placed_y:
                                yy = max(yy, placed_y[-1] + min_gap)
                            placed_y.append(yy)
                        # keep inside axes and back-propagate if needed
                        upper_cap = y1 - 0.03 * yr
                        lower_cap = y0 + 0.03 * yr
                        if placed_y:
                            if placed_y[-1] > upper_cap:
                                shift = placed_y[-1] - upper_cap
                                placed_y = [yy - shift for yy in placed_y]
                            if placed_y[0] < lower_cap:
                                shift = lower_cap - placed_y[0]
                                placed_y = [yy + shift for yy in placed_y]
                            for k in range(len(placed_y) - 2, -1, -1):
                                placed_y[k] = min(placed_y[k], placed_y[k + 1] - min_gap)
                            for k in range(1, len(placed_y)):
                                placed_y[k] = max(placed_y[k], placed_y[k - 1] + min_gap)

                        for d, yy in zip(end_label_specs, placed_y):
                            ax.plot([d["x"], label_x - 0.02], [d["y"], yy], color=d["color"], alpha=0.55,
                                    lw=1.0 if d["is_winner"] else 0.8, zorder=d["z"])
                            ax.text(label_x, yy, d["name"], fontsize=8.2 if d["is_winner"] else 7.8,
                                    fontweight="bold" if d["is_winner"] else "normal",
                                    color=d["color"], va="center", ha="left", zorder=d["z"] + 1,
                                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.7))

                    ax.set_xticks(x_vals)
                    ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
                    ax.set_ylabel(ylab)
                    ax.set_title(f"Pace Curve — {title_tail}")
                    ax.grid(True, ls="--", alpha=0.28)
                    ax.set_xlim(-0.35, len(x_vals)-1 + (1.30 if (show_end_labels and len(picked_names) > 12) else (1.05 if show_end_labels else 0.25)))

                    # cleaner legend: just field avg + winner + top placers when available
                    handles, labels = ax.get_legend_handles_labels()
                    keep_labels = ["Field average"]
                    if winner_name:
                        keep_labels.append(winner_name)
                    keep_labels.extend([nm for nm in picked_names[1:min(4, len(picked_names))]])
                    keep = [(h, l) for h, l in zip(handles, labels) if l in keep_labels]
                    if keep:
                        ax.legend([h for h, _ in keep], [l for _, l in keep],
                                  loc="upper center", bbox_to_anchor=(0.5, -0.14),
                                  ncol=min(4, len(keep)), frameon=False, fontsize=8)

                    st.pyplot(fig)

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

                    plt.close(fig)
