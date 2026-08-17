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
            moved = [f"{k}×{mult[k]:.3f}" for k in ["Accel","F200_idx","tsSPI","Grind"] if abs(mult.get(k,1.0)-1.0) >= 0.005]
            if moved:
                st.caption(f"Going: {g} — PI weight multipliers: " + ", ".join(moved) + f" (field={n}).")

        render_rpss_section(RPSS_INFO)

        # GCI-based Race Class Summary removed.

        # ======================= Ahead of the Handicap — One-Run Weight Intelligence =======================
        st.markdown("## Ahead of the Handicap — One-Run (Race-local)")

        # ---- Safe helpers (local, conflict-free) ----
        def _as_num(s):
            return pd.to_numeric(s, errors="coerce")

        def _field_corr(x, y):
            x = _as_num(x); y = _as_num(y)
            df = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(df) < 6:  # tiny fields → correlation not reliable
                return np.nan
            c = df["x"].corr(df["y"])
            try:
                return float(c) if np.isfinite(c) else np.nan
            except Exception:
                return np.nan

        def _beta_base(distance_m: float) -> float:
            """Baseline PI-per-kg by trip; conservative, race-agnostic."""
            d = float(distance_m)
            if d <= 1200: return 0.30
            if d <= 1600: return 0.35
            if d <= 2000: return 0.40
            if d <= 2400: return 0.45
            return 0.50

        def _shape_adjust(beta: float, rsi: float, sci: float) -> float:
            """Light-touch modulation by race shape (one-run safe)."""
            if np.isfinite(rsi) and np.isfinite(sci) and sci >= 0.5:
                if rsi < -0.6:  # fast-early
                    beta *= 1.10
                elif rsi > 0.6: # slow-early
                    beta *= 0.90
            return beta

        # ---- Inputs / guards ----
        # ---------------- Safe Horse Weight Resolver ----------------
        weight_col_candidates = ["Horse Weight", "Horse_Weight", "Wt", "Weight", "Weight (kg)"]
        weight_col = next((c for c in weight_col_candidates if c in metrics.columns), None)

        # if still missing, create it directly before loc[]
        if weight_col is None:
            st.warning("No weight column found — assigning 60 kg baseline for all horses.")
            metrics["Horse Weight"] = 60.0
            weight_col = "Horse Weight"
        else:
            metrics["Horse Weight"] = pd.to_numeric(metrics[weight_col], errors="coerce").fillna(60.0)
            weight_col = "Horse Weight"  # standardize name
        # ------------------------------------------------------------

        # now this will never KeyError
        df = metrics.loc[:, ["Horse", "PI", weight_col]].copy()

        need_cols = {"Horse", "PI"}
        if weight_col: need_cols.add(weight_col)
        missing = [c for c in need_cols if c not in metrics.columns]

        if missing:
            st.warning("Ahead of the Handicap: missing columns → " + ", ".join(missing))
        else:
            df = metrics.loc[:, ["Horse", "PI", weight_col]].copy()
            df["PI"] = _as_num(df["PI"])
            df[weight_col] = _as_num(df[weight_col])
            df = df.dropna(subset=["PI", weight_col])
            if df.empty:
                st.info("No valid PI/weight rows to evaluate.")
            else:
                # Distance & shape inputs (safe defaults)
                D_m  = float(race_distance_input)
                RSI  = float(metrics.attrs.get("RSI", np.nan))
                SCI  = float(metrics.attrs.get("SCI", np.nan))

                # 1) Baseline slope (PI per kg)
                beta0 = _beta_base(D_m)

                # 2) Race-local realism: how much did weight correlate with performance here?
                #    Use magnitude of correlation as a realism knob, damped for tiny fields.
                corr_w_pi = _field_corr(df[weight_col], df["PI"])
                n = df["PI"].notna().sum()
                tiny_dampen = 0.0 if n < 6 else min(1.0, (n - 5) / 7.0)  # ramps in from n=6 to ~n=12
                corr_mag = 0.0 if not np.isfinite(corr_w_pi) else abs(corr_w_pi)

                # Blend: mostly baseline, plus up to +40% from race-local signal (scaled by field size)
                beta_local = beta0 * (1.0 + 0.40 * corr_mag * tiny_dampen)

                # 3) Shape modulation (light touch)
                beta_eff = _shape_adjust(beta_local, RSI, SCI)

                # 4) Safety rails (avoid crazy kg if β too tiny or huge)
                beta_eff = float(np.clip(beta_eff, 0.22, 0.70))  # keep within realistic PI/kg bounds

                # 5) Convert each horse’s PI to ΔPI vs field median, then to kg and MR.
                #    PI itself remains unchanged. Only the PI-to-weight/MR conversion is
                #    confidence-adjusted in fields with fewer than 12 valid runners.
                PI_med = float(np.nanmedian(df["PI"]))
                df["ΔPI_vs_med"] = df["PI"] - PI_med

                field_size = int(df["PI"].notna().sum())
                field_conversion_factor = {
                    7: 0.68,
                    8: 0.76,
                    9: 0.84,
                    10: 0.90,
                    11: 0.95,
                }.get(field_size, 1.00 if field_size >= 12 else 0.60)

                raw_ran_above_kg = df["ΔPI_vs_med"] / beta_eff
                df["RanAbove_kg"] = raw_ran_above_kg * field_conversion_factor
                df["RanAbove_MR"] = df["RanAbove_kg"] * 2
        
                # 6) Friendly view
                view = df.copy()
                view = view.rename(columns={
                    weight_col: "Wt (kg)"
                })
                view["β_eff (PI/kg)"] = beta_eff
                view = view[["Horse", "Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg", "RanAbove_MR", "β_eff (PI/kg)"]]
                view = view.sort_values("RanAbove_kg", ascending=False)

                # Round for display only (keep raw in df if you need later)
                for c in ["Wt (kg)", "PI", "ΔPI_vs_med", "RanAbove_kg", "RanAbove_MR", "β_eff (PI/kg)"]:
                    view[c] = pd.to_numeric(view[c], errors="coerce").round(2)

                st.dataframe(view, use_container_width=True)

                # Key readout + tiny legend
                colA, colB, colC = st.columns([1,1,1])
                with colA:
                    st.metric("Field median PI", f"{PI_med:.2f}")
                with colB:
                    corr_str = "n/a" if not np.isfinite(corr_w_pi) else f"{corr_w_pi:+.2f}"
                    st.metric("Weight↔PI correlation (|r| used)", corr_str)
                with colC:
                    st.metric("β_eff (this race)", f"{beta_eff:.2f} PI per kg")

                if field_size < 12:
                    st.caption(
                        f"Small-field conversion adjustment: {field_size} valid runners → "
                        f"{field_conversion_factor:.0%} of the raw PI-to-kg/MR conversion. PI scores are unchanged."
                    )

                st.caption(
                    "Interpretation: **RanAbove (kg)** estimates how many kilograms a horse effectively ran above/below the "
                    "field median, *within this single race*. Positive = ran as if it could carry more and still match median. "
                    "Slope (β_eff) is distance-based, gently adjusted by (i) how weight correlated with PI in this field and "
                    "(ii) race shape. For fields below 12 valid runners, only the kg/MR conversion is reduced to reflect the "
                    "smaller dataset; the underlying PI scores and rankings remain unchanged."
                )

                # Optional CSV download (small footprint)
                csv_bytes = view.to_csv(index=False).encode("utf-8")
                st.download_button("Download Ahead-of-Handicap table (CSV)", csv_bytes,
                                   file_name="ahead_of_handicap_one_run.csv", mime="text/csv", use_container_width=True)
        # ======================= /Ahead of the Handicap =======================
        # ======================= End of Batch 2 =======================

        # ======================= Batch 3 — Visuals + Hidden v2 + Ability v2 =======================
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
