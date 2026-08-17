"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_advanced_models(ctx):
    globals().update(ctx)
    if _view_is("Advanced Models"):
        st.markdown("## 🧠 Race Intelligence")
        st.caption(
            "Combines RPSS, the Race Plane slope and horse-level phase evidence to explain what the race tested, "
            "which horses were suited by that test and what the performance may mean next time. No rating calculation is altered."
        )

        intelligence_profile = compute_race_test_profile(metrics, RPSS_INFO, "Grind")
        verdict_view = compute_race_shape_verdict(metrics, RPSS_INFO, intelligence_profile)
        if verdict_view.empty:
            st.info("Race Intelligence needs usable tsSPI, Accel and Grind metrics and could not be generated for this race.")
        else:
            _rp = verdict_view.attrs.get("rpss", np.nan)
            _ctx = verdict_view.attrs.get("context", "Unknown")
            _rp_txt = f"{float(_rp):.2f}" if np.isfinite(_rp) else "unavailable"
            st.markdown("### Race Test Profile")
            test_cols = st.columns(4)
            _tr = intelligence_profile.get("travel_reward_share", np.nan)
            _ar = intelligence_profile.get("accel_reward_share", np.nan)
            _r2 = intelligence_profile.get("r2", np.nan)
            test_cols[0].metric("Race test", intelligence_profile.get("label", "Inconclusive"))
            test_cols[1].metric("Travel reward", "-" if not np.isfinite(_tr) else f"{_tr*100:.0f}%")
            test_cols[2].metric("Acceleration reward", "-" if not np.isfinite(_ar) else f"{_ar*100:.0f}%")
            test_cols[3].metric("Confidence", intelligence_profile.get("confidence", "Low"), "-" if not np.isfinite(_r2) else f"R² {_r2:.2f}")
            st.info(
                f"**Race context: {_ctx} • RPSS: {_rp_txt}.**  "
                f"{intelligence_profile.get('summary', '')}"
            )

            st.markdown("### Horse Verdicts")
            horse_options = verdict_view["Horse"].astype(str).tolist()
            default_horses = horse_options[:min(6, len(horse_options))]
            selected_horses = st.multiselect(
                "Horses to display",
                options=horse_options,
                default=default_horses,
                help="The verdict is calculated for every runner. Select the horses you want to review in detail."
            )

            selected_view = verdict_view[verdict_view["Horse"].astype(str).isin(selected_horses)]
            for _, vr in selected_view.iterrows():
                st.markdown(f"### {vr['Horse']} — {vr['Verdict']}")
                meta = []
                if pd.notna(vr.get("Finish")):
                    meta.append(f"Finish {int(vr['Finish'])}")
                if pd.notna(vr.get("PI")):
                    meta.append(f"PI {float(vr['PI']):.2f}")
                meta.append(f"Confidence: {vr['Confidence']}")
                st.caption(" • ".join(meta))
                st.write(vr["Narrative"])
                st.markdown(f"**Race Edge action:** {vr['Action']}")

            st.markdown("### All-runner Race Intelligence table")
            table_cols = ["Horse", "Finish", "PI", "Verdict", "Confidence", "Action"]
            verdict_table = verdict_view[table_cols].copy()
            verdict_table = verdict_table.rename(columns={"Finish": "Pos"})
            st.dataframe(
                verdict_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Pos": st.column_config.NumberColumn("Pos", format="%d"),
                    "PI": st.column_config.NumberColumn("PI", format="%.2f"),
                },
            )

            with st.expander("How Race Intelligence works"):
                st.markdown(
                    """
    - **RPSS** establishes whether the race was slow, even or fast.
    - **Race Test Profile** uses the fitted plane coefficients and R² to identify an acceleration-led, sustained-pressure, balanced or inconclusive test.
    - Positive reward percentages use only positive coefficients. Negative coefficients remain visible as inverse relationships and are never presented as positive reward.
    - Each horse is then interpreted relative to that race test using **tsSPI, Accel, Grind/Corrected Grind and PI quality**.
    - The module deliberately ignores position, positional gains and draw.
    - The narrative is an interpretation of this single performance, not a permanent statement about the horse.
                    """
                )

        st.divider()
        # ======================= Hidden Horses =======================
        st.markdown("## Hidden Horses v2 (Shape-aware)")

        hh = metrics.copy()
        gr_col = metrics.attrs.get("GR_COL", "Grind")

        # --- SOS (robust z-score blend) ---
        need_cols = {"tsSPI", "Accel", gr_col}
        if need_cols.issubset(hh.columns) and len(hh) > 0:
            ts_w = winsorize(pd.to_numeric(hh["tsSPI"], errors="coerce"))
            ac_w = winsorize(pd.to_numeric(hh["Accel"], errors="coerce"))
            gr_w = winsorize(pd.to_numeric(hh[gr_col], errors="coerce"))

            def rz(s):
                mu, sd = np.nanmedian(s), mad_std(s)
                return (s - mu) / (sd if np.isfinite(sd) and sd > 0 else 1.0)

            z_ts, z_ac, z_gr = rz(ts_w), rz(ac_w), rz(gr_w)
            hh["SOS_raw"] = 0.45*z_ts + 0.35*z_ac + 0.20*z_gr
            q5, q95 = hh["SOS_raw"].quantile(0.05), hh["SOS_raw"].quantile(0.95)
            denom = max(q95 - q5, 1.0)
            hh["SOS"] = (2.0 * (hh["SOS_raw"] - q5) / denom).clip(0, 2)
        else:
            hh["SOS"] = 0.0

        # --- TFS (trip friction) ---  (moved above ASI so ASI can use TFS_plus)
        def tfs_row(r):
            last_cols = [c for c in ["300_Time", "200_Time", "100_Time"] if c in r.index]
            spds = [metrics.attrs.get("STEP", 100) / as_num(r.get(c))
                    for c in last_cols if pd.notna(r.get(c)) and as_num(r.get(c)) > 0]
            if len(spds) < 2:
                return np.nan
            sigma = np.std(spds, ddof=0)
            mid = as_num(r.get("_MID_spd"))
            return np.nan if not np.isfinite(mid) or mid <= 0 else 100.0 * (sigma / mid)

        hh["TFS"] = hh.apply(tfs_row, axis=1)
        D_rounded = int(np.ceil(float(race_distance_input) / 200.0) * 200)
        _gate = 4.0 if D_rounded <= 1200 else (3.5 if D_rounded < 1800 else 3.0)
        hh["TFS_plus"] = hh["TFS"].apply(lambda x: 0.0 if pd.isna(x) or x < _gate else min(0.6, (x - _gate) / 3.0))


        # --- ASI (Against-Shape Index, v3; race-local, 0–2 scale) ---
        def _rz(s):
            s = winsorize(pd.to_numeric(s, errors="coerce"))
            mu = np.nanmedian(s)
            mad = np.nanmedian(np.abs(s - mu))
            sd = 1.4826 * mad if mad > 0 else np.nanstd(s)
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            return (s - mu) / sd

        # 1) Flow strength (FS) from RacePulse if available, else a safe proxy
        RSI = metrics.attrs.get("RSI", np.nan)
        SCI = metrics.attrs.get("SCI", np.nan)
        collapse = float(metrics.attrs.get("CollapseSeverity", 0.0) or 0.0)

        if not np.isfinite(RSI) or not np.isfinite(SCI):
            # Fallback proxy using early/late distribution
            zE = _rz(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0, index=hh.index)
            zL = _rz(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0, index=hh.index)
            RSI = float(np.nanmedian(zE) - np.nanmedian(zL))  # >0 early tilt
            SCI = 0.50  # neutral clarity if unknown

        _dir = 0 if (not np.isfinite(RSI) or abs(RSI) < 1e-6) else (1 if RSI > 0 else -1)
        FS = 0.0 if _dir == 0 else (0.6 + 0.4 * max(0.0, min(1.0, float(SCI)))) * min(1.0, abs(float(RSI)) / 2.0)
        if collapse >= 3.0:
            FS *= 0.75  # collapse guard

        # 2) Style opposition (SO): early vs late style using Accel vs Grind
        zA, zG = _rz(hh.get("Accel")), _rz(hh.get(gr_col))
        if _dir == 1:   # early-favoured race
            SO = (zG - zA).clip(lower=0)
        elif _dir == -1:  # late-favoured race
            SO = (zA - zG).clip(lower=0)
        else:
            SO = pd.Series(0.0, index=hh.index)

        # 3) Segment execution opposition (XO): EARLY_idx vs LATE_idx
        zE = _rz(hh.get("EARLY_idx")) if "EARLY_idx" in hh.columns else pd.Series(0.0, index=hh.index)
        zL = _rz(hh.get("LATE_idx"))  if "LATE_idx"  in hh.columns else pd.Series(0.0, index=hh.index)
        if _dir == 1:
            XO = (zL - zE).clip(lower=0)
        elif _dir == -1:
            XO = (zE - zL).clip(lower=0)
        else:
            XO = pd.Series(0.0, index=hh.index)

        # 4) False-positive dampeners (trip friction & grind anomalies)
        tfs_plus = pd.to_numeric(hh.get("TFS_plus"), errors="coerce").fillna(0.0)
        gr_adj  = pd.to_numeric(hh.get("GrindAdjPts"), errors="coerce").fillna(1.0)

        D1 = 1.0 - np.minimum(0.35, tfs_plus.clip(lower=0.0))                   # up to -35%
        D2 = 1.0 - np.minimum(0.25, ((gr_adj - 1.0).clip(lower=0.0) / 3.0))      # up to -25%
        D  = D1 * D2

        # Combine (more weight on style than execution), scale to 0–10, then to 0–2
        Opp   = 0.6 * SO + 0.4 * XO
        ASI10 = 10.0 * FS * Opp * D
        hh["ASI2"] = (0.2 * ASI10).clip(0.0, 2.0).fillna(0.0)
        # --- UEI (underused engine) ---
        def uei_row(r):
            ts, ac, gr = [as_num(r.get(k)) for k in ("tsSPI", "Accel", gr_col)]
            if any(pd.isna([ts,ac,gr])): return 0.0
            val = 0.0
            if ts >= 102 and ac <= 98 and gr <= 98:
                val = 0.3 + 0.3 * min((ts-102)/3.0, 1.0)
            if ts >= 102 and gr >= 102 and ac <= 100:
                val = max(val, 0.3 + 0.3 * min(((ts-102)+(gr-102))/6.0, 1.0))
            return round(val, 3)
        hh["UEI"] = hh.apply(uei_row, axis=1)

        # --- HiddenScore ---
        hidden = (0.55*hh["SOS"] + 0.30*hh["ASI2"] + 0.10*hh["TFS_plus"] + 0.05*hh["UEI"]).fillna(0.0)
        if len(hh) <= 6: hidden *= 0.9
        h_med, h_mad = float(np.nanmedian(hidden)), float(np.nanmedian(np.abs(hidden - np.nanmedian(hidden))))
        h_sigma = max(1e-6, 1.4826*h_mad)
        hh["HiddenScore"] = (1.2 + (hidden - h_med) / (2.5*h_sigma)).clip(0.0, 3.0)

        # --- Tier logic (race-shape-aware) ---
        def hh_tier_row(r):
            """Return a tier label for Hidden Horses v2."""
            hs = as_num(r.get("HiddenScore"))
            if not np.isfinite(hs):
                return ""

            # PI-only baseline gate so weak raw performances are not crowned as hidden horses.
            pi_val = as_num(r.get("PI"))

            def baseline_ok_for(top: bool) -> bool:
                threshold = 5.4 if top else 4.8
                return bool(np.isfinite(pi_val) and pi_val >= threshold)

            if hs >= 1.8 and baseline_ok_for(top=True):
                return "🔥 Top Hidden"
            if hs >= 1.2 and baseline_ok_for(top=False):
                return "🟡 Notable Hidden"
            return ""
        hh["Tier"] = hh.apply(hh_tier_row, axis=1)

        # --- Descriptive note ---
        def hh_note(r):
            pi = as_num(r.get("PI"))
            bits=[]
            if np.isfinite(pi):
                bits.append(f"PI {pi:.2f}")
            else:
                if as_num(r.get("SOS")) >= 1.2: bits.append("sectionals superior")
                asi2 = as_num(r.get("ASI2"))
                if asi2 >= 0.8: bits.append("ran against strong bias")
                elif asi2 >= 0.4: bits.append("ran against bias")
                if as_num(r.get("TFS_plus")) > 0: bits.append("trip friction late")
                if as_num(r.get("UEI")) >= 0.5: bits.append("latent potential if shape flips")
            return "; ".join(bits).capitalize()+"."
        hh["Note"] = hh.apply(hh_note, axis=1)

        # ---- Build ranked, presentation-friendly Hidden Horses table ----
        cols_hh = ["Horse","Finish_Pos","PI","tsSPI","Accel",gr_col,
                   "SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
        for c in cols_hh:
            if c not in hh.columns:
                hh[c] = np.nan

        # numeric hygiene
        num_cols = ["PI","tsSPI","Accel",gr_col,"SOS","ASI2","TFS","UEI","HiddenScore"]
        for c in num_cols:
            hh[c] = pd.to_numeric(hh[c], errors="coerce")

        # explicit tier ordering (for secondary sort / grouping)
        _tier_order = {"🔥 Top Hidden": 0, "🟡 Notable Hidden": 1, "": 2}
        hh["_tier_order"] = hh["Tier"].map(_tier_order).fillna(2)

        # primary sort = HiddenScore (desc), then Tier order, then PI (desc)
        hh_ranked = (
            hh.sort_values(["HiddenScore", "_tier_order", "PI"],
                           ascending=[False, True, False])
              .reset_index(drop=True)
        )

        # --- Build final table -------------------------------------------------
        cols_hh = ["Horse","Finish_Pos","PI","tsSPI","Accel",gr_col,"SOS","ASI2","TFS","UEI","HiddenScore","Tier","Note"]
        for c in cols_hh:
            if c not in hh.columns:
                hh[c] = np.nan

        # Tier order: put the best at the top
        tier_order = pd.CategoricalDtype(categories=["🔥 Top Hidden","🟡 Notable Hidden",""], ordered=True)
        hh["Tier"] = hh["Tier"].astype(tier_order)

        hh_view = hh[cols_hh].copy()
        hh_view = hh_view.sort_values(["Tier","HiddenScore","PI"], ascending=[True, False, False])

        # ---- Safe numeric casting & rounding (no helper, no NameError) ----
        def _cast_round(df, col):
            if col not in df.columns:
                return
            # Make a 1-D Series aligned to df.index regardless of the source shape/type
            s = pd.Series(np.ravel(df[col].values), index=df.index)
            df[col] = pd.to_numeric(s, errors="coerce").round(2)

        for c in ["PI","ASI2","SOS","TFS","UEI","Accel",gr_col,"tsSPI","HiddenScore"]:
            _cast_round(hh_view, c)

        st.dataframe(hh_view, use_container_width=True)
        st.caption("Hidden Horses v2 — sorted by Tier, then HiddenScore, then PI (Top first).")

    if _view_is("Advanced Models"):
        # ======================= xWin — Probability to Win (100-replay view) =======================
        st.markdown("## xWin — Probability to Win")

        XW = metrics.copy()
        gr_col = metrics.attrs.get("GR_COL", "Grind")
        D_m    = float(race_distance_input)
        RSI    = float(metrics.attrs.get("RSI", 0.0))        # + slow-early, − fast-early
        SCI    = float(metrics.attrs.get("SCI", 0.0))        # 0..1 (shape consensus)
        going  = str(metrics.attrs.get("GOING", "Good"))

        # ---------- helpers ----------
        def _clip(x, lo, hi):
            try:
                x = float(x); 
                return lo if x < lo else (hi if x > hi else x)
            except:
                return lo

        def _lerp(a, b, t): 
            t = _clip(t, 0.0, 1.0)
            return a + (b - a) * t

        def _winsor(s: pd.Series, p=0.02):
            s = pd.to_numeric(s, errors="coerce")
            lo, hi = s.quantile(p), s.quantile(1-p)
            return s.clip(lower=lo, upper=hi)

        def _robust_z(s: pd.Series):
            """Median / MAD z-score; clipped to ±3 for stability."""
            x  = _winsor(pd.to_numeric(s, errors="coerce"))
            mu = np.nanmedian(x)
            sd = mad_std(x)
            if not np.isfinite(sd) or sd <= 0:
                z = (x - mu) / 1.0
            else:
                z = (x - mu) / sd
            return z.clip(-3.0, 3.0)

        def _weights_for_distance(dm):
            """Distance-aware weights for Travel/Kick/Sustain (sum=1 before going tweak)."""
            dm = float(dm)
            knots = [
                (1000, dict(T=0.30, K=0.45, S=0.25)),   # sprints → K heavier
                (1200, dict(T=0.30, K=0.40, S=0.30)),
                (1400, dict(T=0.32, K=0.36, S=0.32)),
                (1600, dict(T=0.34, K=0.32, S=0.34)),
                (1800, dict(T=0.36, K=0.28, S=0.36)),
                (2000, dict(T=0.38, K=0.25, S=0.37)),
                (2400, dict(T=0.40, K=0.22, S=0.38)),   # staying → S heavier
            ]
            if dm <= knots[0][0]: return knots[0][1]
            if dm >= knots[-1][0]: return knots[-1][1]
            for (a_dm, a_w), (b_dm, b_w) in zip(knots, knots[1:]):
                if a_dm <= dm <= b_dm:
                    t = (dm - a_dm) / (b_dm - a_dm)
                    return {k: _lerp(a_w[k], b_w[k], t) for k in a_w}
            return knots[-1][1]

        def _apply_going_nudge(w, going_str, field_n=12):
            """Small surface/going tweak; renormalises to 1."""
            w = w.copy()
            scale = min(1.0, max(1, int(field_n)) / 12.0)
            if going_str == "Firm":
                w["K"] *= (1.00 + 0.04*scale)
                w["T"] *= (1.00 + 0.02*scale)
                w["S"] *= (1.00 - 0.04*scale)
            elif going_str in ("Soft","Heavy"):
                amp = 0.05 if going_str == "Soft" else 0.08
                w["S"] *= (1.00 + amp*scale)
                w["T"] *= (1.00 + 0.02*scale)
                w["K"] *= (1.00 - amp*scale)
            S = sum(w.values()) or 1.0
            for k in w: w[k] /= S
            return w

        def _temperature(N, accel, grind, dm, tfs_plus=None):
            """
            Race 'temperature' τ for softmax: lower = sharper probs (decisive ability gaps),
            higher = flatter probs (chaos, bunching, traffic).
            """
            N = max(1, int(N))

            def _mad01(s):
                s = pd.to_numeric(s, errors="coerce")
                d = mad_std(s)
                if not np.isfinite(d): return 0.0
                # ~ 1σ ~ 4–5 idx pts → map near 1.0
                return float(min(1.0, d / 4.5))

            d_ac  = _mad01(accel)
            d_gr  = _mad01(grind)
            base  = 0.95
            size_adj = -0.04*np.log1p(N)                    # bigger fields → lower τ
            disp_adj = -0.16*(0.5*d_ac + 0.5*d_gr)          # clear sectional separation → lower τ
            dist_adj = (0.04 if dm <= 1100 else (0.00 if dm <= 1800 else -0.02))
            tfs_adj  = 0.0
            if tfs_plus is not None:
                # more widespread friction → noisier replays → higher τ
                tp = pd.to_numeric(tfs_plus, errors="coerce").fillna(0.0)
                tfs_adj = 0.08 * float(np.clip(np.nanmean(np.maximum(0.0, (tp - 0.2)/0.4)), 0.0, 1.0))

            tau = base + size_adj + disp_adj + dist_adj + tfs_adj
            return float(_clip(tau, 0.55, 1.15))

        def _to_fractional_odds(p):
            """p in [0,1] → 'x.y/1' (fair fractional style)."""
            try:
                p = float(p)
                if p <= 0: return "-"
                dec = 1.0 / p
                frac = dec - 1.0
                return f"{frac:.1f}/1"
            except:
                return "-"

        # ---------- ensure inputs ----------
        for c in ["tsSPI","Accel",gr_col,"F200_idx","PI"]:
            if c not in XW.columns: XW[c] = np.nan

        # ---------- robust sectionals → within-race latent ability (z) ----------
        zT = _robust_z(XW["tsSPI"])   # Travel
        zK = _robust_z(XW["Accel"])   # Kick
        zS = _robust_z(XW[gr_col])    # Sustain

        # pace legitimacy guard (if race crawled mid, trim Travel influence a bit)
        ts_med = pd.to_numeric(XW["tsSPI"], errors="coerce").median(skipna=True)
        trim_T = 0.0
        if np.isfinite(ts_med) and ts_med < 100.0:
            trim_T = min(0.20, max(0.0, (100.0 - ts_med) / 10.0))  # up to 20%
        zT_eff = zT * (1.0 - trim_T)

        # ---------- distance + going weights ----------
        W = _weights_for_distance(D_m)               # {'T','K','S'}
        W = _apply_going_nudge(W, going, field_n=len(XW))

        # ---------- shape de-bias (KSI proxy) ----------
        # Positive when horse ran AGAINST prevailing shape; negative when WITH shape
        ksi_raw = -np.sign(RSI) * (pd.to_numeric(XW["Accel"], errors="coerce") - pd.to_numeric(XW["tsSPI"], errors="coerce"))
        ksi01   = np.tanh((ksi_raw / 6.0).fillna(0.0))        # ~[-1..+1]
        shape_boost = 0.15 * np.clip(ksi01, 0, 1) * SCI       # up to +15% (against)
        shape_damp  = 0.08 * np.clip(-ksi01, 0, 1) * SCI      # up to −8%  (with)

        # ---------- trip friction (from Hidden Horses if present) ----------
        tfs_plus = None
        try:
            if 'hh' in locals() and "TFS_plus" in hh.columns and "Horse" in XW.columns:
                tmp = hh[["Horse","TFS_plus"]].copy()
                XW = XW.merge(tmp, on="Horse", how="left")
                tfs_plus = pd.to_numeric(XW["TFS_plus"], errors="coerce").fillna(0.0)
        except Exception:
            pass
        if tfs_plus is None:
            tfs_plus = pd.Series(0.0, index=XW.index)

        # harsher in sprints, softer in staying trips
        if D_m <= 1400:   tfs_cap = 0.12
        elif D_m >= 1800: tfs_cap = 0.08
        else:             tfs_cap = _lerp(0.12, 0.08, (D_m-1400)/400.0)
        tfs_pen = np.minimum(tfs_cap, np.maximum(0.0, (tfs_plus - 0.2)/0.4))

        # ---------- core latent score (no history; pure one-run) ----------
        # small optional stability from sectionals dispersion (SOS-like)
        sos = (0.45*zT + 0.35*zK + 0.20*zS).fillna(0.0)
        sos01 = ((sos - np.nanpercentile(sos, 5)) /
                 max(1e-9, (np.nanpercentile(sos,95) - np.nanpercentile(sos,5))))
        sos01 = sos01.clip(0, 1)

        core = (
            W["T"] * zT_eff.fillna(0.0) +
            W["K"] * zK.fillna(0.0)     +
            W["S"] * zS.fillna(0.0)     +
            0.05   * sos01.fillna(0.0)  # very light stabiliser
        )

        # multiplicative de-lucking: reward against-shape, damp with-shape, damp friction
        mult_adj = (1.0 + shape_boost - shape_damp) * (1.0 - tfs_pen)
        power = (core * mult_adj).fillna(0.0)

        # ---------- field-size shrink & temperature ----------
        N    = int(len(XW.index))
        tau  = _temperature(N, XW["Accel"], XW[gr_col], D_m, tfs_plus=tfs_plus)
        alpha = N / (N + 6.0)              # small-field shrink (same motif you use elsewhere)
        if N <= 6:
            power = 0.90 * power           # reduce overconfidence in tiny fields

        # ---------- softmax → probabilities ----------
        logits   = power / max(1e-6, tau)
        mx       = float(np.nanmax(logits)) if np.isfinite(logits).any() else 0.0
        exps     = np.exp((logits - mx).clip(-50, 50))
        sum_exps = float(np.nansum(exps)) or 1.0
        probs    = (exps / sum_exps) * alpha
        probs    = probs / (probs.sum() or 1.0)     # renormalise after shrink

        XW["xWin"] = probs

        # ---------- tidy drivers ----------
        def _driver_line(r):
            bits = []
            # early hint
            f200 = float(r.get("F200_idx", np.nan))
            if np.isfinite(f200):
                if f200 >= 101: bits.append("Quick early")
                elif f200 <= 98: bits.append("Slower away")

            # sectional pillars
            if float(zT_eff.get(r.name, 0)) >= 0.5: bits.append("Travel +")
            if float(zK.get(r.name,     0)) >= 0.5: bits.append("Kick ++")
            if float(zS.get(r.name,     0)) >= 0.5: bits.append("Sustain +")

            # shape cue (only if SCI decent)
            if SCI >= 0.6:
                k = float(ksi01.get(r.name, 0))
                if k > 0.35:   bits.append("Against shape")
                elif k < -0.35: bits.append("With shape")

            # midrace trim note
            if trim_T > 0: bits.append("(slow mid)")

            return " · ".join(bits)

        XW["Drivers"] = XW.apply(_driver_line, axis=1)

        # ---------- view ----------
        view = XW.loc[:, ["Horse","xWin","Drivers"]].copy()
        view["xWin"] = (100.0 * view["xWin"]).round(1)
        view["Odds (≈fair)"] = XW["xWin"].apply(lambda p: _to_fractional_odds(p))

        view = view.sort_values("xWin", ascending=False).reset_index(drop=True)

        st.dataframe(
            view.style.format({"xWin": "{:.1f}%"}),
            use_container_width=True
        )

        with st.expander("xWin settings & notes"):
            w_note = ", ".join([f"{k}:{W[k]:.2f}" for k in ["T","K","S"]])
            st.caption(
                f"xWin = softmax of within-race latent ability (Travel/Kick/Sustain) with distance/going weights ({w_note}), "
                f"shape de-bias via RSI×SCI, trip friction damp, and a race 'temperature' τ={tau:.2f} from field size & dispersion. "
                f"Interpretation: chance to win if this same race were replayed 100 times."
            )
