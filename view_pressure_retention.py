"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_pressure_retention(ctx):
    globals().update(ctx)
    if _view_is("Pressure Retention"):
        st.markdown("## Pressure Retention Index (PRI)")
        st.caption(
            "PRI measures how much sustained pressure a horse absorbed after the opening 200m "
            "and, crucially, how much of that pressure it retained through the final 400m. "
            "Pressure credit is reduced when a horse fades. PRI remains separate from PI."
        )

        pri = PRI_TABLE.copy()
        valid_pri = pri.dropna(subset=["Pressure_Delta_pct", "Retention_pct", "PRI"]).copy()

        if valid_pri.empty:
            st.info("PRI could not be calculated from the available sectional columns.")
        else:
            top = valid_pri.sort_values("PRI", ascending=False).iloc[0]
            high_pressure = valid_pri.sort_values("Pressure_Delta_pct", ascending=False).iloc[0]
            best_retention = valid_pri.sort_values("Retention_pct", ascending=False).iloc[0]

            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Top PRI", str(top["Horse"]), f'{float(top["PRI"]):.2f}')
            pc2.metric("Most pressure absorbed", str(high_pressure["Horse"]), f'{float(high_pressure["Pressure_Delta_pct"]):+.2f}%')
            pc3.metric("Best late retention", str(best_retention["Horse"]), f'{float(best_retention["Retention_pct"]):.1f}%')

            st.markdown("### Pressure vs Retention map")
            fig, ax = plt.subplots(figsize=(10.5, 6.6))
            x = pd.to_numeric(valid_pri["Pressure_Delta_pct"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(valid_pri["Retention_pct"], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(valid_pri["PRI"], errors="coerce").to_numpy(dtype=float)
            sizes = 70 + 18 * np.clip(c, 0, 10)

            sc = ax.scatter(x, y, c=c, s=sizes, cmap="viridis", vmin=0, vmax=10,
                            edgecolors="white", linewidths=0.8, alpha=0.92, zorder=3)
            retention_med = float(valid_pri.attrs.get("retention_median", np.nanmedian(y)))
            ax.axvline(0.0, color="grey", lw=1.1, ls="--", alpha=0.75)
            ax.axhline(retention_med, color="grey", lw=1.1, ls="--", alpha=0.75)

            for _, r in valid_pri.iterrows():
                ax.annotate(
                    str(r["Horse"]),
                    (float(r["Pressure_Delta_pct"]), float(r["Retention_pct"])),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, alpha=0.9
                )

            ax.set_xlabel("Pressure absorbed vs field median (%)")
            ax.set_ylabel("Late retention: final-400 speed / pressure-phase speed (%)")
            ax.set_title("Pressure Retention Map")
            ax.grid(alpha=0.18)
            cb = fig.colorbar(sc, ax=ax, pad=0.02)
            cb.set_label("PRI (0–10)")
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            st.markdown("### PRI leaderboard")
            pri_view_cols = [
                "PRI_Rank", "Horse", "Finish_Pos", "PI",
                "Pressure_Speed", "Late_Speed",
                "Pressure_Delta_pct", "Retention_pct", "Retention_Gate",
                "Pressure_Gate", "Retained_Pressure", "PRI", "Profile"
            ]
            pri_view = pri[[c for c in pri_view_cols if c in pri.columns]].copy()
            pri_view = pri_view.sort_values(["PRI", "Pressure_Delta_pct"], ascending=[False, False], na_position="last")
            rename = {
                "PRI_Rank": "Rank",
                "Pressure_Speed": "Pressure speed (m/s)",
                "Late_Speed": "Late speed (m/s)",
                "Pressure_Delta_pct": "Pressure vs field (%)",
                "Retention_pct": "Retention (%)",
                "Retention_Gate": "Retention gate",
                "Pressure_Gate": "Pressure gate",
                "Retained_Pressure": "Retained pressure",
            }
            pri_view = pri_view.rename(columns=rename)
            for col in ["PI", "Pressure speed (m/s)", "Late speed (m/s)", "Pressure vs field (%)", "Retention (%)", "Retention gate", "Pressure gate", "Retained pressure", "PRI"]:
                if col in pri_view.columns:
                    pri_view[col] = pd.to_numeric(pri_view[col], errors="coerce").round(2)
            st.dataframe(pri_view, width="stretch", hide_index=True)

            csv_data = pri_view.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download PRI CSV",
                data=csv_data,
                file_name=f"race_edge_PRI_{int(race_distance_input)}m.csv",
                mime="text/csv",
            )

            with st.expander("How to read PRI"):
                st.markdown(
                    """
    - **Pressure vs field (%)** measures the horse's speed through the pressure phase against the field median.
    - **Retention (%)** compares final-400 speed with the horse's own pressure-phase speed.
    - **Retention gate** controls how much positive pressure credit survives: 90% retention gives no pressure credit, while 100% gives full credit.
    - **Pressure gate** prevents a horse that sat well off the pace from earning a top PRI purely because it flew home. Negative pressure is heavily restricted; strong positive pressure receives full access.
    - **Retained pressure** is the horse's positive pressure score multiplied by that retention gate.
    - **Pressure resistant**: above-median pressure and above-median retention.
    - **Brave but faded**: absorbed above-median pressure but retained less late; its pressure credit is therefore reduced.
    - **Pace-assisted closer**: absorbed less pressure but retained strongly late.
    - **Low-pressure performer**: below median in both dimensions.
    - PRI combines **50% retained pressure and 50% race-relative retention** and remains separate from PI.
                    """
                )
