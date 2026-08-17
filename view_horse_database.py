"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_horse_database_view(ctx):
    globals().update(ctx)
    if _view_is("Horse Database"):
        st.title("Horse Database")
        st.caption("Research saved horse histories or compare multiple horses. This module works without a race file loaded.")
        search_tab, compare_tab = st.tabs(["Horse Search", "Compare Horses"])
        with search_tab:
            render_horse_search()
        with compare_tab:
            render_horse_compare()
