"""Race Edge Streamlit view extracted from the original stable application.

This module intentionally receives the live application context so calculation
behaviour remains identical while the UI is maintained independently.
"""

def render_race_card_view(ctx):
    globals().update(ctx)
    if _view_is("Race Card"):
        render_race_card()
