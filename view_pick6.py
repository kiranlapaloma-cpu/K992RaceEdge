"""Race Edge Suggested Pick 6: Recent Best, no apprentice allowances."""
from math import prod

import pandas as pd
import streamlit as st


def _pick6_keys(options):
    """Race-number rules: 7 -> 2–7, 8 -> 3–8, 9+ -> 4–9."""
    numbered = []
    for key, _label in options:
        try:
            numbered.append((int(key), str(key)))
        except (TypeError, ValueError):
            continue
    numbered.sort()
    count = len(numbered)
    if count < 7:
        return []
    start = 2 if count == 7 else 3 if count == 8 else 4
    return [(number, key) for number, key in numbered if start <= number < start + 6]


def render_suggested_pick6(
    meeting, options, *, card_loader, runner_frame, db_counts_loader,
    history_loader, prediction_builder, database_configured,
):
    st.markdown('## Suggested Pick 6')
    legs = _pick6_keys(options)
    if len(legs) != 6:
        st.warning('This meeting does not contain six Pick 6 legs under the configured race rules.')
        return

    st.caption(
        f'R{legs[0][0]}–R{legs[-1][0]} · Recent Best only · '
        'No apprentice allowances · Every rated runner within 3.00 lengths of the leader.'
    )
    if not database_configured():
        st.warning('Race Edge database is not configured. Suggestions cannot be calculated.')
        return
    try:
        counts = db_counts_loader()
    except Exception as exc:
        st.error(f'Could not retrieve saved Race Edge run counts: {exc}')
        return

    # Avoid repeatedly loading the same horse's history across six races.
    history_cache = {}
    def cached_history(horse):
        if horse not in history_cache:
            history_cache[horse] = history_loader(horse)
        return history_cache[horse]

    warnings = []
    leg_selections = []
    for leg_number, (race_number, race_key) in enumerate(legs, 1):
        try:
            card = card_loader(meeting, race_key)
            field = runner_frame(card, counts)
        except Exception as exc:
            st.error(f'Leg {leg_number} / R{race_number}: could not load race: {exc}')
            leg_selections.append([])
            continue
        if field.empty:
            st.warning(f'Leg {leg_number} / R{race_number}: no runners found.')
            leg_selections.append([])
            continue
        active = field.loc[field['Status'] == 'Runner'].copy()
        if active.empty:
            st.warning(f'Leg {leg_number} / R{race_number}: no active runners.')
            leg_selections.append([])
            continue

        for _, runner in active.iterrows():
            no = runner.get('No.')
            number = '?' if pd.isna(no) else str(int(no))
            label = f'No. {number} {runner["Horse"]}'
            saved = runner.get('Race Edge Runs')
            saved = 0 if pd.isna(saved) else int(saved)
            if saved == 0:
                warnings.append(f'Leg {leg_number} · R{race_number} · {label} — NO DATA')
            elif saved == 1:
                warnings.append(f'Leg {leg_number} · R{race_number} · {label} — LIMITED DATA (1 saved race)')
            claim = pd.to_numeric(pd.Series([runner.get('Claim')]), errors='coerce').fillna(0).iloc[0]
            if claim > 0:
                warnings.append(f'Leg {leg_number} · R{race_number} · {label} — APPRENTICE CLAIM ({claim:g} kg; not applied)')

        try:
            distance = float(card.get('distance'))
            prediction = prediction_builder(
                active,
                race_date=card.get('date') or card.get('dateFormat'),
                distance_m=distance,
                history_loader=cached_history,
                apply_apprentice_claims=False,
            )
            recent = prediction.get('scenarios', {}).get('Recent Best', pd.DataFrame())
            recent = recent.copy()
        except Exception as exc:
            st.error(f'Leg {leg_number} / R{race_number}: prediction unavailable: {exc}')
            leg_selections.append([])
            continue

        if recent.empty or 'Recent Best Projection' not in recent.columns:
            st.warning(f'Leg {leg_number} / R{race_number}: NO QUALIFYING SELECTIONS — MANUAL REVIEW REQUIRED.')
            leg_selections.append([])
            continue
        recent['Recent Best Projection'] = pd.to_numeric(recent['Recent Best Projection'], errors='coerce')
        recent = recent.dropna(subset=['Recent Best Projection']).copy()
        if recent.empty:
            st.warning(f'Leg {leg_number} / R{race_number}: NO QUALIFYING SELECTIONS — MANUAL REVIEW REQUIRED.')
            leg_selections.append([])
            continue
        leader = float(recent['Recent Best Projection'].max())
        recent['Behind leader (L)'] = ((leader - recent['Recent Best Projection']) * 0.5).clip(lower=0).round(2)
        recent = recent.sort_values(['Recent Best Projection', 'Horse'], ascending=[False, True])
        suggested = recent.loc[recent['Behind leader (L)'] <= 3.0 + 1e-9]
        def label(row):
            no = row.get('No.')
            no_text = '?' if pd.isna(no) else str(int(no))
            return f'{no_text} · {row["Horse"]}'
        labels = [label(row) for _, row in recent.iterrows()]
        defaults = [label(row) for _, row in suggested.iterrows()]
        with st.container(border=True):
            st.markdown(f'**Leg {leg_number} · Race {race_number}**')
            st.caption(f'{len(defaults)} suggested · {len(active)} active runners · 3.00 L cutoff')
            st.dataframe(
                recent.assign(Selected=recent['Behind leader (L)'] <= 3.0 + 1e-9)[
                    ['No.', 'Horse', 'Recent Best Projection', 'Behind leader (L)', 'Selected']
                ],
                hide_index=True, width='stretch',
            )
            state_key = f'pick6_selection_{race_number}'
            # A newly loaded meeting should not inherit a different meeting's ticket.
            identity = str(meeting.get('date') or meeting.get('dateFormat') or '') + ':' + str(meeting.get('club') or meeting.get('clubName') or '')
            meeting_key = f'pick6_meeting_{race_number}'
            if st.session_state.get(meeting_key) != identity:
                st.session_state.pop(state_key, None)
                st.session_state[meeting_key] = identity
            if state_key in st.session_state:
                st.session_state[state_key] = [item for item in st.session_state[state_key] if item in labels]
            selected = st.multiselect('Your ticket selections', labels, default=defaults, key=state_key)
            leg_selections.append(selected)

    counts_per_leg = [len(items) for items in leg_selections]
    combinations = prod(counts_per_leg) if len(counts_per_leg) == 6 else 0
    st.markdown('### Ticket summary')
    st.dataframe(pd.DataFrame([
        {'Leg': i, 'Race': f'R{number}', 'Horses': ', '.join(items) if items else 'NONE', 'Selections': len(items)}
        for i, ((number, _key), items) in enumerate(zip(legs, leg_selections), 1)
    ]), hide_index=True, width='stretch')
    unit = st.number_input('Stake per combination (R)', min_value=0.01, value=1.0, step=0.5, format='%.2f', key='pick6_unit_stake')
    st.markdown(f'**{combinations:,} combinations · R{combinations * unit:,.2f} ticket cost**')
    if not combinations:
        st.warning('Ticket is incomplete: select at least one horse in every leg.')
    st.markdown('### Race Edge Cautions')
    if warnings:
        for item in warnings:
            st.warning(item)
    else:
        st.success('No missing-history, single-run or apprentice-claim cautions among active runners.')
