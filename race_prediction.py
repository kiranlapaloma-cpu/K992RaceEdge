from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from performance_profile import build_performance_profile
from wfa import get_wfa_lb


def _num(value: Any):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _parse_race_date(value: Any):
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    text = str(value or "").strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            pass
    return datetime.now().date()


def _evidence_score(label: str) -> int:
    return {
        "Established": 3,
        "Emerging": 2,
        "Limited": 1,
        "No evidence": 0,
    }.get(str(label or ""), 0)


def _group_label(index: int) -> str:
    """Return A, B, ... Z, AA, AB ... for zero-based group indices."""
    index = int(index)
    label = ""
    while True:
        index, rem = divmod(index, 26)
        label = chr(65 + rem) + label
        if index == 0:
            return label
        index -= 1


def assign_rating_groups(
    ratings: pd.Series,
    *,
    threshold_points: float = 5.0,
) -> pd.Series:
    """
    Assign Group A, B, C... using a fixed leader anchor for each group.

    A horse stays in the current group while it is LESS THAN threshold_points
    below that group's leader. A gap of threshold_points or more starts a
    new group.

    Example with threshold 5:
        58, 57, 55 -> Group A
        53, 52, 51 -> Group B
    because 53 is exactly 5 points below the Group A leader (58).
    """
    numeric = pd.to_numeric(ratings, errors="coerce")
    result = pd.Series(index=ratings.index, dtype="object")

    valid = numeric.dropna().sort_values(ascending=False, kind="stable")
    if valid.empty:
        return result

    group_index = 0
    group_leader = float(valid.iloc[0])

    for idx, rating in valid.items():
        rating = float(rating)
        if group_leader - rating >= float(threshold_points):
            group_index += 1
            group_leader = rating
        result.loc[idx] = f"Group {_group_label(group_index)}"

    return result


def build_race_predictions(
    active: pd.DataFrame,
    *,
    race_date: Any,
    distance_m: float,
    history_loader,
) -> dict:
    """
    Build three independent Race Edge projections for today's race.

    Latest, Established and Peak ability are never blended.

    Today's terms:
        Effective Weight = carded weight + WFA allowance in kg
        1 kg = 2 MR points

    Each scenario is projected to the lightest effective weight in today's field.
    The common reference affects the displayed rating level, not the ranking.
    """
    if active is None or active.empty:
        return {"rows": pd.DataFrame(), "scenarios": {}, "consensus": []}

    race_date = _parse_race_date(race_date)
    distance_m = float(distance_m)

    rows = []
    for _, runner in active.iterrows():
        horse = str(runner.get("Horse") or "").strip()
        if not horse:
            continue

        age = _num(runner.get("Age"))
        carded_weight = _num(runner.get("Weight"))
        current_mr = _num(runner.get("Official MR"))
        if age is None or carded_weight is None:
            continue

        try:
            history = history_loader(horse)
        except Exception:
            continue
        if history is None or history.empty:
            continue

        profile = build_performance_profile(
            history,
            current_official_mr=current_mr,
        )
        latest = _num(profile.get("latest_mr_achieved"))
        established = _num(profile.get("established_mr"))
        peak = _num(profile.get("highest_mr_achieved"))
        if latest is None and established is None and peak is None:
            continue

        try:
            wfa_lb = float(get_wfa_lb(race_date, distance_m, int(round(age))))
        except Exception:
            wfa_lb = 0.0

        wfa_kg = wfa_lb * 0.5
        effective_weight = float(carded_weight) + wfa_kg

        rows.append({
            "Horse": horse,
            "Current MR": current_mr,
            "Age": int(round(age)),
            "Carded Weight": float(carded_weight),
            "WFA lb": wfa_lb,
            "WFA kg": wfa_kg,
            "Effective Weight": effective_weight,
            "Latest MR": latest,
            "Established MR": established,
            "Peak MR": peak,
            "Evidence": str(profile.get("established_evidence") or "No evidence"),
            "Evidence Score": _evidence_score(profile.get("established_evidence")),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {"rows": df, "scenarios": {}, "consensus": []}

    reference_weight = float(df["Effective Weight"].min())
    weight_penalty = 2.0 * (df["Effective Weight"] - reference_weight)

    scenario_defs = {
        "Latest Form": ("Latest MR", "Latest Projection"),
        "Established Ability": ("Established MR", "Established Projection"),
        "Peak Ability": ("Peak MR", "Peak Projection"),
    }

    scenarios = {}
    for scenario_name, (source_col, projection_col) in scenario_defs.items():
        df[projection_col] = pd.to_numeric(df[source_col], errors="coerce") - weight_penalty

        valid = df.loc[df[projection_col].notna()].copy()
        valid = valid.sort_values(
            [projection_col, "Evidence Score", "Horse"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        valid["Rank"] = np.arange(1, len(valid) + 1)

        # Sequential predicted margin:
        # each horse is measured only against the horse immediately above it.
        # Race Edge convention: 1 rating point = 0.5 lengths.
        valid["Margin Behind Previous (L)"] = 0.0
        if len(valid) > 1:
            previous_rating = pd.to_numeric(valid[projection_col], errors="coerce").shift(1)
            current_rating = pd.to_numeric(valid[projection_col], errors="coerce")
            margins = (previous_rating - current_rating) * 0.5
            valid.loc[1:, "Margin Behind Previous (L)"] = margins.loc[1:].clip(lower=0.0)
        valid["Margin Behind Previous (L)"] = (
            pd.to_numeric(valid["Margin Behind Previous (L)"], errors="coerce")
            .fillna(0.0)
            .round(2)
        )

        scenarios[scenario_name] = valid[
            [
                "Rank", "Horse", projection_col, "Margin Behind Previous (L)",
                "Current MR", "Evidence"
            ]
        ].copy()

        df[f"{scenario_name} Rank"] = df["Horse"].map(
            dict(zip(valid["Horse"], valid["Rank"]))
        )

    # Consensus is rank-based only. No projected rating blend is calculated.
    rank_cols = [f"{name} Rank" for name in scenario_defs]
    df["Scenarios Available"] = df[rank_cols].notna().sum(axis=1)
    df["Consensus Rank Sum"] = df[rank_cols].sum(axis=1, skipna=True)

    consensus_df = df.loc[df["Scenarios Available"] > 0].copy()
    consensus_df = consensus_df.sort_values(
        [
            "Scenarios Available",
            "Consensus Rank Sum",
            "Established Ability Rank",
            "Latest Form Rank",
            "Peak Ability Rank",
            "Horse",
        ],
        ascending=[False, True, True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    consensus_df["Consensus Position"] = np.arange(1, len(consensus_df) + 1)

    consensus = []
    for _, row in consensus_df.head(4).iterrows():
        consensus.append({
            "position": int(row["Consensus Position"]),
            "horse": row["Horse"],
            "latest_rank": None if pd.isna(row["Latest Form Rank"]) else int(row["Latest Form Rank"]),
            "established_rank": None if pd.isna(row["Established Ability Rank"]) else int(row["Established Ability Rank"]),
            "peak_rank": None if pd.isna(row["Peak Ability Rank"]) else int(row["Peak Ability Rank"]),
            "evidence": row["Evidence"],
        })

    return {
        "rows": df,
        "scenarios": scenarios,
        "consensus": consensus,
        "reference_effective_weight": reference_weight,
        "race_date": race_date,
        "distance": distance_m,
    }


def prediction_display_table(prediction: dict) -> pd.DataFrame:
    """
    Detailed prediction table with independent 5-point rating groups.

    Group A/B/C... is calculated separately for Latest, Established and Peak.
    A new group begins when a horse is 5 or more rating points below the
    leader of the current group.
    """
    df = prediction.get("rows")
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()

    scenario_columns = [
        ("Latest Projection", "Latest Group"),
        ("Established Projection", "Established Group"),
        ("Peak Projection", "Peak Group"),
    ]

    for projection_col, group_col in scenario_columns:
        if projection_col in work.columns:
            work[group_col] = assign_rating_groups(
                work[projection_col],
                threshold_points=5.0,
            )

    cols = [
        "Horse",
        "Current MR",
        "Latest Projection",
        "Latest Group",
        "Established Projection",
        "Established Group",
        "Peak Projection",
        "Peak Group",
        "Evidence",
    ]
    out = work[[c for c in cols if c in work.columns]].copy()

    for col in [
        "Current MR",
        "Latest Projection",
        "Established Projection",
        "Peak Projection",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)

    helper = work[
        ["Horse", "Latest Form Rank", "Established Ability Rank", "Peak Ability Rank"]
    ].copy()
    rank_cols = ["Latest Form Rank", "Established Ability Rank", "Peak Ability Rank"]
    helper["Rank Sum"] = helper[rank_cols].sum(axis=1, skipna=True)
    helper["Available"] = helper[rank_cols].notna().sum(axis=1)

    out = out.merge(
        helper[["Horse", "Rank Sum", "Available"]],
        on="Horse",
        how="left",
    )
    out = out.sort_values(
        ["Available", "Rank Sum", "Horse"],
        ascending=[False, True, True],
    ).drop(columns=["Rank Sum", "Available"])

    return out.reset_index(drop=True)
