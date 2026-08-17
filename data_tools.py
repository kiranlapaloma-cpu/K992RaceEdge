import re
import numpy as np
import pandas as pd

from common import canon_horse

def to_num(x):
    """Coerce to numeric with NaN on failure (alias for consistency)."""
    return pd.to_numeric(x, errors="coerce")

def _first_present_value(df: pd.DataFrame | None, candidates, default=None):
    """Return the first non-empty value from the first matching column."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default
    column_map = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        col = column_map.get(str(candidate).strip().lower())
        if col is None:
            continue
        series = df[col]
        for value in series.tolist():
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except Exception:
                pass
            if str(value).strip() != "":
                return value
    return default

def _parse_race_date_value(value, default=None):
    if value is None:
        return default
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return default if pd.isna(parsed) else parsed.date()

def _normalise_going_for_pi(value: object) -> str:
    txt = str(value or "").strip().lower()
    if any(token in txt for token in ["heavy", "hvy"]):
        return "Heavy"
    if any(token in txt for token in ["soft", "v/soft", "very soft", "yield", "yld"]):
        return "Soft"
    if any(token in txt for token in ["firm", "fast"]):
        return "Firm"
    return "Good"

def _uploaded_file_preview(uploaded_file) -> pd.DataFrame | None:
    """Read a small metadata preview without consuming the uploaded file."""
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        name = str(getattr(uploaded_file, "name", "")).lower()
        preview = (
            pd.read_csv(uploaded_file)
            if name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )
        uploaded_file.seek(0)
        return preview
    except Exception:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return None

def _horse_metadata_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return canonical Horse/Age/Weight/Finish/Official MR data when present."""
    if df is None or not isinstance(df, pd.DataFrame) or "Horse" not in df.columns:
        return pd.DataFrame(columns=["Horse", "Age", "Horse Weight", "Finish_Pos", "Official MR"])
    out = pd.DataFrame({"Horse": df["Horse"].astype(str)})
    aliases = {
        "Age": ["Age", "A"],
        "Horse Weight": ["Horse Weight", "Horse_Weight", "Wt", "Weight", "Weight (kg)", "Wgh"],
        "Finish_Pos": ["Finish_Pos", "Finish Position", "Fin", "Position"],
        "Official MR": ["Official MR", "Official_MR", "MR", "Merit Rating"],
    }
    cmap = {str(c).strip().lower(): c for c in df.columns}
    for target, candidates in aliases.items():
        source = next((cmap.get(str(c).strip().lower()) for c in candidates if cmap.get(str(c).strip().lower()) is not None), None)
        out[target] = df[source] if source is not None else pd.NA
    out["Horse Key"] = out["Horse"].map(canon_horse)
    return out

def normalize_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize common variants (case-insensitive):
      • '<meters>_time' or '<meters>m_time'      -> '<meters>_Time'
      • '<meters>_split' or '<meters>m_split'    -> '<meters>_Time'
      • 'finish_time' / 'finish_split' / 'finish'-> 'Finish_Time'
      • 'finish_pos'                              -> 'Finish_Pos'
      • pass-through every other column
    """
    notes = []
    lmap = {c.lower().strip().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    def alias(src_key, alias_col):
        nonlocal df, notes
        if src_key in lmap and alias_col not in df.columns:
            df[alias_col] = df[lmap[src_key]]
            notes.append(f"Aliased `{lmap[src_key]}` → `{alias_col}`")

    # Finish variants (for 200m data, this is 200→0)
    for k in ("finish_time", "finish_split", "finish"):
        alias(k, "Finish_Time")
    alias("finish_pos", "Finish_Pos")

    # Segment columns: accept optional 'm' before the underscore
    pat = re.compile(r"^(\d{2,4})m?_(time|split)$")
    for lk, orig in lmap.items():
        m = pat.match(lk)
        if m:
            alias_col = f"{m.group(1)}_Time"
            if alias_col not in df.columns:
                df[alias_col] = df[orig]
                notes.append(f"Aliased `{orig}` → `{alias_col}`")

    return df, notes

def detect_step(df: pd.DataFrame) -> int:
    """
    Detect whether the splits are 100m or 200m based on gaps between *_Time columns.
    """
    markers = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                markers.append(int(c.split("_")[0]))
            except Exception:
                pass
    markers = sorted(set(markers), reverse=True)
    if len(markers) < 2:
        return 100
    diffs = [markers[i] - markers[i+1] for i in range(len(markers)-1)]
    cnt100 = sum(60 <= d <= 140 for d in diffs)
    cnt200 = sum(160 <= d <= 240 for d in diffs)
    return 200 if cnt200 > cnt100 else 100

def normalize_200m_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace("\u2013","-").replace("\u2014","-") for c in df.columns]
    # coerce obvious numeric fields
    for c in df.columns:
        if c.endswith("_Time") or c.endswith("_Pos") or c in ("Race Time","800-400","400-Finish","Horse Weight","Weight Allocated","Finish_Time","Finish_Pos"):
            df[c] = to_num(df[c])
    if "Finish_Pos" not in df.columns:
        df["Finish_Pos"] = np.arange(1, len(df) + 1)
    return df

def expected_segments_from_df(df: pd.DataFrame) -> list[str]:
    """Use only *_Time columns that actually exist (highest→lowest) + Finish_Time if present."""
    marks = []
    for c in df.columns:
        if c.endswith("_Time") and c != "Finish_Time":
            try:
                marks.append(int(c.split("_")[0]))
            except Exception:
                pass
    marks = sorted(set(marks), reverse=True)
    cols = [f"{m}_Time" for m in marks if f"{m}_Time" in df.columns]
    if "Finish_Time" in df.columns:
        cols.append("Finish_Time")
    return cols

def integrity_scan(df: pd.DataFrame, distance_m: float, step: int):
    """
    Validate only the columns that truly exist in the file.
    Reports rows where times are <=0 or NaN (treated as missing).
    """
    exp_cols = expected_segments_from_df(df)
    missing = []  # by construction we only check columns that exist
    invalid_counts = {}
    for c in exp_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        invalid_counts[c] = int(((s <= 0) | s.isna()).sum())
    msgs = []
    bads = [f"{k} ({v} rows)" for k, v in invalid_counts.items() if v > 0]
    if bads:
        msgs.append("Invalid/zero times → treated as missing: " + ", ".join(bads))
    return " • ".join(msgs), missing, invalid_counts
