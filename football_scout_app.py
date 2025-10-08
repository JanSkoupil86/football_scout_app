# app.py
# ⚽ Wyscout Scouting — Profile Scoring App (with preset weights & Inverted Full-Back)

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(layout="wide", page_title="Wyscout Scouting App", page_icon="⚽")
st.title("⚽ Wyscout Scouting — Profile Scoring App")

st.markdown(
    """
Upload your Wyscout CSV or use the bundled file.  
Pick a **profile**, apply **filters**, tweak **weights** (presets load correctly now), and export the ranking.
"""
)

# =========================
# Data loading
# =========================
DEFAULT_PATH = "/mnt/data/Wyscout_League_Export-new.csv"

uploader = st.sidebar.file_uploader("Upload Wyscout CSV", type=["csv"])
if uploader is not None:
    df = pd.read_csv(uploader)
else:
    try:
        df = pd.read_csv(DEFAULT_PATH)
        st.sidebar.info("Loaded default dataset from: /mnt/data/Wyscout_League_Export-new.csv")
    except Exception:
        st.error("No data found. Please upload a CSV.")
        st.stop()

all_cols = df.columns.tolist()

# =========================
# Column helpers
# =========================
PLAYER_COL = "Player"
TEAM_COL = "Team"
POS_COL = "Position"
AGE_COL = "Age"
MINS_COL = "Minutes played"
LEAGUE_COL = "League"
MAIN_POS_COL = "Main Position" if "Main Position" in df.columns else POS_COL

def has(col: str) -> bool:
    return col in df.columns

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd

# Metrics where lower is better (invert before z-scoring)
LOWER_BETTER = {
    "Conceded goals per 90",
    "Fouls per 90",
    "Yellow cards per 90",
    "Red cards per 90",
    "Shots against per 90",
}

def adjust_direction(series: pd.Series, colname: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return -s if colname in LOWER_BETTER else s

def weighted_score(frame: pd.DataFrame, metrics: dict) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for col, w in metrics.items():
        if col not in frame.columns or w <= 0:
            continue
        z = zscore(adjust_direction(frame[col], col))
        score = score + z * (w / 100.0)
    return score

def normalize_weights(d: dict) -> dict:
    s = sum(d.values())
    if s <= 0:
        return d
    return {k: round(v * 100.0 / s, 2) for k, v in d.items()}

# =========================
# Profile library (exact CSV column names)
# =========================

# ---- GOALKEEPERS ----
goalkeeper_profiles = {
    "Traditional Goalkeeper": {
        "Save rate, %": 25,
        "Prevented goals per 90": 25,
        "Clean sheets": 15,
        "Shots against per 90": 10,
        "Conceded goals per 90": 10,
        "Aerial duels per 90.1": 5,
        "Exits per 90": 5,
        "Accurate long passes, %": 5,
    },
    "Sweeper Keeper": {
        "Exits per 90": 20,
        "Save rate, %": 20,
        "Prevented goals per 90": 10,
        "Clean sheets": 10,
        "Conceded goals per 90": 5,
        "Accurate long passes, %": 10,
        "Accurate short / medium passes, %": 10,
        "Aerial duels per 90.1": 5,
        "Progressive passes per 90": 10,
    },
    "Build-Up Keeper": {
        "Accurate passes, %": 20,
        "Accurate long passes, %": 15,
        "Progressive passes per 90": 20,
        "Save rate, %": 10,
        "Prevented goals per 90": 10,
        "Exits per 90": 10,
        "Clean sheets": 10,
        "Accurate short / medium passes, %": 5,
    },
}

# ---- DEFENDERS ----
defender_profiles = {
    "Ball-Playing CB": {
        "Progressive passes per 90": 20,
        "Accurate passes, %": 15,
        "Accurate long passes, %": 10,
        "Defensive duels won, %": 15,
        "Interceptions per 90": 10,
        "Aerial duels won, %": 10,
        "Deep completions per 90": 10,
        "Successful defensive actions per 90": 10,
    },
    "Libero / Middle Pin CB": {
        "Interceptions per 90": 15,
        "Defensive duels won, %": 15,
        "Progressive passes per 90": 15,
        "Deep completions per 90": 10,
        "Accurate passes, %": 10,
        "Aerial duels won, %": 10,
        "Successful defensive actions per 90": 10,
        "Accurate long passes, %": 15,
    },
    "Wide CB (Back 3)": {
        "Defensive duels won, %": 15,
        "Progressive runs per 90": 15,
        "Progressive passes per 90": 15,
        "Interceptions per 90": 10,
        "Accurate passes, %": 10,
        "Crosses per 90": 10,
        "Aerial duels won, %": 10,
        "Successful defensive actions per 90": 15,
    },
    "Full-Back / Wing-Back": {
        "Progressive runs per 90": 20,
        "Crosses per 90": 15,
        "Accurate crosses, %": 10,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 10,
        "Progressive passes per 90": 10,
        "Aerial duels won, %": 5,
        "Successful defensive actions per 90": 20,
    },
    # NEW: Inverted Full-Back
    "Inverted Full-Back": {
        "Progressive passes per 90": 15,
        "Progressive runs per 90": 10,
        "Forward passes per 90": 10 if has("Forward passes per 90") else 0,
        "Accurate passes, %": 10,
        "Accurate short / medium passes, %": 5,
        "Smart passes per 90": 5 if has("Smart passes per 90") else 0,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 10,
        "Successful defensive actions per 90": 10,
        "Aerial duels won, %": 5,
        "xA per 90" if has("xA per 90") else "xA": 5,
    },
}

# ---- MIDFIELD ----
midfield_profiles = {
    "Defensive Midfielder (6)": {
        "Interceptions per 90": 20,
        "Defensive duels won, %": 15,
        "Accurate passes, %": 15,
        "Progressive passes per 90": 10,
        "Successful defensive actions per 90": 15,
        "Forward passes per 90": 10 if has("Forward passes per 90") else 0,
        "Accurate long passes, %": 10,
        "Aerial duels won, %": 5,
    },
    "Deep-Lying Playmaker": {
        "Progressive passes per 90": 25,
        "Accurate passes, %": 15,
        "Accurate long passes, %": 10,
        "Deep completions per 90": 10,
        "Forward passes per 90": 10 if has("Forward passes per 90") else 0,
        "Interceptions per 90": 10,
        "Successful defensive actions per 90": 10,
        "Defensive duels won, %": 10,
    },
    "Box-to-Box (8)": {
        "Progressive runs per 90": 15,
        "Defensive duels won, %": 10,
        "Progressive passes per 90": 10,
        "Interceptions per 90": 10,
        "Successful defensive actions per 90": 10,
        "Shots per 90": 10,
        "xG per 90" if has("xG per 90") else "xG": 10,
        "Key passes per 90": 10,
        "Deep completions per 90": 10,
        "Accurate passes, %": 5,
    },
    "Advanced Playmaker (10)": {
        "Key passes per 90": 25,
        "xA per 90" if has("xA per 90") else "xA": 20,
        "Progressive passes per 90": 10,
        "Deep completions per 90": 10,
        "Accurate passes, %": 10,
        "Progressive runs per 90": 10,
        "Shots per 90": 5,
        "xG per 90" if has("xG per 90") else "xG": 5,
    },
}

# ---- ATTACK ----
attack_profiles = {
    "Inverted Winger": {
        "Shots per 90": 20,
        "xG per 90" if has("xG per 90") else "xG": 15,
        "xA per 90" if has("xA per 90") else "xA": 10,
        "Key passes per 90": 10,
        "Progressive runs per 90": 10,
        "Successful dribbles, %": 10,
        "Accurate crosses, %": 5,
        "Deep completions per 90": 10,
        "Touches in box per 90": 10,
    },
    "Traditional Winger": {
        "Crosses per 90": 15,
        "Accurate crosses, %": 10,
        "Dribbles per 90": 15,
        "Progressive runs per 90": 15,
        "Key passes per 90": 10,
        "xA per 90" if has("xA per 90") else "xA": 10,
        "Shots per 90": 10,
        "xG per 90" if has("xG per 90") else "xG": 10,
        "Touches in box per 90": 5,
    },
    "Striker / Advanced Forward": {
        "xG per 90" if has("xG per 90") else "xG": 25,
        "Goals per 90" if has("Goals per 90") else "Goals": 15,
        "Shots per 90": 15,
        "xA per 90" if has("xA per 90") else "xA": 10,
        "Key passes per 90": 5,
        "Progressive passes per 90": 5,
        "Touches in box per 90": 10,
        "Aerial duels won, %": 10,
        "Successful attacking actions per 90": 5,
    },
    "Second Striker / False 9": {
        "Key passes per 90": 15,
        "xA per 90" if has("xA per 90") else "xA": 15,
        "xG per 90" if has("xG per 90") else "xG": 10,
        "Shots per 90": 10,
        "Progressive passes per 90": 15,
        "Deep completions per 90": 10,
        "Dribbles per 90": 10,
        "Touches in box per 90": 10,
        "Accurate passes, %": 5,
    },
}

# Merge & clean (remove zero-weight placeholders where a column didn’t exist)
library = {}
library.update(goalkeeper_profiles)
library.update(defender_profiles)
library.update(midfield_profiles)
library.update(attack_profiles)
for role, metrics in list(library.items()):
    library[role] = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v > 0}

# =========================
# Sidebar — Filters
# =========================
st.sidebar.header("Filters")

leagues = sorted(df[LEAGUE_COL].dropna().unique().tolist()) if has(LEAGUE_COL) else []
selected_leagues = st.sidebar.multiselect("League(s)", leagues, default=leagues[:6] if leagues else [])

positions = sorted(df[MAIN_POS_COL].dropna().unique().tolist()) if has(MAIN_POS_COL) else []
selected_positions = st.sidebar.multiselect("Position(s)", positions, default=[])

min_minutes = int(pd.to_numeric(df[MINS_COL], errors="coerce").min()) if has(MINS_COL) else 0
max_minutes = int(pd.to_numeric(df[MINS_COL], errors="coerce").max()) if has(MINS_COL) else 4000
minutes_threshold = st.sidebar.slider("Minimum minutes played", min_value=min_minutes, max_value=max_minutes, value=min(600, max_minutes))

min_age = int(pd.to_numeric(df[AGE_COL], errors="coerce").min()) if has(AGE_COL) else 15
max_age = int(pd.to_numeric(df[AGE_COL], errors="coerce").max()) if has(AGE_COL) else 42
age_range = st.sidebar.slider("Age range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

# Apply filters
f = df.copy()
if selected_leagues and has(LEAGUE_COL):
    f = f[f[LEAGUE_COL].isin(selected_leagues)]
if selected_positions and has(MAIN_POS_COL):
    f = f[f[MAIN_POS_COL].isin(selected_positions)]
if has(MINS_COL):
    f = f[pd.to_numeric(f[MINS_COL], errors="coerce") >= minutes_threshold]
if has(AGE_COL):
    ages = pd.to_numeric(f[AGE_COL], errors="coerce")
    f = f[(ages >= age_range[0]) & (ages <= age_range[1])]
f = f.reset_index(drop=True)

st.caption(f"Filtered players: **{len(f)}**")

# =========================
# Profile selection + PRESET WEIGHTS (fixed)
# =========================
st.sidebar.header("Profile")
role = st.sidebar.selectbox("Select a profile", options=list(library.keys()))

# Save defaults and user-edited copies in session_state
if "profile_defaults" not in st.session_state:
    st.session_state.profile_defaults = {r: m.copy() for r, m in library.items()}
if "profile_weights" not in st.session_state:
    st.session_state.profile_weights = {r: m.copy() for r, m in library.items()}
if role not in st.session_state.profile_weights:
    st.session_state.profile_weights[role] = st.session_state.profile_defaults[role].copy()

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Reset weights to defaults"):
        st.session_state.profile_weights[role] = st.session_state.profile_defaults[role].copy()

edited = {}
with st.expander("Adjust weights for this profile (optional)", expanded=False):
    for metric, default_w in st.session_state.profile_weights[role].items():
        if metric not in f.columns:
            st.warning(f"Missing column in dataset: **{metric}** — ignored for scoring.")
        edited[metric] = st.slider(
            f"Weight %: {metric}", min_value=0, max_value=100,
            value=int(default_w), step=1, key=f"w_{role}_{metric}"
        )
st.session_state.profile_weights[role] = edited.copy()

use_normalized = st.sidebar.checkbox("Normalize weights to sum 100%", value=True)
top_n = st.sidebar.slider("Show top N", min_value=5, max_value=100, value=25)
show_cols = st.sidebar.multiselect(
    "Extra columns to display",
    options=[c for c in all_cols if c not in {"__Score__", PLAYER_COL, TEAM_COL, POS_COL, MAIN_POS_COL}],
    default=[]
)

# Prepare final weights
weights = {k: v for k, v in edited.items() if k in f.columns and v > 0}
if use_normalized and sum(weights.values()) > 0:
    weights = normalize_weights(weights)
if not weights:
    st.error("No valid metrics available for scoring (missing columns or all weights = 0). Adjust weights or pick another profile.")
    st.stop()

# =========================
# Scoring & display
# =========================
scored = f.copy()
scored["__Score__"] = weighted_score(scored, weights)
scored = scored.sort_values("__Score__", ascending=False)

base_cols = [PLAYER_COL, TEAM_COL]
if has(MAIN_POS_COL): base_cols.append(MAIN_POS_COL)
elif has(POS_COL):    base_cols.append(POS_COL)
if has(MINS_COL):     base_cols.append(MINS_COL)
if has(AGE_COL):      base_cols.append(AGE_COL)
if has(LEAGUE_COL):   base_cols.append(LEAGUE_COL)

display_cols = [c for c in base_cols + list(weights.keys()) + show_cols + ["__Score__"] if c in scored.columns]

st.subheader(f"🏆 Top {top_n} — {role}")
st.dataframe(scored[display_cols].head(top_n), use_container_width=True)

# =========================
# Download
# =========================
csv_export = scored[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download ranked results (CSV)",
    data=csv_export,
    file_name=f"{role.replace(' ', '_')}_ranked.csv",
    mime="text/csv",
)

# =========================
# Diagnostics
# =========================
with st.expander("Diagnostics: Missing Columns by Profile"):
    missing = {r: [m for m in metrics if m not in df.columns] for r, metrics in library.items()}
    st.json(missing)

with st.expander("About scoring"):
    st.markdown(
        """
**Method**  
- Each metric is standardized to a **z-score** over the **filtered dataset**.  
- Metrics flagged as *lower is better* (e.g., `Conceded goals per 90`) are inverted first.  
- Scores = sum of z-scores × **weights (%)**.  
- Toggle **Normalize weights** to force them to sum to 100%.

**Notes**  
- Missing metrics are ignored (with a warning).  
- Filters (league, minutes, age, position) are applied **before** scoring.
"""
    )
