# app.py
# ⚽ Advanced Football Scouting App — Profile Scoring (GK + Outfield)
# Run: streamlit run app.py

import math
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
Upload your data (Wyscout export) or use the preloaded file.  
Pick a **profile**, apply **filters**, and get a ranked list by **weighted score**.  
**Tip:** You can tweak weights on the fly.
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
    except Exception as e:
        st.error("No data found. Please upload a CSV.")
        st.stop()

# Clean column names (no op but handy preview)
all_cols = df.columns.tolist()

# =========================
# Helpers
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

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mu) / sd

# Metrics where **lower is better** (we invert them before z-scoring so higher is better for scoring)
LOWER_BETTER = {
    "Conceded goals per 90",
    "Fouls per 90",
    "Yellow cards per 90",
    "Red cards per 90",
    "Shots against per 90",  # optional: treat as lower-better so we don't reward workload
}

def direction_adjusted(series: pd.Series, colname: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if colname in LOWER_BETTER:
        return -s
    return s

def weighted_score(frame: pd.DataFrame, metrics: dict) -> pd.Series:
    """Compute weighted z-score using provided metric->weight dict. Skips missing columns safely."""
    score = pd.Series(0.0, index=frame.index)
    total_weight = 0
    for col, w in metrics.items():
        if col not in frame.columns:
            continue
        adj = direction_adjusted(frame[col], col)
        z = zscore(adj)
        score = score + z * (w / 100.0)
        total_weight += w
    # If weights don't sum to 100 (due to missing columns or user edits), keep as-is (relative).
    return score

def normalize_weights(d: dict) -> dict:
    s = sum(d.values())
    if s <= 0:
        return d
    return {k: round(v * 100.0 / s, 2) for k, v in d.items()}

# =========================
# Profile library (ALL in one place)
# Use exact CSV column names that exist in Wyscout exports
# =========================

# ---------- GOALKEEPERS ----------
goalkeeper_profiles = {
    "Traditional Goalkeeper": {
        "Save rate, %": 25,
        "Prevented goals per 90": 25,
        "Clean sheets": 15,
        "Shots against per 90": 10,
        "Conceded goals per 90": 10,
        "Aerial duels per 90.1": 5,         # proxy for aerial control
        "Exits per 90": 5,                  # sweeping actions
        "Accurate long passes, %": 5
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
        "Progressive passes per 90": 10
    },
    "Build-Up Keeper": {
        "Accurate passes, %": 20,
        "Accurate long passes, %": 15,
        "Progressive passes per 90": 20,
        "Save rate, %": 10,
        "Prevented goals per 90": 10,
        "Exits per 90": 10,
        "Clean sheets": 10,
        "Accurate short / medium passes, %": 5
    }
}

# ---------- DEFENDERS ----------
defender_profiles = {
    "Ball-Playing CB": {
        "Progressive passes per 90": 20,
        "Accurate passes, %": 15,
        "Accurate long passes, %": 10,
        "Defensive duels won, %": 15,
        "Interceptions per 90": 10,
        "Aerial duels won, %": 10,
        "Deep completions per 90": 10,
        "Successful defensive actions per 90": 10
    },
    "Libero / Middle Pin CB": {
        "Interceptions per 90": 15,
        "Defensive duels won, %": 15,
        "Progressive passes per 90": 15,
        "Deep completions per 90": 10,
        "Accurate passes, %": 10,
        "Aerial duels won, %": 10,
        "Successful defensive actions per 90": 10,
        "Accurate long passes, %": 15
    },
    "Wide CB (Back 3)": {
        "Defensive duels won, %": 15,
        "Progressive runs per 90": 15,
        "Progressive passes per 90": 15,
        "Interceptions per 90": 10,
        "Accurate passes, %": 10,
        "Crosses per 90": 10,
        "Aerial duels won, %": 10,
        "Successful defensive actions per 90": 15
    },
    "Full-Back / Wing-Back": {
        "Progressive runs per 90": 20,
        "Crosses per 90": 15,
        "Accurate crosses, %": 10,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 10,
        "Progressive passes per 90": 10,
        "Aerial duels won, %": 5,
        "Successful defensive actions per 90": 20
    }
}

# ---------- MIDFIELD ----------
midfield_profiles = {
    "Defensive Midfielder (6)": {
        "Interceptions per 90": 20,
        "Defensive duels won, %": 15,
        "Accurate passes, %": 15,
        "Progressive passes per 90": 10,
        "Successful defensive actions per 90": 15,
        "Forward passes per 90": 10 if has("Forward passes per 90") else 0,
        "Accurate long passes, %": 10,
        "Aerial duels won, %": 5
    },
    "Deep-Lying Playmaker": {
        "Progressive passes per 90": 25,
        "Accurate passes, %": 15,
        "Accurate long passes, %": 10,
        "Deep completions per 90": 10,
        "Forward passes per 90": 10 if has("Forward passes per 90") else 0,
        "Interceptions per 90": 10,
        "Successful defensive actions per 90": 10,
        "Defensive duels won, %": 10
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
        "Accurate passes, %": 5
    },
    "Advanced Playmaker (10)": {
        "Key passes per 90": 25,
        "xA per 90" if has("xA per 90") else "xA": 20,
        "Progressive passes per 90": 10,
        "Deep completions per 90": 10,
        "Accurate passes, %": 10,
        "Progressive runs per 90": 10,
        "Shots per 90": 5,
        "xG per 90" if has("xG per 90") else "xG": 5
    }
}

# ---------- ATTACK ----------
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
        "Touches in box per 90": 10
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
        "Touches in box per 90": 5
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
        "Successful attacking actions per 90": 5
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
        "Accurate passes, %": 5
    }
}

# Merge all profiles
library = {}
library.update(goalkeeper_profiles)
library.update(defender_profiles)
library.update(midfield_profiles)
library.update(attack_profiles)

# Remove any zero-weight placeholders (from conditional cols)
for role, metrics in list(library.items()):
    library[role] = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v > 0}

# =========================
# Sidebar — Filters
# =========================
st.sidebar.header("Filters")

# League filter
leagues = sorted(df[LEAGUE_COL].dropna().unique().tolist()) if has(LEAGUE_COL) else []
selected_leagues = st.sidebar.multiselect("League(s)", leagues, default=leagues[:6] if leagues else [])

# Position / Main Position
positions = sorted(df[MAIN_POS_COL].dropna().unique().tolist()) if has(MAIN_POS_COL) else []
selected_positions = st.sidebar.multiselect("Position(s)", positions, default=[])

# Minutes / Age
min_minutes = int(np.nanmin(pd.to_numeric(df[MINS_COL], errors="coerce"))) if has(MINS_COL) else 0
max_minutes = int(np.nanmax(pd.to_numeric(df[MINS_COL], errors="coerce"))) if has(MINS_COL) else 4000
minutes_threshold = st.sidebar.slider("Minimum minutes played", min_value=min_minutes, max_value=max_minutes, value=min(600, max_minutes))

min_age = int(np.nanmin(pd.to_numeric(df[AGE_COL], errors="coerce"))) if has(AGE_COL) else 15
max_age = int(np.nanmax(pd.to_numeric(df[AGE_COL], errors="coerce"))) if has(AGE_COL) else 45
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
# Profile selection
# =========================
st.sidebar.header("Profile")
role = st.sidebar.selectbox("Select a profile", options=list(library.keys()))

# Editable weights
with st.expander("Adjust weights for this profile (optional)", expanded=False):
    edited = {}
    for metric, w in library[role].items():
        if metric not in f.columns:
            st.warning(f"Missing column in dataset: **{metric}** — it will be ignored for scoring.")
        new_w = st.number_input(f"{metric}", min_value=0, max_value=100, value=int(w), step=1, key=f"w_{role}_{metric}")
        edited[metric] = new_w
    # Normalize preview (we do not force it, but show it)
    norm_preview = normalize_weights(edited)
    st.write("Normalized weights (sum=100):")
    st.json(norm_preview)

use_normalized = st.sidebar.checkbox("Normalize weights to sum 100%", value=True)
top_n = st.sidebar.slider("Show top N", min_value=5, max_value=100, value=25)
show_cols = st.sidebar.multiselect(
    "Extra columns to display",
    options=[c for c in all_cols if c not in {PLAYER_COL, TEAM_COL, POS_COL, MAIN_POS_COL}],
    default=[]
)

# Prepare weight dict
weights = edited if 'edited' in locals() and edited else library[role]
weights = {k: v for k, v in weights.items() if k in f.columns and v > 0}
if use_normalized and sum(weights.values()) > 0:
    weights = normalize_weights(weights)

if not weights:
    st.error("No valid metrics available for scoring (missing columns or all weights = 0). Adjust weights or pick another profile.")
    st.stop()

# =========================
# Scoring
# =========================
scored = f.copy()
scored["__Score__"] = weighted_score(scored, weights)

# Sort
scored = scored.sort_values("__Score__", ascending=False)

# =========================
# Display
# =========================
base_cols = [PLAYER_COL, TEAM_COL]
if has(MAIN_POS_COL):
    base_cols.append(MAIN_POS_COL)
elif has(POS_COL):
    base_cols.append(POS_COL)
if has(MINS_COL):
    base_cols.append(MINS_COL)
if has(AGE_COL):
    base_cols.append(AGE_COL)
if has(LEAGUE_COL):
    base_cols.append(LEAGUE_COL)

display_cols = base_cols + list(weights.keys()) + show_cols + ["__Score__"]
display_cols = [c for c in display_cols if c in scored.columns]

st.subheader(f"🏆 Top {top_n} — {role}")
st.dataframe(scored[display_cols].head(top_n), use_container_width=True)

# =========================
# Download
# =========================
csv_export = scored[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download ranked results (CSV)",
    data=csv_export,
    file_name=f"{role.replace(' ', '_')}_ranked.csv",
    mime="text/csv"
)

# =========================
# Diagnostics
# =========================
with st.expander("Diagnostics: Missing Columns by Profile"):
    missing = {}
    for r, metrics in library.items():
        missing[r] = [m for m in metrics.keys() if m not in df.columns]
    st.json(missing)

with st.expander("About scoring"):
    st.markdown(
        """
**Method**  
- Convert each metric to a **z-score** (standardize across the filtered dataset).  
- If a metric is in the *lower-is-better* list (e.g., `Conceded goals per 90`), it is inverted before z-scoring.  
- Combine using **weights (%)** → final score.  
- Weights can be normalized to sum to 100% (recommended).

**Notes**  
- If a selected metric is missing in your CSV, it’s ignored (and you’ll see a warning).  
- Filters (league, minutes, age, position) are applied **before** scoring to keep comparison fair.
"""
    )
