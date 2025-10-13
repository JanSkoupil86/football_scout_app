# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re

# =========================
# Page config
# =========================
st.set_page_config(layout="wide", page_title="Advanced Football Scouting App", page_icon="⚽")
st.title("⚽ Advanced Football Player Scouting App — Season-aware Edition")
st.markdown(
    "Upload your football data CSV to analyze player metrics. "
    "Includes season normalization, robust metric aliasing, no market value bar, "
    "Built-in & Custom profiles with recommended weights, direction-aware z-scores, "
    "sparse-metric skipping, Top-N tables, scatter & radar, and CSV downloads."
)

# =========================
# Constants & Utilities
# =========================
PCT_SUFFIX = ", %"  # dataset columns typically use "..., %"
NON_FEATURE_COLUMNS = {
    'Column1', 'Player', 'Team', 'Team within selected timeframe', 'Position', 'Birth country',
    'Passport country', 'Foot', 'On loan', 'Contract expires', 'League', 'Main Position'
}

def multiselect_all(label: str, options: list, *, default_all: bool = True,
                    help: str | None = None, key: str | None = None):
    """A multiselect with an 'ALL' option that maps to all items."""
    all_token = "ALL"
    opts = [all_token] + options
    default = [all_token] if default_all else []
    picked = st.sidebar.multiselect(label, opts, default=default or None, help=help, key=key)
    use_all = (not picked) or (all_token in picked)
    return (options if use_all else [o for o in picked if o != all_token]), use_all

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")

def parse_market_value(series: pd.Series) -> pd.Series:
    """Parse '€12.5m', '€800k', '12,000,000' into float (EUR M)."""
    if series.dtype.kind in 'iuf':
        s = series.astype(float)
        return (s / 1e6) if s.max() > 1e6 else s

    def to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace('€', '').replace(',', '').lower()
        mult = 1.0
        if s.endswith('m'):
            mult = 1_000_000.0; s = s[:-1]
        elif s.endswith('k'):
            mult = 1_000.0; s = s[:-1]
        try:
            val = float(s) * mult
        except ValueError:
            return np.nan
        return val / 1e6
    return series.apply(to_float)

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Market value' in df.columns:
        df['Market value (M€)'] = parse_market_value(df['Market value'])
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col == 'Market value':
            continue
        if col.endswith(PCT_SUFFIX):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', '', regex=False), errors='coerce')
        else:
            if df[col].dtype.kind not in 'iuf':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def get_numeric_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]

# ---- Alias resolver (handles commas, spaces, percent signs, minor spelling) ----
def _norm(s: str) -> str:
    return re.sub(r'[\s,%–-]+', '', s).lower()

def resolve_metrics_aliases(requested: list[str], columns: list[str]) -> tuple[list[str], list[str]]:
    col_norm_map = {_norm(c): c for c in columns}
    resolved, missing = [], []
    for name in requested:
        # try exact
        if name in columns:
            resolved.append(name); continue
        # common “%” variants (e.g., "Accurate passes %" -> "Accurate passes, %")
        alt = name
        if "%" in name and ", %" not in name:
            alt = name.replace(" %", ", %").replace("%", ", %")
            if alt in columns:
                resolved.append(alt); continue
        # try normalized match
        key = _norm(name)
        if key in col_norm_map:
            resolved.append(col_norm_map[key]); continue
        missing.append(name)
    return resolved, missing

# ---- Direction-aware z-score & sparse detection ----
LOWER_IS_BETTER = {
    "Conceded goals per 90",
    "Fouls per 90",
    "Turnovers per 90",
    "Miscontrols per 90",
    "Yellow cards per 90",
    "Red cards per 90",
}

def _zscore_directional(series: pd.Series, metric_name: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    z = (s - m) / sd
    return -z if metric_name in LOWER_IS_BETTER else z

def _is_sparse_metric(series: pd.Series, zero_nan_ratio_threshold: float = 0.95) -> bool:
    """True if ≥ threshold of values are 0 or NaN (too sparse to score)."""
    s = pd.to_numeric(series, errors="coerce")
    if len(s) == 0:
        return True
    zeros_or_nan = s.isna() | (s == 0)
    return float(zeros_or_nan.mean()) >= zero_nan_ratio_threshold

def normalize_weights(pcts: np.ndarray) -> np.ndarray:
    total = pcts.sum()
    return (pcts / total) if total else (np.ones_like(pcts) / len(pcts) if len(pcts) else pcts)

def make_profile_score(df: pd.DataFrame, metrics: list[str], weights: np.ndarray, new_col: str) -> pd.DataFrame:
    """Weighted, direction-aware z-score with sparse-metric skipping."""
    present = [m for m in metrics if m in df.columns]
    if not present:
        df[new_col] = 0.0
        return df

    usable_metrics = [m for m in present if not _is_sparse_metric(df[m])]
    skipped_sparse = [m for m in present if m not in usable_metrics]

    if not usable_metrics:
        df[new_col] = 0.0
        try:
            st.info(f"All selected metrics for **{new_col}** are too sparse to score.")
        except Exception:
            pass
        return df

    w_map = {m: w for m, w in zip(metrics, weights)}
    used_weights = np.array([w_map[m] for m in usable_metrics], dtype=float)
    used_weights = normalize_weights(used_weights)

    z_cols = []
    for m in usable_metrics:
        z = _zscore_directional(df[m], m)
        z_cols.append(z.values.reshape(-1, 1))
    Z = np.hstack(z_cols)
    df[new_col] = np.round((Z * used_weights.reshape(1, -1)).sum(axis=1), 2)

    if skipped_sparse:
        try:
            st.caption("⚠️ Skipped sparse metrics (≥95% zeros/NaNs): " + ", ".join(skipped_sparse))
        except Exception:
            pass
    return df

# =========================
# Final PROFILES + Default Weights (dataset-aligned)
# =========================
PROFILES = {
    # ---------------- GOALKEEPERS ----------------
    "Classic Goalkeeper": [
        "Save rate, %", "Prevented goals per 90", "Conceded goals per 90", "Shots against per 90",
        "Aerial duels won, %", "Exits per 90", "Aerial duels per 90",
        "Accurate long passes, %", "Accurate passes, %", "Average pass length, m"
    ],
    "Sweeper Keeper": [
        "Exits per 90", "Aerial duels per 90", "Aerial duels won, %", "Shots against per 90",
        "Prevented goals per 90", "Save rate, %", "Progressive passes per 90",
        "Forward passes per 90", "Accurate long passes, %", "Passes to final third per 90"
    ],
    "Build-Up Keeper": [
        "Accurate passes, %", "Accurate long passes, %", "Progressive passes per 90", "Forward passes per 90",
        "Passes to final third per 90", "Average pass length, m", "Passes per 90",
        "Save rate, %", "Prevented goals per 90", "Exits per 90"
    ],

    # ---------------- CENTRE-BACKS ----------------
    "Ball-Playing CB": [
        "Passes per 90", "Progressive passes per 90", "Accurate progressive passes, %", "Accurate long passes, %",
        "Forward passes per 90", "Interceptions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels won, %", "Successful defensive actions per 90"
    ],
    "Combative CB / Stopper": [
        "Defensive duels per 90", "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Shots blocked per 90", "Interceptions per 90", "Fouls per 90",
        "Successful defensive actions per 90", "Passes per 90", "Accurate passes, %"
    ],
    "Libero / Middle Pin CB": [
        "Progressive passes per 90", "Accurate passes, %", "Accurate long passes, %", "Passes to final third per 90",
        "Interceptions per 90", "Defensive duels per 90", "Defensive duels won, %",
        "Deep completions per 90", "Aerial duels won, %", "Passes per 90"
    ],
    "Wide CB (in 3)": [
        "Defensive duels per 90", "Defensive duels won, %", "Progressive runs per 90", "Progressive passes per 90",
        "Accurate progressive passes, %", "Interceptions per 90", "Aerial duels won, %",
        "Successful defensive actions per 90", "Deep completions per 90", "Crosses per 90"
    ],

    # ---------------- MIDFIELDERS ----------------
    "Defensive Midfielder #6": [
        "Interceptions per 90", "Defensive duels per 90", "Defensive duels won, %", "Successful defensive actions per 90",
        "Accurate passes, %", "Forward passes per 90", "Passes to final third per 90",
        "Progressive passes per 90", "Average pass length, m", "Aerial duels won, %"
    ],
    "Attacking Midfielder #8": [
        "Progressive passes per 90", "Accurate progressive passes, %", "Progressive runs per 90", "xA per 90",
        "Shots per 90", "Touches in box per 90", "Interceptions per 90",
        "Key passes per 90", "Deep completions per 90", "Successful attacking actions per 90"
    ],
    "Deep-Lying Playmaker": [
        "Received passes per 90", "Passes per 90", "Progressive passes per 90", "Accurate progressive passes, %",
        "Accurate long passes, %", "Forward passes per 90", "Passes to final third per 90",
        "Interceptions per 90", "Defensive duels per 90", "Aerial duels won, %"
    ],
    "Box-to-Box Midfielder": [
        "Progressive runs per 90", "xG per 90", "Shots per 90", "Interceptions per 90",
        "Defensive duels per 90", "Defensive duels won, %", "Touches in box per 90",
        "Successful attacking actions per 90", "Forward passes per 90", "Passes to final third per 90"
    ],

    # ------------- WIDE & ATTACKING ROLES -------------
    "Full-Back": [
        "Defensive duels per 90", "Defensive duels won, %", "Interceptions per 90", "Crosses per 90",
        "Accurate crosses, %", "Progressive runs per 90", "Progressive passes per 90",
        "Forward passes per 90", "Successful defensive actions per 90", "Deep completions per 90"
    ],
    "Wing-Back": [
        "Progressive runs per 90", "Crosses per 90", "Accurate crosses, %", "Shot assists per 90",
        "Progressive passes per 90", "Interceptions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Touches in box per 90", "Successful attacking actions per 90"
    ],
    "Inverted Full-Back": [
        "Progressive passes per 90", "Progressive runs per 90", "Forward passes per 90", "Accurate passes, %",
        "Accurate short / medium passes, %", "Smart passes per 90", "Defensive duels won, %",
        "Interceptions per 90", "Successful defensive actions per 90", "Aerial duels won, %"
    ],
    "Classic Winger": [
        "Dribbles per 90", "Successful dribbles, %", "Progressive runs per 90", "Crosses per 90",
        "Accurate crosses, %", "Shot assists per 90", "Touches in box per 90",
        "Shots per 90", "xA per 90", "Successful attacking actions per 90"
    ],
    "Inverted Winger": [
        "Shots per 90", "xG per 90", "xA per 90", "Progressive runs per 90",
        "Shot assists per 90", "Touches in box per 90", "Dribbles per 90",
        "Successful dribbles, %", "Deep completions per 90", "Key passes per 90"
    ],
    "Playmaker #10": [
        "Progressive passes per 90", "Accurate progressive passes, %", "Deep completions per 90", "Key passes per 90",
        "xA per 90", "Shot assists per 90", "Shots per 90", "xG per 90",
        "Progressive runs per 90", "Successful attacking actions per 90"
    ],

    # ------------------- FORWARDS -------------------
    "Target Man #9": [
        "Aerial duels per 90", "Aerial duels won, %", "Received long passes per 90", "Passes to final third per 90",
        "Fouls suffered per 90", "xG per 90", "Shots per 90",
        "Non-penalty goals per 90", "Touches in box per 90", "Received passes per 90"
    ],
    "Poacher": [
        "Non-penalty goals per 90", "xG per 90", "Shots per 90", "Goal conversion, %",
        "Touches in box per 90", "Received passes per 90", "xA per 90",
        "Progressive runs per 90", "Key passes per 90", "Successful attacking actions per 90"
    ],
    "Pressing Forward": [
        "Defensive duels per 90", "Defensive duels won, %", "Interceptions per 90", "Successful defensive actions per 90",
        "Progressive runs per 90", "Shots per 90", "xG per 90",
        "xA per 90", "Touches in box per 90", "Successful attacking actions per 90"
    ],
}

DEFAULT_WEIGHTS = {
    # --------------- GK ---------------
    "Classic Goalkeeper": {
        "Save rate, %": 30, "Prevented goals per 90": 20, "Conceded goals per 90": 5, "Shots against per 90": 10,
        "Aerial duels won, %": 10, "Exits per 90": 5, "Aerial duels per 90": 5,
        "Accurate long passes, %": 5, "Accurate passes, %": 5, "Average pass length, m": 5
    },
    "Sweeper Keeper": {
        "Exits per 90": 25, "Aerial duels per 90": 10, "Aerial duels won, %": 10, "Shots against per 90": 5,
        "Prevented goals per 90": 10, "Save rate, %": 10, "Progressive passes per 90": 10,
        "Forward passes per 90": 10, "Accurate long passes, %": 5, "Passes to final third per 90": 5
    },
    "Build-Up Keeper": {
        "Accurate passes, %": 25, "Accurate long passes, %": 15, "Progressive passes per 90": 15, "Forward passes per 90": 10,
        "Passes to final third per 90": 10, "Average pass length, m": 10, "Passes per 90": 5,
        "Save rate, %": 5, "Prevented goals per 90": 5, "Exits per 90": 0  # keep as 0 if you prefer 9 active metrics = 95 -> normalize in UI
    },

    # --------------- CB ---------------
    "Ball-Playing CB": {
        "Passes per 90": 10, "Progressive passes per 90": 25, "Accurate progressive passes, %": 20, "Accurate long passes, %": 10,
        "Forward passes per 90": 10, "Interceptions per 90": 10, "Defensive duels per 90": 5,
        "Defensive duels won, %": 5, "Aerial duels won, %": 5, "Successful defensive actions per 90": 0
    },
    "Combative CB / Stopper": {
        "Defensive duels per 90": 20, "Defensive duels won, %": 20, "Aerial duels per 90": 15, "Aerial duels won, %": 15,
        "Shots blocked per 90": 10, "Interceptions per 90": 10, "Fouls per 90": 5,
        "Successful defensive actions per 90": 5, "Passes per 90": 0, "Accurate passes, %": 0
    },
    "Libero / Middle Pin CB": {
        "Progressive passes per 90": 20, "Accurate passes, %": 10, "Accurate long passes, %": 15, "Passes to final third per 90": 15,
        "Interceptions per 90": 10, "Defensive duels per 90": 5, "Defensive duels won, %": 5,
        "Deep completions per 90": 10, "Aerial duels won, %": 5, "Passes per 90": 5
    },
    "Wide CB (in 3)": {
        "Defensive duels per 90": 15, "Defensive duels won, %": 10, "Progressive runs per 90": 15, "Progressive passes per 90": 15,
        "Accurate progressive passes, %": 10, "Interceptions per 90": 10, "Aerial duels won, %": 5,
        "Successful defensive actions per 90": 10, "Deep completions per 90": 5, "Crosses per 90": 5
    },

    # --------------- MID ---------------
    "Defensive Midfielder #6": {
        "Interceptions per 90": 20, "Defensive duels per 90": 15, "Defensive duels won, %": 10, "Successful defensive actions per 90": 10,
        "Accurate passes, %": 10, "Forward passes per 90": 10, "Passes to final third per 90": 10,
        "Progressive passes per 90": 10, "Average pass length, m": 5, "Aerial duels won, %": 0
    },
    "Attacking Midfielder #8": {
        "Progressive passes per 90": 15, "Accurate progressive passes, %": 10, "Progressive runs per 90": 15, "xA per 90": 15,
        "Shots per 90": 15, "Touches in box per 90": 10, "Interceptions per 90": 5,
        "Key passes per 90": 5, "Deep completions per 90": 5, "Successful attacking actions per 90": 5
    },
    "Deep-Lying Playmaker": {
        "Received passes per 90": 10, "Passes per 90": 10, "Progressive passes per 90": 20, "Accurate progressive passes, %": 15,
        "Accurate long passes, %": 10, "Forward passes per 90": 10, "Passes to final third per 90": 10,
        "Interceptions per 90": 5, "Defensive duels per 90": 5, "Aerial duels won, %": 5
    },
    "Box-to-Box Midfielder": {
        "Progressive runs per 90": 20, "xG per 90": 15, "Shots per 90": 15, "Interceptions per 90": 10,
        "Defensive duels per 90": 10, "Defensive duels won, %": 5, "Touches in box per 90": 10,
        "Successful attacking actions per 90": 10, "Forward passes per 90": 5, "Passes to final third per 90": 0
    },

    # -------- WIDE & ATTACKING ROLES --------
    "Full-Back": {
        "Defensive duels per 90": 15, "Defensive duels won, %": 10, "Interceptions per 90": 10, "Crosses per 90": 10,
        "Accurate crosses, %": 10, "Progressive runs per 90": 15, "Progressive passes per 90": 10,
        "Forward passes per 90": 10, "Successful defensive actions per 90": 5, "Deep completions per 90": 5
    },
    "Wing-Back": {
        "Progressive runs per 90": 20, "Crosses per 90": 15, "Accurate crosses, %": 10, "Shot assists per 90": 15,
        "Progressive passes per 90": 10, "Interceptions per 90": 5, "Defensive duels per 90": 10,
        "Defensive duels won, %": 5, "Touches in box per 90": 5, "Successful attacking actions per 90": 5
    },
    "Inverted Full-Back": {
        "Progressive passes per 90": 15, "Progressive runs per 90": 10, "Forward passes per 90": 10, "Accurate passes, %": 10,
        "Accurate short / medium passes, %": 5, "Smart passes per 90": 5, "Defensive duels won, %": 10,
        "Interceptions per 90": 10, "Successful defensive actions per 90": 10, "Aerial duels won, %": 5
    },
    "Classic Winger": {
        "Dribbles per 90": 20, "Successful dribbles, %": 15, "Progressive runs per 90": 15, "Crosses per 90": 10,
        "Accurate crosses, %": 10, "Shot assists per 90": 10, "Touches in box per 90": 5,
        "Shots per 90": 5, "xA per 90": 5, "Successful attacking actions per 90": 5
    },
    "Inverted Winger": {
        "Shots per 90": 20, "xG per 90": 15, "xA per 90": 10, "Progressive runs per 90": 15,
        "Shot assists per 90": 10, "Touches in box per 90": 5, "Dribbles per 90": 10,
        "Successful dribbles, %": 5, "Deep completions per 90": 5, "Key passes per 90": 5
    },
    "Playmaker #10": {
        "Progressive passes per 90": 20, "Accurate progressive passes, %": 15, "Deep completions per 90": 15, "Key passes per 90": 10,
        "xA per 90": 10, "Shot assists per 90": 10, "Shots per 90": 5, "xG per 90": 5,
        "Progressive runs per 90": 5, "Successful attacking actions per 90": 5
    },

    # --------------- FWD ---------------
    "Target Man #9": {
        "Aerial duels per 90": 20, "Aerial duels won, %": 15, "Received long passes per 90": 10, "Passes to final third per 90": 10,
        "Fouls suffered per 90": 10, "xG per 90": 10, "Shots per 90": 5,
        "Non-penalty goals per 90": 5, "Touches in box per 90": 5, "Received passes per 90": 10
    },
    "Poacher": {
        "Non-penalty goals per 90": 30, "xG per 90": 20, "Shots per 90": 10, "Goal conversion, %": 10,
        "Touches in box per 90": 10, "Received passes per 90": 5, "xA per 90": 5,
        "Progressive runs per 90": 5, "Key passes per 90": 0, "Successful attacking actions per 90": 5
    },
    "Pressing Forward": {
        "Defensive duels per 90": 15, "Defensive duels won, %": 10, "Interceptions per 90": 15, "Successful defensive actions per 90": 10,
        "Progressive runs per 90": 10, "Shots per 90": 15, "xG per 90": 10,
        "xA per 90": 5, "Touches in box per 90": 5, "Successful attacking actions per 90": 5
    }
}

# --- NEW: map default weights to resolved names (after aliasing) ---
def _defaults_for_resolved(profile_name: str, resolved_metric_names: list[str]) -> list[int]:
    """Map DEFAULT_WEIGHTS[profile] to the resolved dataset column names.
    Falls back to equal split for metrics missing in DEFAULT_WEIGHTS."""
    dm = DEFAULT_WEIGHTS.get(profile_name, {})
    dm_norm = {_norm(k): v for k, v in dm.items()}
    if not resolved_metric_names:
        return []
    equal = max(1, int(100 / len(resolved_metric_names)))
    return [int(dm_norm.get(_norm(m), equal)) for m in resolved_metric_names]

# =========================
# Upload & Season parsing
# =========================
uploaded = st.sidebar.file_uploader("Upload your Football Data CSV", type=["csv"])
if uploaded is None:
    st.info("Please upload your football data CSV (e.g., Wyscout export).")
    st.stop()

try:
    df_raw = load_csv(uploaded)
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty."); st.stop()
except pd.errors.ParserError:
    st.error("Could not parse the CSV file. Please ensure it is a valid CSV."); st.stop()

required_cols = ['Player', 'Team', 'Main Position', 'Age', 'League']
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing critical column(s): {', '.join(missing)}.")
    st.stop()

if 'Column1' in df_raw.columns and df_raw['Column1'].nunique() == len(df_raw):
    df_raw = df_raw.drop(columns=['Column1'])

df = coerce_numeric(df_raw)

# --- Season columns from "League" tail (e.g. "2024-25") ---
SEASON_RX = re.compile(r'(\d{4})(?:\s*-\s*(\d{2,4}))?$')
def extract_season_label(league_value: str) -> str | None:
    if not isinstance(league_value, str):
        return None
    parts = league_value.strip().split()
    if not parts:
        return None
    tail = parts[-1]
    return tail if SEASON_RX.fullmatch(tail) else None

def season_start_end(season_label: str) -> tuple[int, int]:
    m = SEASON_RX.fullmatch(season_label)
    if not m:
        return (None, None)
    start = int(m.group(1))
    end_raw = m.group(2)
    if not end_raw:
        return (start, start)
    end = int(end_raw)
    if end < 100:
        end = 2000 + end if end < 70 else 1900 + end
    return (start, end)

df['Season label'] = df['League'].apply(extract_season_label)
df[['Season start', 'Season end']] = (
    df['Season label'].apply(lambda s: season_start_end(s) if isinstance(s, str) else (None, None)).apply(pd.Series)
)
df['Season group'] = df['Season start'].fillna(df['Season end']).astype('Int64')

# numeric rounding
for col in get_numeric_columns(df):
    df[col] = df[col].fillna(0).round(2)

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

# Seasons first
st.sidebar.subheader("Season")
season_groups = sorted(df['Season group'].dropna().unique().tolist())
selected_groups = st.sidebar.multiselect("Season group (start year)", season_groups, default=season_groups)
use_exact_label = st.sidebar.checkbox("Also filter by exact Season label", value=False)
if use_exact_label:
    season_labels = sorted(df['Season label'].dropna().unique().tolist())
    selected_labels = st.sidebar.multiselect("Season label(s)", season_labels, default=season_labels)
else:
    selected_labels = None

season_mask = df['Season group'].isin(selected_groups)
if use_exact_label and selected_labels:
    season_mask &= df['Season label'].isin(selected_labels)
df = df.loc[season_mask].copy()

# League / Team / Position filters
leagues = sorted(df['League'].dropna().unique().tolist())
selected_leagues, _ = multiselect_all("League(s)", leagues, default_all=True, help="Choose specific leagues or ALL")
filtered = df[df['League'].isin(selected_leagues)].copy()
if filtered.empty:
    st.warning("No players found for selected seasons/leagues."); st.stop()

teams = sorted(filtered['Team'].dropna().unique().tolist())
positions = sorted(filtered['Main Position'].dropna().unique().tolist())
selected_teams, _ = multiselect_all("Team(s)", teams, default_all=True)
selected_positions, _ = multiselect_all("Main Position(s)", positions, default_all=True)

age_min, age_max = int(filtered['Age'].min()), int(filtered['Age'].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

# minutes threshold (market value slider intentionally removed)
if 'Minutes played' in filtered.columns:
    min_minutes_max = int(filtered['Minutes played'].max())
    min_minutes = st.sidebar.slider("Minimum minutes this season", 0, max(0, min_minutes_max),
                                    min(500, max(0, min_minutes_max)))
else:
    min_minutes = 0

remove_outliers = st.sidebar.checkbox("Remove outliers (|Z| > 3) — for plots only", value=False)

mask = (
    filtered['Team'].isin(selected_teams) &
    filtered['Main Position'].isin(selected_positions) &
    filtered['Age'].between(age_range[0], age_range[1])
)
if 'Minutes played' in filtered.columns:
    mask &= filtered['Minutes played'] >= min_minutes

filtered = filtered.loc[mask].copy()
for col in get_numeric_columns(filtered):
    filtered[col] = filtered[col].round(2)

st.sidebar.markdown(f"**Players matching filters: {len(filtered)}**")
if filtered.empty:
    st.warning("No players match the selected filters."); st.stop()

# =========================
# Player Profiles (Built-in + Custom)  — with PRESET %
# =========================
calc_col_name = None
profile_metrics_in_use: list[str] = []

with st.sidebar.expander("Player profiles (calculated z-score)", expanded=True):
    st.caption("Scores are weighted sums of direction-aware z-scores across the currently filtered players.")
    mode = st.radio("Profile mode", ["Built-in", "Custom"], index=0, horizontal=True)

    if mode == "Built-in":
        profile = st.selectbox("Choose profile", list(PROFILES.keys()))
        requested_metrics = PROFILES[profile]

        # Resolve aliases to actual dataset columns
        resolved_metrics, missing_names = resolve_metrics_aliases(requested_metrics, filtered.columns.tolist())
        if missing_names:
            st.info("Skipped missing metrics: " + ", ".join(missing_names))

        if resolved_metrics:
            profile_metrics_in_use = resolved_metrics.copy()

            # --- preset % stored per profile (persist across switches) ---
            ss_key = f"preset::{profile}"
            if ss_key not in st.session_state:
                st.session_state[ss_key] = {
                    "metrics": resolved_metrics[:],
                    "weights": _defaults_for_resolved(profile, resolved_metrics)
                }
            else:
                state = st.session_state[ss_key]
                # If metrics set changed due to aliasing/filtering, reseed while keeping any matching edits
                if state.get("metrics") != resolved_metrics:
                    old_map = {m: w for m, w in zip(state["metrics"], state["weights"])}
                    new_defaults = _defaults_for_resolved(profile, resolved_metrics)
                    new_weights = [int(old_map.get(m, d)) for m, d in zip(resolved_metrics, new_defaults)]
                    st.session_state[ss_key] = {"metrics": resolved_metrics[:], "weights": new_weights}

            state = st.session_state[ss_key]

            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Reset weights to defaults"):
                    state["weights"] = _defaults_for_resolved(profile, resolved_metrics)

            # Sliders seeded with PRESETs (not equal splits)
            weights_pct = []
            for i, (m, dflt) in enumerate(zip(state["metrics"], state["weights"]), start=1):
                if m not in filtered.columns:
                    st.warning(f"Missing column: **{m}** — ignored for scoring.")
                w = st.slider(f"Weight %: {m}", 0, 100, int(dflt), 1, key=f"w_{profile}_{i}")
                weights_pct.append(w)

            # Persist edits
            st.session_state[ss_key]["weights"] = [int(w) for w in weights_pct]

            weights_pct = np.array(weights_pct, dtype=float)
            if int(weights_pct.sum()) != 100:
                st.warning(f"Total weight ≠ 100 (currently {int(weights_pct.sum())}). We’ll normalize for the score.")
            weights = normalize_weights(weights_pct)

            calc_col_name = f"Score: {profile}"
            filtered = make_profile_score(filtered, resolved_metrics, weights, calc_col_name)
            st.caption(f"✅ Added column **{calc_col_name}**.")
        else:
            st.warning("No valid metrics for this profile in the current dataset.")

    else:
        st.subheader("Custom Profile")

        # 1) Name first
        custom_name: str = st.text_input("Profile name", value="Custom Profile").strip() or "Custom Profile"

        # 2) Metrics (numeric only)
        numeric_cols = get_numeric_columns(filtered)
        custom_metrics = st.multiselect("Pick metrics to include", options=numeric_cols, default=numeric_cols[:5])

        if custom_metrics:
            profile_metrics_in_use = custom_metrics.copy()

            default_pct = max(1, int(100 / len(custom_metrics)))
            weights_pct = []
            for i, m in enumerate(custom_metrics, start=1):
                weights_pct.append(st.slider(f"Weight %: {m}", 0, 100, default_pct, 1, key=f"w_custom_{i}"))
            weights_pct = np.array(weights_pct, dtype=float)
            if int(weights_pct.sum()) != 100:
                st.warning(f"Total weight ≠ 100 (currently {int(weights_pct.sum())}). We’ll normalize for the score.")
            weights = normalize_weights(weights_pct)

            calc_col_name = f"Score: {custom_name}"
            filtered = make_profile_score(filtered, custom_metrics, weights, calc_col_name)
            st.caption(f"✅ Added column **{calc_col_name}**.")
        else:
            st.info("Select at least one metric to build a custom profile.")

# =========================
# Data table (Top-N)
# =========================
st.subheader("Filtered Player Data")

def try_reorder(items: list, *, key: str, direction: str = "horizontal") -> list:
    try:
        from streamlit_sortable import sort_items  # type: ignore
        ordered = sort_items(items=items, direction=direction, key=key)
        return ordered or items
    except Exception:
        try:
            from streamlit_sortables import sort_items  # type: ignore
            ordered = sort_items(items=items, direction=direction, key=key)
            return ordered or items
        except Exception:
            return items

num_cols_for_rank = get_numeric_columns(filtered)
_default_rank = next(
    (m for m in [calc_col_name, 'Assists per 90','Goals per 90','xA per 90','xG per 90','xA','xG'] if m and m in filtered.columns),
    (num_cols_for_rank[0] if num_cols_for_rank else 'Minutes played')
)
rank_metric = st.session_state.get('rank_metric', _default_rank)

ID_COLS = ['Season label','Player','Team','League','Main Position','Age',
           'Market value (M€)' if 'Market value (M€)' in filtered.columns else 'Market value',
           'Goals','Assists','xG','xA','Minutes played']
default_cols = [c for c in ID_COLS + ([calc_col_name] if calc_col_name and calc_col_name in filtered.columns else []) if c in filtered.columns]

exclude_cols = set()
if 'Market value (M€)' in filtered.columns:
    exclude_cols.add('Market value')

display_options = [c for c in filtered.columns if c not in exclude_cols]
selected_display_cols = st.multiselect("Columns to display", options=display_options, default=default_cols)
ordered_display_cols = try_reorder(selected_display_cols, key="order_display_cols")

if selected_display_cols:
    rank_by = st.selectbox("Sort Top-N rows by", options=num_cols_for_rank,
                           index=num_cols_for_rank.index(rank_metric) if rank_metric in num_cols_for_rank else 0)
    row_limit = st.slider(f"Number of rows to show (Top-N by {rank_by})", 1, 30, 15)
    rank_metric = rank_by
    table_df = (
        filtered.sort_values(by=rank_metric, ascending=False)
                [ordered_display_cols]
                .reset_index(drop=True)
                .head(row_limit)
    )
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("Please select at least one column to display.")

csv_buf = StringIO()
(filtered[ordered_display_cols or default_cols]).to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered data (CSV)", data=csv_buf.getvalue(),
                   file_name="filtered_players.csv", mime="text/csv")

# =========================
# Scatter plot
# =========================
st.subheader("Player Performance Visualization")
num_cols = get_numeric_columns(filtered)
plot_metrics = [c for c in num_cols if c not in {'Age', 'Market value'}]

x_default = 'Goals per 90' if 'Goals per 90' in plot_metrics else (plot_metrics[0] if plot_metrics else None)
y_default = (calc_col_name if calc_col_name and calc_col_name in plot_metrics
             else ('Assists per 90' if 'Assists per 90' in plot_metrics
                   else (plot_metrics[1] if len(plot_metrics) > 1 else x_default)))

if x_default is None or y_default is None:
    st.warning("No numerical metrics available for plotting.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", plot_metrics, index=plot_metrics.index(x_default))
    with c2:
        y_axis = st.selectbox("Y-axis", plot_metrics, index=plot_metrics.index(y_default))
    with c3:
        color_by = st.selectbox(
            "Color by",
            options=[o for o in ['Season label', 'Main Position', 'Team', 'League', 'Foot', 'None'] if o == 'None' or o in filtered.columns],
            index=0 if 'Season label' in filtered.columns else 0
        )

    size_by = st.selectbox(
        "Size by",
        options=[o for o in ['None', 'Minutes played', 'Market value (M€)', 'Age', 'Matches played'] if o == 'None' or o in filtered.columns],
        index=1 if 'Minutes played' in filtered.columns else 0
    )

    rank_axis = st.radio("Sort Top-N players by", ["X-axis", "Y-axis"], index=1, horizontal=True)
    sort_metric = y_axis if rank_axis == "Y-axis" else x_axis
    plot_limit = st.slider(f"Number of players to plot (Top-N by {sort_metric})",
                           1, min(30, len(filtered)), min(15, len(filtered)))
    plot_df = filtered.sort_values(by=sort_metric, ascending=False).head(plot_limit).copy()

    if remove_outliers:
        for ax in [x_axis, y_axis]:
            std = plot_df[ax].std(ddof=0)
            if std > 0:
                z = (plot_df[ax] - plot_df[ax].mean()) / std
                plot_df = plot_df[np.abs(z) <= 3]

    plot_df[x_axis] = plot_df[x_axis].round(2)
    plot_df[y_axis] = plot_df[y_axis].round(2)
    show_labels = st.checkbox("Show player labels on chart", value=False)

    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        hover_name="Player" if 'Player' in plot_df.columns else None,
        color=None if color_by == 'None' else color_by,
        size=None if size_by == 'None' else size_by,
        text=plot_df['Player'] if show_labels and 'Player' in plot_df.columns else None,
        title=f"{y_axis} vs. {x_axis} by Player",
        template="plotly_white",
        height=620,
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        hovertemplate="Player: %{hovertext}<br>" + x_axis + ": %{x:.2f}<br>" + y_axis + ": %{y:.2f}<extra></extra>",
        textposition="top center",
        textfont=dict(size=12),
        cliponaxis=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Player comparison + Radar (z-score across filtered set)
# =========================
st.subheader("Compare Selected Players")
compare_players = st.multiselect(
    "Players to compare (max 5 recommended)",
    options=sorted(filtered['Player'].dropna().unique().tolist()),
    default=[]
)

if compare_players:
    comp_df = filtered[filtered['Player'].isin(compare_players)].set_index('Player')
    comp_metric_choices = get_numeric_columns(filtered).copy()

    default_comp = [m for m in profile_metrics_in_use if m in comp_metric_choices]
    if calc_col_name and calc_col_name in comp_metric_choices:
        default_comp = [calc_col_name] + default_comp
    if not default_comp:
        fallback_defaults = ['Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90', 'Successful defensive actions per 90', 'Duels won, %']
        if calc_col_name and calc_col_name in comp_metric_choices:
            fallback_defaults = [calc_col_name] + fallback_defaults
        default_comp = [m for m in fallback_defaults if m in comp_metric_choices] or comp_metric_choices[:6]

    comp_metrics = st.multiselect("Metrics for comparison table & radar", options=comp_metric_choices, default=default_comp)

    if comp_metrics:
        # Optional reordering UI (if streamlit_sortable available)
        ordered_comp_metrics = comp_metrics

        comp_df = comp_df.round(2)
        st.dataframe(
            comp_df[ordered_comp_metrics].transpose().round(2).style.format("{:.2f}").highlight_max(axis=1, color='#C8E6C9'),
            use_container_width=True
        )

        mm_base = filtered.set_index('Player')
        means = {m: mm_base[m].mean() for m in ordered_comp_metrics}
        stds  = {m: mm_base[m].std(ddof=0) for m in ordered_comp_metrics}

        def zscore(val, mean, std):
            if pd.isna(val) or pd.isna(mean) or pd.isna(std) or std == 0:
                return 0.0
            return float((val - mean) / std)

        theta = ordered_comp_metrics
        fig_radar = go.Figure()
        for player in compare_players:
            row = comp_df.loc[player, ordered_comp_metrics]
            r = [zscore(float(row[m]), means[m], stds[m]) if pd.notna(row[m]) else 0.0 for m in ordered_comp_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill='toself',
                name=player,
                text=[f"{player}: z={val:.2f}" for val in r] + [f"{player}: z={r[0]:.2f}"],
                hoverinfo="text"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            showlegend=True,
            template='plotly_white',
            height=640
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        csv_buf2 = StringIO()
        comp_df[ordered_comp_metrics].to_csv(csv_buf2)
        st.download_button("⬇️ Download comparison (CSV)", data=csv_buf2.getvalue(),
                           file_name="player_comparison.csv", mime="text/csv")
    else:
        st.info("Select metrics to compare players.")
else:
    st.info("Select players above to compare their stats and see a radar chart.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Plotly | Season-aware ✨")
