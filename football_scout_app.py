# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from io import BytesIO, StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(layout="wide", page_title="Advanced Football Scouting App", page_icon="⚽")
st.title("⚽ Advanced Football Player Scouting App — Season-aware Edition")
st.markdown(
    "Upload your football data CSV to analyze player metrics. "
    "Includes season normalization, robust metric aliasing, built-in & custom profiles with weights, "
    "direction-aware z-scores, sparse-metric skipping, Top-N tables, scatter & radar, and CSV downloads."
)

# =========================
# Constants
# =========================
PCT_SUFFIX = ", %"
ALL_TOKEN = "ALL"

REQUIRED_COLS = ["Player", "Team", "Main Position", "Age", "League"]

NON_FEATURE_COLUMNS = {
    "Column1",
    "Player",
    "Team",
    "Team within selected timeframe",
    "Position",
    "Birth country",
    "Passport country",
    "Foot",
    "On loan",
    "Contract expires",
    "League",
    "Main Position",
    "Age",
    "Season label",
    "Season start",
    "Season end",
    "Season group",
}

LOWER_IS_BETTER = {
    "Conceded goals per 90",
    "Fouls per 90",
    "Turnovers per 90",
    "Miscontrols per 90",
    "Yellow cards per 90",
    "Red cards per 90",
}

SEASON_RX = re.compile(r"(\d{4})(?:\s*-\s*(\d{2,4}))?$")


# =========================
# Session state (persist score across reruns)
# =========================
if "active_profile" not in st.session_state:
    st.session_state["active_profile"] = None
# active_profile structure:
# {
#   "calc_col": str,
#   "metrics": list[str],
#   "weights_pct": list[int]
# }


# =========================
# Profiles
# =========================
PROFILES: Dict[str, List[str]] = {
    # 🧤 GOALKEEPERS
    "Classic Goalkeeper": [
        "Save rate, %",
        "Prevented goals per 90",
        "Conceded goals per 90",
        "Shots against per 90",
        "Aerial duels won, %",
        "Exits per 90",
        "Aerial duels per 90",
        "Accurate long passes, %",
        "Accurate passes, %",
        "Average pass length, m",
    ],
    "Sweeper Keeper": [
        "Exits per 90",
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Shots against per 90",
        "Prevented goals per 90",
        "Save rate, %",
        "Progressive passes per 90",
        "Forward passes per 90",
        "Accurate long passes, %",
        "Passes to final third per 90",
    ],
    "Build-Up Keeper": [
        "Accurate passes, %",
        "Accurate long passes, %",
        "Progressive passes per 90",
        "Forward passes per 90",
        "Passes to final third per 90",
        "Average pass length, m",
        "Passes per 90",
        "Save rate, %",
        "Prevented goals per 90",
        "Exits per 90",
    ],

    # 🛡️ CENTRE-BACKS
    "Ball-Playing CB": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Forward passes per 90",
        "Accurate passes, %",
        "Accurate long passes, %",
        "Passes per 90",
        "Average pass length, m",
        "Interceptions per 90",
        "Defensive duels won, %",
        "Aerial duels won, %",
    ],
    "Combative CB / Stopper": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Shots blocked per 90",
        "Interceptions per 90",
        "Fouls per 90",
        "Successful defensive actions per 90",
        "Passes per 90",
        "Accurate passes, %",
    ],
    "Libero / Middle Pin CB": [
        "Progressive passes per 90",
        "Accurate long passes, %",
        "Passes to final third per 90",
        "Accurate passes, %",
        "Deep completions per 90",
        "Smart passes per 90",
        "xA per 90",
        "Interceptions per 90",
        "Aerial duels won, %",
        "Defensive duels won, %",
    ],
    "Wide CB (in 3)": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Progressive runs per 90",
        "Interceptions per 90",
        "Aerial duels won, %",
        "Successful defensive actions per 90",
        "Crosses per 90",
        "Accurate crosses, %",
        "Touches in box per 90",
        "Progressive passes per 90",
    ],

    # ⚙️ MIDFIELDERS
    "Defensive Midfielder #6": [
        "Interceptions per 90",
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Successful defensive actions per 90",
        "Accurate passes, %",
        "Forward passes per 90",
        "Passes to final third per 90",
        "Progressive passes per 90",
        "Average pass length, m",
        "Aerial duels won, %",
    ],
    "Attacking Midfielder #8": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Progressive runs per 90",
        "xA per 90",
        "Shots per 90",
        "Touches in box per 90",
        "Interceptions per 90",
        "Key passes per 90",
        "Deep completions per 90",
        "Successful attacking actions per 90",
    ],
    "Deep-Lying Playmaker": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Received passes per 90",
        "Accurate long passes, %",
        "Forward passes per 90",
        "Passes per 90",
        "Passes to final third per 90",
        "Interceptions per 90",
        "Defensive duels per 90",
        "Aerial duels won, %",
    ],
    "Box-to-Box Midfielder": [
        "Progressive runs per 90",
        "xG per 90",
        "Shots per 90",
        "Interceptions per 90",
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Touches in box per 90",
        "Successful attacking actions per 90",
        "Forward passes per 90",
        "Passes to final third per 90",
    ],

    # 🌊 WIDE / ATTACKING ROLES
    "Full-Back": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Interceptions per 90",
        "Crosses per 90",
        "Accurate crosses, %",
        "Progressive runs per 90",
        "Progressive passes per 90",
        "Forward passes per 90",
        "Successful defensive actions per 90",
        "Deep completions per 90",
    ],
    "Wing-Back": [
        "Progressive runs per 90",
        "Crosses per 90",
        "Accurate crosses, %",
        "Shot assists per 90",
        "Progressive passes per 90",
        "Interceptions per 90",
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Touches in box per 90",
        "Successful attacking actions per 90",
    ],
    "Inverted Full-Back": [
        "Progressive passes per 90",
        "Progressive runs per 90",
        "Forward passes per 90",
        "Accurate passes, %",
        "Accurate short / medium passes, %",
        "Smart passes per 90",
        "Defensive duels won, %",
        "Interceptions per 90",
        "Successful defensive actions per 90",
        "Aerial duels won, %",
    ],
    "Classic Winger": [
        "Dribbles per 90",
        "Successful dribbles, %",
        "Progressive runs per 90",
        "Crosses per 90",
        "Accurate crosses, %",
        "Shot assists per 90",
        "Touches in box per 90",
        "Shots per 90",
        "xA per 90",
        "Successful attacking actions per 90",
    ],
    "Inverted Winger": [
        "Shots per 90",
        "xG per 90",
        "xA per 90",
        "Progressive runs per 90",
        "Shot assists per 90",
        "Touches in box per 90",
        "Dribbles per 90",
        "Successful dribbles, %",
        "Deep completions per 90",
        "Key passes per 90",
    ],
    "Playmaker #10": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Deep completions per 90",
        "Key passes per 90",
        "xA per 90",
        "Shot assists per 90",
        "Shots per 90",
        "xG per 90",
        "Progressive runs per 90",
        "Successful attacking actions per 90",
    ],

    # ⚡ FORWARDS
    "Target Man #9": [
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Received long passes per 90",
        "Passes to final third per 90",
        "Fouls suffered per 90",
        "xG per 90",
        "Shots per 90",
        "Non-penalty goals per 90",
        "Touches in box per 90",
        "Received passes per 90",
    ],
    "Poacher": [
        "Non-penalty goals per 90",
        "xG per 90",
        "Shots per 90",
        "Goal conversion, %",
        "Touches in box per 90",
        "Received passes per 90",
        "xA per 90",
        "Progressive runs per 90",
        "Key passes per 90",
        "Successful attacking actions per 90",
    ],
    "Pressing Forward": [
        "Defensive duels per 90",
        "Defensive duels won, %",
        "Interceptions per 90",
        "Successful defensive actions per 90",
        "Progressive runs per 90",
        "Shots per 90",
        "xG per 90",
        "xA per 90",
        "Touches in box per 90",
        "Successful attacking actions per 90",
    ],
    "Creative Forward / False 9": [
        "Progressive passes per 90",
        "Accurate progressive passes, %",
        "Deep completions per 90",
        "Key passes per 90",
        "xA per 90",
        "Progressive runs per 90",
        "Received passes per 90",
        "Shots per 90",
        "xG per 90",
        "Touches in box per 90",
    ],
    "Wide Forward / Inside 9": [
        "Progressive runs per 90",
        "Dribbles per 90",
        "Successful dribbles, %",
        "Shots per 90",
        "xG per 90",
        "xA per 90",
        "Touches in box per 90",
        "Deep completions per 90",
        "Key passes per 90",
        "Successful attacking actions per 90",
    ],
}

DEFAULT_WEIGHTS: Dict[str, Dict[str, int]] = {
    "Classic Goalkeeper": {
        "Save rate, %": 25,
        "Prevented goals per 90": 20,
        "Conceded goals per 90": 5,
        "Shots against per 90": 10,
        "Aerial duels won, %": 10,
        "Exits per 90": 5,
        "Aerial duels per 90": 5,
        "Accurate long passes, %": 10,
        "Accurate passes, %": 5,
        "Average pass length, m": 5,
    },
    "Sweeper Keeper": {
        "Exits per 90": 20,
        "Aerial duels per 90": 10,
        "Aerial duels won, %": 10,
        "Shots against per 90": 5,
        "Prevented goals per 90": 10,
        "Save rate, %": 10,
        "Progressive passes per 90": 10,
        "Forward passes per 90": 10,
        "Accurate long passes, %": 10,
        "Passes to final third per 90": 5,
    },
    "Build-Up Keeper": {
        "Accurate passes, %": 20,
        "Accurate long passes, %": 15,
        "Progressive passes per 90": 15,
        "Forward passes per 90": 10,
        "Passes to final third per 90": 10,
        "Average pass length, m": 10,
        "Passes per 90": 5,
        "Save rate, %": 5,
        "Prevented goals per 90": 5,
        "Exits per 90": 5,
    },
    "Ball-Playing CB": {
        "Progressive passes per 90": 20,
        "Accurate progressive passes, %": 15,
        "Forward passes per 90": 10,
        "Accurate passes, %": 10,
        "Accurate long passes, %": 10,
        "Passes per 90": 10,
        "Average pass length, m": 5,
        "Interceptions per 90": 10,
        "Defensive duels won, %": 5,
        "Aerial duels won, %": 5,
    },
    "Combative CB / Stopper": {
        "Defensive duels per 90": 20,
        "Defensive duels won, %": 20,
        "Aerial duels per 90": 15,
        "Aerial duels won, %": 15,
        "Shots blocked per 90": 10,
        "Interceptions per 90": 5,
        "Fouls per 90": 5,
        "Successful defensive actions per 90": 5,
        "Passes per 90": 0,
        "Accurate passes, %": 0,
    },
    "Libero / Middle Pin CB": {
        "Progressive passes per 90": 20,
        "Accurate long passes, %": 15,
        "Passes to final third per 90": 10,
        "Accurate passes, %": 10,
        "Deep completions per 90": 10,
        "Smart passes per 90": 10,
        "xA per 90": 5,
        "Interceptions per 90": 10,
        "Aerial duels won, %": 5,
        "Defensive duels won, %": 5,
    },
    "Wide CB (in 3)": {
        "Defensive duels per 90": 15,
        "Defensive duels won, %": 10,
        "Progressive runs per 90": 15,
        "Interceptions per 90": 10,
        "Aerial duels won, %": 5,
        "Successful defensive actions per 90": 10,
        "Crosses per 90": 10,
        "Accurate crosses, %": 10,
        "Touches in box per 90": 5,
        "Progressive passes per 90": 10,
    },
    "Defensive Midfielder #6": {
        "Interceptions per 90": 20,
        "Defensive duels per 90": 15,
        "Defensive duels won, %": 10,
        "Successful defensive actions per 90": 10,
        "Accurate passes, %": 10,
        "Forward passes per 90": 10,
        "Passes to final third per 90": 10,
        "Progressive passes per 90": 10,
        "Average pass length, m": 5,
        "Aerial duels won, %": 0,
    },
    "Attacking Midfielder #8": {
        "Progressive passes per 90": 15,
        "Accurate progressive passes, %": 10,
        "Progressive runs per 90": 15,
        "xA per 90": 15,
        "Shots per 90": 15,
        "Touches in box per 90": 10,
        "Interceptions per 90": 5,
        "Key passes per 90": 5,
        "Deep completions per 90": 5,
        "Successful attacking actions per 90": 5,
    },
    "Deep-Lying Playmaker": {
        "Progressive passes per 90": 20,
        "Accurate progressive passes, %": 15,
        "Received passes per 90": 10,
        "Accurate long passes, %": 10,
        "Forward passes per 90": 10,
        "Passes per 90": 10,
        "Passes to final third per 90": 10,
        "Interceptions per 90": 5,
        "Defensive duels per 90": 5,
        "Aerial duels won, %": 5,
    },
    "Box-to-Box Midfielder": {
        "Progressive runs per 90": 20,
        "xG per 90": 15,
        "Shots per 90": 15,
        "Interceptions per 90": 10,
        "Defensive duels per 90": 10,
        "Defensive duels won, %": 5,
        "Touches in box per 90": 10,
        "Successful attacking actions per 90": 10,
        "Forward passes per 90": 5,
        "Passes to final third per 90": 0,
    },
    "Full-Back": {
        "Defensive duels per 90": 15,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 10,
        "Crosses per 90": 10,
        "Accurate crosses, %": 10,
        "Progressive runs per 90": 15,
        "Progressive passes per 90": 10,
        "Forward passes per 90": 10,
        "Successful defensive actions per 90": 5,
        "Deep completions per 90": 5,
    },
    "Wing-Back": {
        "Progressive runs per 90": 20,
        "Crosses per 90": 15,
        "Accurate crosses, %": 10,
        "Shot assists per 90": 15,
        "Progressive passes per 90": 10,
        "Interceptions per 90": 5,
        "Defensive duels per 90": 10,
        "Defensive duels won, %": 5,
        "Touches in box per 90": 5,
        "Successful attacking actions per 90": 5,
    },
    "Inverted Full-Back": {
        "Progressive passes per 90": 15,
        "Progressive runs per 90": 10,
        "Forward passes per 90": 10,
        "Accurate passes, %": 10,
        "Accurate short / medium passes, %": 5,
        "Smart passes per 90": 5,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 10,
        "Successful defensive actions per 90": 10,
        "Aerial duels won, %": 5,
    },
    "Classic Winger": {
        "Dribbles per 90": 20,
        "Successful dribbles, %": 15,
        "Progressive runs per 90": 15,
        "Crosses per 90": 10,
        "Accurate crosses, %": 10,
        "Shot assists per 90": 10,
        "Touches in box per 90": 5,
        "Shots per 90": 5,
        "xA per 90": 5,
        "Successful attacking actions per 90": 5,
    },
    "Inverted Winger": {
        "Shots per 90": 20,
        "xG per 90": 15,
        "xA per 90": 10,
        "Progressive runs per 90": 15,
        "Shot assists per 90": 10,
        "Touches in box per 90": 5,
        "Dribbles per 90": 10,
        "Successful dribbles, %": 5,
        "Deep completions per 90": 5,
        "Key passes per 90": 5,
    },
    "Playmaker #10": {
        "Progressive passes per 90": 20,
        "Accurate progressive passes, %": 15,
        "Deep completions per 90": 15,
        "Key passes per 90": 10,
        "xA per 90": 10,
        "Shot assists per 90": 10,
        "Shots per 90": 5,
        "xG per 90": 5,
        "Progressive runs per 90": 5,
        "Successful attacking actions per 90": 5,
    },
    "Target Man #9": {
        "Aerial duels per 90": 20,
        "Aerial duels won, %": 15,
        "Received long passes per 90": 10,
        "Passes to final third per 90": 10,
        "Fouls suffered per 90": 10,
        "xG per 90": 10,
        "Shots per 90": 5,
        "Non-penalty goals per 90": 5,
        "Touches in box per 90": 5,
        "Received passes per 90": 10,
    },
    "Poacher": {
        "Non-penalty goals per 90": 25,
        "xG per 90": 20,
        "Shots per 90": 10,
        "Goal conversion, %": 10,
        "Touches in box per 90": 10,
        "Received passes per 90": 5,
        "xA per 90": 5,
        "Progressive runs per 90": 5,
        "Key passes per 90": 5,
        "Successful attacking actions per 90": 5,
    },
    "Pressing Forward": {
        "Defensive duels per 90": 15,
        "Defensive duels won, %": 10,
        "Interceptions per 90": 15,
        "Successful defensive actions per 90": 10,
        "Progressive runs per 90": 10,
        "Shots per 90": 10,
        "xG per 90": 10,
        "xA per 90": 5,
        "Touches in box per 90": 10,
        "Successful attacking actions per 90": 5,
    },
    "Creative Forward / False 9": {
        "Progressive passes per 90": 15,
        "Accurate progressive passes, %": 10,
        "Deep completions per 90": 10,
        "Key passes per 90": 15,
        "xA per 90": 10,
        "Progressive runs per 90": 10,
        "Received passes per 90": 5,
        "Shots per 90": 10,
        "xG per 90": 10,
        "Touches in box per 90": 5,
    },
    "Wide Forward / Inside 9": {
        "Progressive runs per 90": 15,
        "Dribbles per 90": 10,
        "Successful dribbles, %": 10,
        "Shots per 90": 15,
        "xG per 90": 15,
        "xA per 90": 10,
        "Touches in box per 90": 10,
        "Deep completions per 90": 5,
        "Key passes per 90": 5,
        "Successful attacking actions per 90": 5,
    },
}


# =========================
# Utilities
# =========================
def norm_key(s: str) -> str:
    return re.sub(r"[\s,%–\-]+", "", str(s)).lower()


def safe_widget_key(*parts: str) -> str:
    raw = "::".join(parts)
    return re.sub(r"[^a-zA-Z0-9_:\-]+", "_", raw)


def multiselect_all(label: str, options: List[str], default_all: bool = True, help: str | None = None, key: str | None = None):
    opts = [ALL_TOKEN] + options
    default = [ALL_TOKEN] if default_all else []
    picked = st.sidebar.multiselect(label, opts, default=default, help=help, key=key)
    use_all = (not picked) or (ALL_TOKEN in picked)
    return (options if use_all else [o for o in picked if o != ALL_TOKEN]), use_all


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes, filename: str) -> pd.DataFrame:
    # cache key includes bytes + filename
    try:
        return pd.read_csv(StringIO(file_bytes.decode("utf-8")))
    except UnicodeDecodeError:
        return pd.read_csv(StringIO(file_bytes.decode("latin-1")))
    except Exception:
        bio = BytesIO(file_bytes)
        try:
            return pd.read_csv(bio)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, encoding="latin-1")


def parse_market_value(series: pd.Series) -> pd.Series:
    """Parse '€12.5m', '€800k', '12,000,000' into float (EUR M)."""
    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        mx = s.max(skipna=True)
        if pd.notna(mx) and mx > 1e6:
            return s / 1e6
        return s

    def to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace("€", "").replace(",", "").lower()
        mult = 1.0
        if s.endswith("m"):
            mult = 1_000_000.0
            s = s[:-1]
        elif s.endswith("k"):
            mult = 1_000.0
            s = s[:-1]
        try:
            val = float(s) * mult
        except ValueError:
            return np.nan
        return val / 1e6

    return series.apply(to_float)


@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns + create season fields."""
    df = df_raw.copy()

    if "Column1" in df.columns and df["Column1"].nunique(dropna=False) == len(df):
        df = df.drop(columns=["Column1"])

    if "Market value" in df.columns:
        df["Market value (M€)"] = parse_market_value(df["Market value"])

    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or col == "Market value":
            continue
        if col.endswith(PCT_SUFFIX):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False), errors="coerce")
        else:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def extract_season_label(league_value: object) -> str | None:
        if not isinstance(league_value, str):
            return None
        parts = league_value.strip().split()
        if not parts:
            return None
        tail = parts[-1]
        return tail if SEASON_RX.fullmatch(tail) else None

    def season_start_end(season_label: str) -> Tuple[int | None, int | None]:
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

    df["Season label"] = df["League"].apply(extract_season_label)
    starts_ends = df["Season label"].apply(lambda s: season_start_end(s) if isinstance(s, str) else (None, None))
    df["Season start"] = starts_ends.apply(lambda t: t[0])
    df["Season end"] = starts_ends.apply(lambda t: t[1])
    df["Season group"] = pd.Series(df["Season start"]).fillna(df["Season end"]).astype("Int64")

    return df


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric = df.select_dtypes(include="number").columns
    numeric = [c for c in numeric if c not in NON_FEATURE_COLUMNS and c != "Market value"]
    return numeric


def resolve_metrics_aliases(requested: List[str], columns: List[str]) -> Tuple[List[str], List[str]]:
    col_norm_map = {norm_key(c): c for c in columns}
    resolved, missing = [], []
    for name in requested:
        if name in columns:
            resolved.append(name)
            continue
        if "%" in name and ", %" not in name:
            alt = name.replace(" %", ", %").replace("%", ", %")
            if alt in columns:
                resolved.append(alt)
                continue
        key = norm_key(name)
        if key in col_norm_map:
            resolved.append(col_norm_map[key])
            continue
        missing.append(name)
    return resolved, missing


def normalize_weights(pcts: np.ndarray) -> np.ndarray:
    total = float(np.nansum(pcts))
    if total <= 0 or len(pcts) == 0:
        return np.ones_like(pcts, dtype=float) / max(1, len(pcts))
    return (pcts / total).astype(float)


def sparse_mask(X: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    zeros_or_nan = X.isna() | (X == 0)
    return zeros_or_nan.mean(axis=0) >= threshold


def make_profile_score_vectorized(
    df: pd.DataFrame,
    metrics: List[str],
    weights: np.ndarray,
    new_col: str,
    sparse_threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Vectorized weighted directional z-score with sparse-metric skipping."""
    present = [m for m in metrics if m in df.columns]
    if not present:
        out = df.copy()
        out[new_col] = 0.0
        return out, [], metrics

    X = df[present].apply(pd.to_numeric, errors="coerce")
    sparse = sparse_mask(X, threshold=sparse_threshold)
    usable_metrics = [m for m in present if not bool(sparse.get(m, False))]
    skipped_sparse = [m for m in present if m not in usable_metrics]

    if not usable_metrics:
        out = df.copy()
        out[new_col] = 0.0
        return out, [], skipped_sparse

    w_map = {m: float(w) for m, w in zip(metrics, weights)}
    used_w = np.array([w_map.get(m, 0.0) for m in usable_metrics], dtype=float)
    used_w = normalize_weights(used_w)

    Xu = X[usable_metrics]
    means = Xu.mean(axis=0)
    stds = Xu.std(ddof=0, axis=0).replace(0, np.nan)

    Z = (Xu - means) / stds
    Z = Z.fillna(0.0)

    flip_cols = [c for c in usable_metrics if c in LOWER_IS_BETTER]
    if flip_cols:
        Z[flip_cols] = -Z[flip_cols]

    scores = (Z.to_numpy(dtype=float) * used_w.reshape(1, -1)).sum(axis=1)
    out = df.copy()
    out[new_col] = np.round(scores, 2)

    return out, usable_metrics, skipped_sparse


def defaults_for_resolved(profile_name: str, resolved_metric_names: List[str]) -> List[int]:
    dm = DEFAULT_WEIGHTS.get(profile_name, {})
    dm_norm = {norm_key(k): int(v) for k, v in dm.items()}
    if not resolved_metric_names:
        return []
    equal = max(1, int(100 / len(resolved_metric_names)))
    return [int(dm_norm.get(norm_key(m), equal)) for m in resolved_metric_names]


# =========================
# Upload
# =========================
st.sidebar.header("Upload")
uploaded = st.sidebar.file_uploader("Upload your Football Data CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload your football data CSV (e.g., Wyscout export).")
    st.stop()

file_bytes = uploaded.getvalue()
try:
    df_raw = load_csv(file_bytes, uploaded.name)
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty.")
    st.stop()
except pd.errors.ParserError:
    st.error("Could not parse the CSV file. Please ensure it is a valid CSV.")
    st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing:
    st.error(f"Missing critical column(s): {', '.join(missing)}.")
    st.stop()

df_all = preprocess(df_raw)

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

# Season
season_groups = sorted([x for x in df_all["Season group"].dropna().unique().tolist()])
selected_groups = st.sidebar.multiselect("Season group (start year)", season_groups, default=season_groups)

use_exact_label = st.sidebar.checkbox("Also filter by exact Season label", value=False)
selected_labels = None
if use_exact_label:
    season_labels = sorted(df_all["Season label"].dropna().unique().tolist())
    selected_labels = st.sidebar.multiselect("Season label(s)", season_labels, default=season_labels)

season_mask = df_all["Season group"].isin(selected_groups) if selected_groups else pd.Series(True, index=df_all.index)
if use_exact_label and selected_labels:
    season_mask = season_mask & df_all["Season label"].isin(selected_labels)

df_season = df_all.loc[season_mask]

# League
leagues = sorted(df_season["League"].dropna().unique().tolist())
selected_leagues, _ = multiselect_all("League(s)", leagues, default_all=True, help="Choose specific leagues or ALL")

df_league = df_season.loc[df_season["League"].isin(selected_leagues)]
if df_league.empty:
    st.warning("No players found for selected seasons/leagues.")
    st.stop()

# Team / Position
teams = sorted(df_league["Team"].dropna().unique().tolist())
positions = sorted(df_league["Main Position"].dropna().unique().tolist())
selected_teams, _ = multiselect_all("Team(s)", teams, default_all=True)
selected_positions, _ = multiselect_all("Main Position(s)", positions, default_all=True)

# Age
age_series = pd.to_numeric(df_league["Age"], errors="coerce")
age_min = int(age_series.min(skipna=True) or 0)
age_max = int(age_series.max(skipna=True) or 0)
if age_max < age_min:
    age_min, age_max = 0, 0
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max)) if age_max >= age_min else (0, 0)

# Minutes
min_minutes = 0
if "Minutes played" in df_league.columns:
    mm = pd.to_numeric(df_league["Minutes played"], errors="coerce")
    min_minutes_max = int(mm.max(skipna=True) or 0)
    min_minutes = st.sidebar.slider(
        "Minimum minutes this season",
        0,
        max(0, min_minutes_max),
        min(500, max(0, min_minutes_max)),
    )

remove_outliers = st.sidebar.checkbox("Remove outliers (|Z| > 3) — for plots only", value=False)

mask = (
    df_league["Team"].isin(selected_teams)
    & df_league["Main Position"].isin(selected_positions)
    & pd.to_numeric(df_league["Age"], errors="coerce").between(age_range[0], age_range[1])
)
if "Minutes played" in df_league.columns:
    mask = mask & (pd.to_numeric(df_league["Minutes played"], errors="coerce") >= min_minutes)

filtered_base = df_league.loc[mask]
st.sidebar.markdown(f"**Players matching filters: {len(filtered_base)}**")
if filtered_base.empty:
    st.warning("No players match the selected filters.")
    st.stop()

numeric_cols_base = get_numeric_columns(filtered_base)

# =========================
# Profile builder (FORM)
# =========================
st.sidebar.header("Player profiles (z-score)")

# Clear active profile button (outside the form)
if st.sidebar.button("Clear active profile"):
    st.session_state["active_profile"] = None
    st.rerun()

with st.sidebar.form("profile_form"):
    st.caption("Scores are weighted sums of direction-aware z-scores across the currently filtered players.")
    mode = st.radio("Profile mode", ["Built-in", "Custom"], index=0, horizontal=True)

    if mode == "Built-in":
        profile_name = st.selectbox("Choose profile", list(PROFILES.keys()))
        requested_metrics = PROFILES[profile_name]
        resolved_metrics, missing_names = resolve_metrics_aliases(requested_metrics, filtered_base.columns.tolist())

        # persist per-profile slider defaults in session_state
        preset_key = f"preset::{profile_name}"
        if preset_key not in st.session_state:
            st.session_state[preset_key] = {
                "metrics": resolved_metrics[:],
                "weights": defaults_for_resolved(profile_name, resolved_metrics),
            }
        else:
            state = st.session_state[preset_key]
            if state.get("metrics") != resolved_metrics:
                old_map = {m: int(w) for m, w in zip(state.get("metrics", []), state.get("weights", []))}
                new_defaults = defaults_for_resolved(profile_name, resolved_metrics)
                new_weights = [int(old_map.get(m, d)) for m, d in zip(resolved_metrics, new_defaults)]
                st.session_state[preset_key] = {"metrics": resolved_metrics[:], "weights": new_weights}

        state = st.session_state[preset_key]

        if st.form_submit_button("Reset weights to defaults"):
            state["weights"] = defaults_for_resolved(profile_name, resolved_metrics)

        weights_pct: List[int] = []
        for m, dflt in zip(state.get("metrics", []), state.get("weights", [])):
            w = st.slider(
                f"Weight %: {m}",
                0,
                100,
                int(dflt),
                1,
                key=safe_widget_key("w", profile_name, m),
            )
            weights_pct.append(int(w))

        submitted = st.form_submit_button("Apply profile")

        if submitted:
            if missing_names:
                st.info("Skipped missing metrics: " + ", ".join(missing_names))
            if not resolved_metrics:
                st.warning("No valid metrics for this profile in the current dataset.")
            else:
                # Persist active profile config so score survives all reruns
                st.session_state["active_profile"] = {
                    "calc_col": f"Score: {profile_name}",
                    "metrics": resolved_metrics[:],
                    "weights_pct": [int(x) for x in weights_pct],
                }

    else:
        st.subheader("Custom Profile")
        custom_name = st.text_input("Profile name", value="Custom Profile").strip() or "Custom Profile"
        custom_metrics = st.multiselect("Pick metrics to include", options=numeric_cols_base, default=numeric_cols_base[:5])

        weights_pct = []
        if custom_metrics:
            default_pct = max(1, int(100 / len(custom_metrics)))
            for m in custom_metrics:
                w = st.slider(
                    f"Weight %: {m}",
                    0,
                    100,
                    default_pct,
                    1,
                    key=safe_widget_key("w_custom", custom_name, m),
                )
                weights_pct.append(int(w))

        submitted = st.form_submit_button("Apply profile")

        if submitted:
            if not custom_metrics:
                st.info("Select at least one metric to build a custom profile.")
            else:
                st.session_state["active_profile"] = {
                    "calc_col": f"Score: {custom_name}",
                    "metrics": custom_metrics[:],
                    "weights_pct": [int(x) for x in weights_pct],
                }

# =========================
# Apply active profile on EVERY rerun (fixes disappearing score)
# =========================
active = st.session_state.get("active_profile")
calc_col_name: str | None = None
profile_metrics_in_use: List[str] = []

if active and active.get("metrics"):
    calc_col_name = str(active.get("calc_col", "Score"))
    profile_metrics_in_use = list(active.get("metrics", []))
    weights_pct_arr = np.array(active.get("weights_pct", []), dtype=float)

    # if weights mismatch metrics length (e.g., metrics changed), fall back to equal weights
    if len(weights_pct_arr) != len(profile_metrics_in_use) or len(profile_metrics_in_use) == 0:
        weights = np.ones(len(profile_metrics_in_use), dtype=float) / max(1, len(profile_metrics_in_use))
    else:
        weights = normalize_weights(weights_pct_arr)

    filtered, usable_metrics, skipped_sparse = make_profile_score_vectorized(
        filtered_base,
        profile_metrics_in_use,
        weights,
        calc_col_name,
    )

    if skipped_sparse:
        st.caption("⚠️ Skipped sparse metrics (≥95% zeros/NaNs): " + ", ".join(skipped_sparse))
else:
    filtered = filtered_base

numeric_cols = get_numeric_columns(filtered)

# =========================
# Top-N table
# =========================
st.subheader("Filtered Player Data")

ID_COLS = [
    "Season label",
    "Player",
    "Team",
    "League",
    "Main Position",
    "Age",
    "Market value (M€)",
    "Goals",
    "Assists",
    "xG",
    "xA",
    "Minutes played",
]

exclude_cols = {"Market value"} if "Market value (M€)" in filtered.columns else set()
display_options = [c for c in filtered.columns if c not in exclude_cols]

default_cols = [c for c in ID_COLS if c in filtered.columns]
if calc_col_name and calc_col_name in filtered.columns:
    default_cols = default_cols + [calc_col_name]

selected_display_cols = st.multiselect("Columns to display", options=display_options, default=default_cols)

rank_candidates = [calc_col_name, "Assists per 90", "Goals per 90", "xA per 90", "xG per 90", "xA", "xG", "Minutes played"]
rank_candidates = [c for c in rank_candidates if c and c in numeric_cols]
default_rank = rank_candidates[0] if rank_candidates else (numeric_cols[0] if numeric_cols else None)

if not selected_display_cols:
    st.info("Please select at least one column to display.")
elif default_rank is None:
    st.warning("No numerical columns available to sort Top-N.")
else:
    rank_by = st.selectbox(
        "Sort Top-N rows by",
        options=numeric_cols,
        index=numeric_cols.index(default_rank) if default_rank in numeric_cols else 0,
    )
    row_limit = st.slider(f"Number of rows to show (Top-N by {rank_by})", 1, 30, 15)

    table_df = filtered.sort_values(by=rank_by, ascending=False).head(row_limit)[selected_display_cols].copy()
    for c in table_df.select_dtypes(include="number").columns:
        table_df[c] = pd.to_numeric(table_df[c], errors="coerce").round(2)

    st.dataframe(table_df.reset_index(drop=True), use_container_width=True)

# CSV download
csv_buf = StringIO()
dl_cols = selected_display_cols if selected_display_cols else default_cols
filtered[dl_cols].to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Download filtered data (CSV)",
    data=csv_buf.getvalue(),
    file_name="filtered_players.csv",
    mime="text/csv",
)

# =========================
# Scatter plot
# =========================
st.subheader("Player Performance Visualization")

plot_metrics = [c for c in numeric_cols if c not in {"Age", "Market value"}]
if not plot_metrics:
    st.warning("No numerical metrics available for plotting.")
else:
    x_default = "Goals per 90" if "Goals per 90" in plot_metrics else plot_metrics[0]
    if calc_col_name and calc_col_name in plot_metrics:
        y_default = calc_col_name
    elif "Assists per 90" in plot_metrics:
        y_default = "Assists per 90"
    else:
        y_default = plot_metrics[1] if len(plot_metrics) > 1 else plot_metrics[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-axis", plot_metrics, index=plot_metrics.index(x_default))
    with c2:
        y_axis = st.selectbox("Y-axis", plot_metrics, index=plot_metrics.index(y_default))
    with c3:
        color_by = st.selectbox(
            "Color by",
            options=[o for o in ["Season label", "Main Position", "Team", "League", "Foot", "None"] if o == "None" or o in filtered.columns],
            index=0,
        )

    size_by = st.selectbox(
        "Size by",
        options=[o for o in ["None", "Minutes played", "Market value (M€)", "Age", "Matches played"] if o == "None" or o in filtered.columns],
        index=1 if "Minutes played" in filtered.columns else 0,
    )

    rank_axis = st.radio("Sort Top-N players by", ["X-axis", "Y-axis"], index=1, horizontal=True)
    sort_metric = y_axis if rank_axis == "Y-axis" else x_axis
    plot_limit = st.slider(f"Number of players to plot (Top-N by {sort_metric})", 1, min(30, len(filtered)), min(15, len(filtered)))

    plot_df = filtered.sort_values(by=sort_metric, ascending=False).head(plot_limit).copy()

    if remove_outliers:
        for ax in [x_axis, y_axis]:
            s = pd.to_numeric(plot_df[ax], errors="coerce")
            sd = float(s.std(ddof=0) or 0.0)
            if sd > 0:
                z = (s - float(s.mean())) / sd
                plot_df = plot_df.loc[z.abs() <= 3]

    plot_df[x_axis] = pd.to_numeric(plot_df[x_axis], errors="coerce").round(2)
    plot_df[y_axis] = pd.to_numeric(plot_df[y_axis], errors="coerce").round(2)

    show_labels = st.checkbox("Show player labels on chart", value=False)

    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=y_axis,
        hover_name="Player" if "Player" in plot_df.columns else None,
        color=None if color_by == "None" else color_by,
        size=None if size_by == "None" else size_by,
        text=plot_df["Player"] if show_labels and "Player" in plot_df.columns else None,
        title=f"{y_axis} vs. {x_axis} by Player",
        template="plotly_white",
        height=620,
    )

    fig.update_traces(
        marker=dict(size=16, line=dict(width=1.5, color="DarkSlateGrey")),
        textposition="top center",
        textfont=dict(size=16, color="black"),
        hovertemplate="Player: %{hovertext}<br>" + x_axis + ": %{x:.2f}<br>" + y_axis + ": %{y:.2f}<extra></extra>",
        cliponaxis=False,
    )
    fig.update_layout(
        font=dict(size=14),
        title_font=dict(size=20),
        legend=dict(font=dict(size=12)),
        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=12)),
        yaxis=dict(title_font=dict(size=16), tickfont=dict(size=12)),
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# Compare players + Radar (no Pandas Styler)
# =========================
st.subheader("Compare Selected Players")

player_options = sorted(filtered["Player"].dropna().unique().tolist()) if "Player" in filtered.columns else []
compare_players = st.multiselect("Players to compare (max 5 recommended)", options=player_options, default=[])

if compare_players and "Player" in filtered.columns:
    comp_df = filtered.loc[filtered["Player"].isin(compare_players)].set_index("Player")

    comp_metric_choices = get_numeric_columns(filtered)
    default_comp = [m for m in profile_metrics_in_use if m in comp_metric_choices]
    if calc_col_name and calc_col_name in comp_metric_choices:
        default_comp = [calc_col_name] + default_comp

    if not default_comp:
        fallback = [
            "Goals per 90",
            "Assists per 90",
            "xG per 90",
            "xA per 90",
            "Successful defensive actions per 90",
            "Duels won, %",
        ]
        if calc_col_name and calc_col_name in comp_metric_choices:
            fallback = [calc_col_name] + fallback
        default_comp = [m for m in fallback if m in comp_metric_choices] or comp_metric_choices[:6]

    comp_metrics = st.multiselect("Metrics for comparison table & radar", options=comp_metric_choices, default=default_comp)

    if comp_metrics:
        show_table = comp_df[comp_metrics].copy()
        for c in show_table.columns:
            show_table[c] = pd.to_numeric(show_table[c], errors="coerce").round(2)

        st.dataframe(show_table.T, use_container_width=True)

        base = filtered.set_index("Player")
        baseX = base[comp_metrics].apply(pd.to_numeric, errors="coerce")
        means = baseX.mean(axis=0)
        stds = baseX.std(axis=0, ddof=0).replace(0, np.nan)

        theta = comp_metrics
        fig_radar = go.Figure()
        for player in compare_players:
            row = show_table.loc[player, comp_metrics].apply(pd.to_numeric, errors="coerce")
            z = ((row - means) / stds).fillna(0.0)

            for m in theta:
                if m in LOWER_IS_BETTER:
                    z[m] = -z[m]

            r = z.to_list()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r + [r[0]],
                    theta=theta + [theta[0]],
                    fill="toself",
                    name=player,
                    text=[f"{player}: z={val:.2f}" for val in r] + [f"{player}: z={r[0]:.2f}"],
                    hoverinfo="text",
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            showlegend=True,
            template="plotly_white",
            height=640,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        csv_buf2 = StringIO()
        show_table[comp_metrics].to_csv(csv_buf2)
        st.download_button(
            "⬇️ Download comparison (CSV)",
            data=csv_buf2.getvalue(),
            file_name="player_comparison.csv",
            mime="text/csv",
        )
    else:
        st.info("Select metrics to compare players.")
else:
    st.info("Select players above to compare their stats and see a radar chart.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Plotly | Season-aware ✨")
