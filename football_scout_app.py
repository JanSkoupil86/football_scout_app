# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from io import BytesIO, StringIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from profiles_config import (
    PROFILE_CONFIG,
    PROFILES,
    SINGLE_PLAYER_PROFILES,
    DEFAULT_WEIGHTS,
    ROLE_POSITION_HINTS,
    kpi_for_metric,
    validate_profile_config,
)


# =========================
# Page config
# =========================
st.set_page_config(layout="wide", page_title="Advanced Football Scouting App", page_icon="⚽")
st.title("⚽ Advanced Football Player Scouting App — Season-aware Edition")
st.caption("Build: Calculated Metrics v1.1 · Main Position benchmark families")
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
    "xG against per 90",
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


# =========================
# Single-player percentile profiles
# =========================
# These are deliberately separate from the weighted recruitment-score profiles.
# Each tuple is: (KPI group, requested Wyscout metric).


def validate_single_player_metric_design() -> List[str]:
    """
    Single-player performance wheel policy:
    use per-90, percentage, xG/xA per-90, or possession-adjusted metrics.
    Returns any profile metrics that violate the policy.
    """
    invalid = []
    for role, items in SINGLE_PLAYER_PROFILES.items():
        for _, metric in items:
            allowed = (
                "per 90" in metric
                or "%" in metric
                or metric.startswith("PAdj ")
            )
            if not allowed:
                invalid.append(f"{role}: {metric}")
    return invalid


SINGLE_PLAYER_METRIC_POLICY_ISSUES = validate_single_player_metric_design()






def validate_builtin_profile_framework() -> List[str]:
    """Full startup validation for the canonical built-in role framework."""
    issues = validate_profile_config()
    for role, metrics in PROFILES.items():
        if metrics != [m for _, m in SINGLE_PLAYER_PROFILES[role]]:
            issues.append(f"{role}: derived radar and scoring metric order differs.")
        if set(DEFAULT_WEIGHTS[role]) != set(metrics):
            issues.append(f"{role}: weight-map metrics differ from role metrics.")
    return issues


BUILTIN_PROFILE_FRAMEWORK_ISSUES = validate_builtin_profile_framework()


# Main Position eligibility used when a built-in profile is applied.
# Applying a profile updates the Main Position filter to these Wyscout labels.
PROFILE_POSITION_ELIGIBILITY = {
    "Classic Goalkeeper": ["GK"],
    "Sweeper Keeper": ["GK"],
    "Build-Up Keeper": ["GK"],
    "Ball-Playing CB": ["CB", "LCB", "RCB"],
    "Combative CB / Stopper": ["CB", "LCB", "RCB"],
    "Libero / Middle Pin CB": ["CB", "LCB", "RCB"],
    "Wide CB (in 3)": ["CB", "LCB", "RCB"],
    "Defensive Midfielder #6": ["DMF", "LDMF", "RDMF", "LCMF", "RCMF"],
    "Deep-Lying Playmaker": ["DMF", "LDMF", "RDMF", "LCMF", "RCMF"],
    "Box-to-Box Midfielder": ["LCMF", "RCMF"],
    "Playmaker #10": ["AMF"],
    "Full-Back": ["LB", "RB", "LWB", "RWB"],
    "Wing-Back": ["LWB", "RWB", "LB", "RB"],
    "Inverted Full-Back": ["LB", "RB", "LWB", "RWB"],
    "Classic Winger": ["LW", "RW", "LAMF", "RAMF"],
    "Inverted Winger": ["LW", "RW", "LWF", "RWF", "LAMF", "RAMF"],
    "Target Man #9": ["CF"],
    "Poacher": ["CF", "LWF", "RWF"],
    "Pressing Forward": ["CF", "LWF", "RWF"],
    "Creative Forward / False 9": ["CF", "AMF"],
    "Wide Forward / Inside 9": ["LWF", "RWF", "LW", "RW"],
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


@st.cache_data(show_spinner=False, max_entries=3)
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


@st.cache_data(show_spinner=False, max_entries=3)
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

    # Memory optimization: downcast wide numeric columns where possible.
    # This preserves values while reducing RAM for large Wyscout exports.
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Repeated categorical labels are stored once rather than as tens of thousands
    # of duplicate Python strings. Player is deliberately left as text because of
    # its high cardinality and frequent string operations.
    for col in [
        "Team", "League", "Main Position", "Position", "Foot",
        "Birth country", "Passport country", "On loan", "Season label",
    ]:
        if col in df.columns:
            try:
                df[col] = df[col].astype("category")
            except (TypeError, ValueError):
                pass

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



@st.cache_data(show_spinner=False, max_entries=64)
def benchmark_metric_stats(benchmark_numeric: pd.DataFrame) -> pd.DataFrame:
    """Cache benchmark means/SDs/counts; input is already numeric and role-filtered."""
    return pd.DataFrame({
        "mean": benchmark_numeric.mean(axis=0),
        "std": benchmark_numeric.std(axis=0, ddof=0),
        "count": benchmark_numeric.count(axis=0),
    })


@st.cache_data(show_spinner=False, max_entries=32)
def benchmark_population_for_role(
    source_df: pd.DataFrame,
    role_name: str | None,
    mode: str,
    selected_positions: List[str],
) -> pd.DataFrame:
    """Benchmark is intentionally independent from age/team search filters."""
    # Filtering below is read-only, so avoid cloning the entire benchmark source.
    bench = source_df
    if "Main Position" not in bench.columns:
        return bench
    if mode == "Role position" and role_name in PROFILES:
        return bench.loc[position_family_mask(bench["Main Position"], role_name)]
    if mode == "Selected positions" and selected_positions:
        return bench.loc[bench["Main Position"].isin(selected_positions)]
    return bench


def role_kpi_subscores(
    z_values: pd.DataFrame,
    role_name: str,
    resolved_metric_map: Dict[str, str],
    custom_weights_pct: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weighted KPI sub-scores from already direction-aware z-values."""
    out = pd.DataFrame(index=z_values.index)
    for kpi, metric_weights in PROFILE_CONFIG.get(role_name, {}).items():
        cols, ws = [], []
        for requested_metric, default_w in metric_weights.items():
            resolved = resolved_metric_map.get(requested_metric)
            if resolved and resolved in z_values.columns:
                cols.append(resolved)
                ws.append(float((custom_weights_pct or {}).get(resolved, default_w)))
        if not cols:
            continue
        W = pd.Series(ws, index=cols, dtype=float)
        Z = z_values[cols]
        available_w = Z.notna().mul(W, axis=1).sum(axis=1)
        weighted = Z.fillna(0).mul(W, axis=1).sum(axis=1)
        out[f"KPI: {kpi}"] = (weighted / available_w.replace(0, np.nan)).round(2)
    return out

def make_profile_score_vectorized(
    score_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    metrics: List[str],
    weights: np.ndarray,
    new_col: str,
    sparse_threshold: float = 0.95,
    coverage_threshold: float = 0.70,
) -> Tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    Direction-aware weighted z-score.

    Key methodological safeguards:
    - benchmark population is separate from the displayed/search player pool;
    - sparse metrics are assessed on the benchmark;
    - missing player metrics are NOT treated as average;
    - each player's available weights are renormalized;
    - coverage is reported and low-coverage scores are suppressed.
    """
    present = [m for m in metrics if m in score_df.columns and m in benchmark_df.columns]
    if not present:
        out = score_df.copy()
        out[new_col] = np.nan
        out[f"{new_col} Coverage %"] = 0.0
        return out, [], metrics, pd.DataFrame(index=score_df.index)

    B = benchmark_df[present].apply(pd.to_numeric, errors="coerce")
    sparse = sparse_mask(B, threshold=sparse_threshold)
    usable_metrics = [m for m in present if not bool(sparse.get(m, False))]
    skipped_sparse = [m for m in present if m not in usable_metrics]

    out = score_df.copy()
    if not usable_metrics:
        out[new_col] = np.nan
        out[f"{new_col} Coverage %"] = 0.0
        return out, [], skipped_sparse, pd.DataFrame(index=score_df.index)

    w_map = {m: float(w) for m, w in zip(metrics, weights)}
    W = pd.Series({m: w_map.get(m, 0.0) for m in usable_metrics}, dtype=float)
    if W.sum() <= 0:
        W[:] = 1.0
    W = W / W.sum()

    stats = benchmark_metric_stats(B[usable_metrics])
    means = stats["mean"]
    stds = stats["std"].replace(0, np.nan)

    X = score_df[usable_metrics].apply(pd.to_numeric, errors="coerce")
    Z = (X - means) / stds
    flip_cols = [c for c in usable_metrics if c in LOWER_IS_BETTER]
    if flip_cols:
        Z[flip_cols] = -Z[flip_cols]

    available_weight = Z.notna().mul(W, axis=1).sum(axis=1)
    weighted_sum = Z.fillna(0.0).mul(W, axis=1).sum(axis=1)
    score = weighted_sum / available_weight.replace(0, np.nan)
    coverage = available_weight * 100.0

    score = score.where(available_weight >= float(coverage_threshold))
    out[new_col] = score.round(2)
    out[f"{new_col} Coverage %"] = coverage.round(0)

    return out, usable_metrics, skipped_sparse, Z


def defaults_for_resolved(profile_name: str, resolved_metric_names: List[str]) -> List[int]:
    dm = DEFAULT_WEIGHTS.get(profile_name, {})
    dm_norm = {norm_key(k): int(v) for k, v in dm.items()}
    if not resolved_metric_names:
        return []
    equal = max(1, int(100 / len(resolved_metric_names)))
    return [int(dm_norm.get(norm_key(m), equal)) for m in resolved_metric_names]



def percentile_rank_against_population(
    population: pd.Series,
    player_value: float,
    lower_is_better: bool = False,
) -> float:
    """Percentile rank in [0, 100], with ties handled by average rank."""
    s = pd.to_numeric(population, errors="coerce").dropna()
    if s.empty or pd.isna(player_value):
        return np.nan
    less = float((s < player_value).sum())
    equal = float((s == player_value).sum())
    pct = 100.0 * (less + 0.5 * equal) / len(s)
    if lower_is_better:
        pct = 100.0 - pct
    return float(np.clip(pct, 0.0, 100.0))



def percentile_ranks_against_population_vectorized(
    population: pd.Series,
    values: pd.Series,
    lower_is_better: bool = False,
) -> pd.Series:
    """Vectorized equivalent of percentile_rank_against_population.

    Preserves the app's existing midpoint tie handling exactly:
    percentile = 100 * (count_less + 0.5 * count_equal) / N.
    Uses NumPy searchsorted instead of one Python function call per player.
    """
    pop = pd.to_numeric(population, errors="coerce").dropna().to_numpy(dtype=float)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if pop.size == 0:
        return result

    pop.sort()
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid = numeric_values.notna()
    if not valid.any():
        return result

    vals = numeric_values.loc[valid].to_numpy(dtype=float)
    left = np.searchsorted(pop, vals, side="left")
    right = np.searchsorted(pop, vals, side="right")
    pct = 100.0 * (left + 0.5 * (right - left)) / pop.size
    if lower_is_better:
        pct = 100.0 - pct
    result.loc[valid] = np.clip(pct, 0.0, 100.0)
    return result

def position_family_mask(series: pd.Series, role_name: str) -> pd.Series:
    """Best-effort position-relevant benchmark using Main Position text."""
    hints = ROLE_POSITION_HINTS.get(role_name, [])
    if not hints:
        return pd.Series(True, index=series.index)

    def matches(value: object) -> bool:
        s = str(value).upper().strip()
        # Wyscout exports vary: CB, RCB, LCB, Centre Back, etc.
        if "GK" in hints:
            return "GK" in s or "GOALKEEP" in s
        if "CB" in hints:
            return any(x in s for x in ["CB", "CENTRE BACK", "CENTER BACK"])
        if any(x in hints for x in ["LB", "RB", "LWB", "RWB"]):
            return any(x in s for x in ["LB", "RB", "LWB", "RWB", "FULL BACK", "WING BACK"])
        if any(x in hints for x in ["DMF", "CMF", "AMF"]):
            return any(x in s for x in ["DMF", "CMF", "AMF", "DM", "CM", "AM", "MIDFIELD"])
        if any(x in hints for x in ["LW", "RW", "LWF", "RWF", "LMF", "RMF"]):
            return any(x in s for x in ["LW", "RW", "LWF", "RWF", "LMF", "RMF", "WINGER"])
        if any(x in hints for x in ["CF", "ST"]):
            return any(x in s for x in ["CF", "ST", "CENTRE FORWARD", "CENTER FORWARD", "STRIKER"])
        return any(h in s for h in hints)

    return series.apply(matches)


def directional_z_against_population(
    population: pd.Series,
    player_value: float,
    lower_is_better: bool = False,
) -> float:
    """Directional z-score against the selected benchmark population."""
    s = pd.to_numeric(population, errors="coerce").dropna()
    if s.empty or pd.isna(player_value):
        return np.nan
    sd = float(s.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return 0.0
    z = (float(player_value) - float(s.mean())) / sd
    if lower_is_better:
        z = -z
    return float(z)


def build_single_player_profile(
    player_row: pd.Series,
    benchmark_df: pd.DataFrame,
    role_name: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return KPI group, resolved metric, raw value and directional percentile."""
    requested = SINGLE_PLAYER_PROFILES[role_name]
    records = []
    missing = []

    for group, requested_metric in requested:
        resolved, miss = resolve_metrics_aliases([requested_metric], benchmark_df.columns.tolist())
        if not resolved:
            missing.extend(miss or [requested_metric])
            continue
        metric = resolved[0]
        value = pd.to_numeric(pd.Series([player_row.get(metric, np.nan)]), errors="coerce").iloc[0]
        lower_better = metric in LOWER_IS_BETTER
        pct = percentile_rank_against_population(
            benchmark_df[metric],
            value,
            lower_is_better=lower_better,
        )
        z = directional_z_against_population(
            benchmark_df[metric],
            value,
            lower_is_better=lower_better,
        )
        if pd.isna(value) or pd.isna(pct) or pd.isna(z):
            continue
        records.append(
            {
                "KPI Group": group,
                "Metric": metric,
                "Raw Value": float(value),
                "Percentile": float(pct),
                "Z-score": float(z),
            }
        )

    return pd.DataFrame(records), sorted(set(missing))


def performance_band_color(value: float, scale_mode: str) -> str:
    """Performance colour for percentile or direction-aware z-score."""
    if pd.isna(value):
        return "#7F8C8D"
    if scale_mode == "Percentile":
        if value < 25:
            return "#8B1E1E"   # dark red
        if value < 50:
            return "#E57373"   # light red
        if value < 75:
            return "#81C784"   # light green
        return "#1B5E20"       # dark green

    # z thresholds mirror approximately the 25th / 50th / 75th percentiles.
    if value < -0.67:
        return "#8B1E1E"
    if value < 0:
        return "#E57373"
    if value < 0.67:
        return "#81C784"
    return "#1B5E20"


def single_player_wheel(
    profile_df: pd.DataFrame,
    player_name: str,
    subtitle: str,
    scale_mode: str = "Percentile",
) -> go.Figure:
    """
    Fixed-radius scouting wheel.

    Design principle:
    - every metric receives equal visual space;
    - KPI groups are structural coloured bands, not variable-length wedges;
    - performance is shown by a dot on a fixed radial scale;
    - percentile and z-score therefore remain readable even for low values;
    - the centre is reserved for player/profile context.
    """
    if profile_df.empty:
        return go.Figure()

    p = profile_df.reset_index(drop=True).copy()
    groups = p["KPI Group"].drop_duplicates().tolist()

    # Neutral categorical palette: colour = KPI family, NOT performance quality.
    # Avoid traffic-light red/green/orange semantics.
    # High-separation categorical palette for KPI families.
    # Red and green are deliberately excluded because those colours are reserved
    # for performance bands on the radial lines / endpoint dots.
    palette = [
        "#3478B8",  # blue
        "#8E5AB5",  # purple
        "#16A6B6",  # cyan / teal
        "#E58A2B",  # orange
        "#244A73",  # navy
        "#D4A72C",  # gold
        "#C44E8B",  # magenta
        "#687684",  # slate
    ]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # Presentation aliases only; underlying Wyscout columns remain untouched.
    display_aliases = {
        "Save rate, %": "Save Rate %",
        "Prevented goals per 90": "Goals Prevented /90",
        "Conceded goals per 90": "Goals Conceded /90 ↓",
        "Shots against per 90": "Shots Faced /90",
        "xG against per 90": "xGA /90",
        "Exits per 90": "Exits /90",
        "Aerial duels per 90.1": "Aerial Duels /90",
        "Aerial duels per 90": "Aerial Duels /90",
        "Aerial duels won, %": "Aerial Duel Win %",
        "Passes per 90": "Passes /90",
        "Accurate passes, %": "Pass Accuracy %",
        "Long passes per 90": "Long Passes /90",
        "Accurate long passes, %": "Long Pass Accuracy %",
        "Back passes received as GK per 90": "GK Back Passes /90",
        "Forward passes per 90": "Forward Passes /90",
        "Accurate forward passes, %": "Forward Pass Accuracy %",
        "Progressive passes per 90": "Progressive Passes /90",
        "Accurate progressive passes, %": "Progressive Pass Accuracy %",
        "Passes to final third per 90": "Final Third Passes /90",
        "Received passes per 90": "Passes Received /90",
        "Progressive runs per 90": "Progressive Runs /90",
        "Interceptions per 90": "Interceptions /90",
        "PAdj Interceptions": "PAdj Interceptions",
        "Successful defensive actions per 90": "Defensive Actions /90",
        "Defensive duels per 90": "Defensive Duels /90",
        "Defensive duels won, %": "Defensive Duel Win %",
        "PAdj Sliding tackles": "PAdj Sliding Tackles",
        "Sliding tackles per 90": "Sliding Tackles /90",
        "Shots blocked per 90": "Shots Blocked /90",
        "Fouls per 90": "Fouls /90 ↓",
        "Yellow cards per 90": "Yellow Cards /90 ↓",
        "Dribbles per 90": "Dribbles /90",
        "Successful dribbles, %": "Dribble Success %",
        "Accelerations per 90": "Accelerations /90",
        "Crosses per 90": "Crosses /90",
        "Accurate crosses, %": "Cross Accuracy %",
        "Crosses to goalie box per 90": "Box Crosses /90",
        "Passes to penalty area per 90": "Penalty Area Passes /90",
        "Shot assists per 90": "Shot Assists /90",
        "xA per 90": "xA /90",
        "xG per 90": "xG /90",
        "Shots per 90": "Shots /90",
        "Touches in box per 90": "Box Touches /90",
        "Non-penalty goals per 90": "Non-Penalty Goals /90",
        "Successful attacking actions per 90": "Attacking Actions /90",
        "Smart passes per 90": "Smart Passes /90",
        "Key passes per 90": "Key Passes /90",
        "Deep completions per 90": "Deep Completions /90",
        "Through passes per 90": "Through Passes /90",
        "Offensive duels per 90": "Offensive Duels /90",
        "Offensive duels won, %": "Offensive Duel Win %",
        "Goal conversion, %": "Goal Conversion %",
        "Shots on target, %": "Shots on Target %",
        "Head goals per 90": "Headed Goals /90",
        "Received long passes per 90": "Long Passes Received /90",
        "Fouls suffered per 90": "Fouls Won /90",
    }
    labels = [display_aliases.get(m, m) for m in p["Metric"]]
    p["Display Metric"] = labels

    n = len(p)
    # Keep each metric equal-width, but insert a small angular gap at KPI boundaries.
    group_seq = p["KPI Group"].tolist()
    gap_units = 0.48
    boundaries = sum(1 for i in range(1, n) if group_seq[i] != group_seq[i - 1])
    unit = 360.0 / (n + boundaries * gap_units)

    theta = []
    cursor = 0.0
    for i in range(n):
        if i > 0 and group_seq[i] != group_seq[i - 1]:
            cursor += gap_units * unit
        theta.append(cursor)
        cursor += unit
    theta = np.array(theta, dtype=float)
    width = unit * 0.86

    # Thin fixed-radius KPI band: supporting context, while the inner profile is the focal layer.
    band_inner = 78.0
    band_outer = 86.0
    band_len = band_outer - band_inner

    fig = go.Figure()

    for group in groups:
        idx = p.index[p["KPI Group"] == group].tolist()
        positions = [float(theta[i]) for i in idx]
        fig.add_trace(
            go.Barpolar(
                r=[band_len] * len(idx),
                base=[band_inner] * len(idx),
                theta=positions,
                width=[width] * len(idx),
                marker_color=group_colors[group],
                marker_line_color="white",
                marker_line_width=1.2,
                opacity=0.88,
                name=group,
                hoverinfo="skip",
            )
        )

    # Enlarged performance annulus so the player profile is visually dominant.
    perf_inner = 20.0
    perf_outer = 74.0
    perf_span = perf_outer - perf_inner

    if scale_mode == "Percentile":
        values = np.clip(p["Percentile"].to_numpy(dtype=float), 0, 100)
        marker_r = perf_inner + (values / 100.0) * perf_span
        value_text = [f"{int(round(v))}" for v in values]
        scale_ticks = [0, 25, 50, 75, 100]
        scale_r = [perf_inner + (v / 100.0) * perf_span for v in scale_ticks]
        scale_text = [str(v) for v in scale_ticks]
    else:
        raw_z = p["Z-score"].to_numpy(dtype=float)
        clipped = np.clip(raw_z, -2.0, 2.0)
        marker_r = perf_inner + ((clipped + 2.0) / 4.0) * perf_span
        value_text = [f"{v:+.2f}" for v in raw_z]
        scale_ticks = [-2, -1, 0, 1, 2]
        scale_r = [perf_inner + ((v + 2.0) / 4.0) * perf_span for v in scale_ticks]
        scale_text = [f"{v:+d}" if v != 0 else "0" for v in scale_ticks]

    performance_values = (
        p["Percentile"].to_numpy(dtype=float)
        if scale_mode == "Percentile"
        else p["Z-score"].to_numpy(dtype=float)
    )
    performance_colors = [
        performance_band_color(v, scale_mode) for v in performance_values
    ]

    # Calm reference rings. In the default percentile view, 50 is the
    # benchmark median and 25/75 provide quick orientation.
    ring_theta = np.linspace(0, 360, 361)
    for tick, rv in zip(scale_ticks, scale_r):
        if scale_mode == "Percentile":
            if tick == 50:
                ring_color, ring_width = "rgba(45,55,65,0.55)", 2.6
            elif tick in (25, 75):
                ring_color, ring_width = "rgba(95,105,115,0.14)", 1.15
            else:
                ring_color, ring_width = "rgba(120,130,140,0.10)", 1.0
        else:
            if tick == 0:
                ring_color, ring_width = "rgba(45,55,65,0.55)", 2.6
            else:
                ring_color, ring_width = "rgba(120,130,140,0.14)", 1.0
        fig.add_trace(
            go.Scatterpolar(
                r=[rv] * len(ring_theta),
                theta=ring_theta,
                mode="lines",
                line=dict(color=ring_color, width=ring_width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    custom = np.column_stack([
        p["Raw Value"].to_numpy(dtype=float),
        p["Percentile"].to_numpy(dtype=float),
        p["Z-score"].to_numpy(dtype=float),
        p["Metric"].astype(str).to_numpy(),
        p["KPI Group"].astype(str).to_numpy(),
        p["Display Metric"].astype(str).to_numpy(),
    ])

    # Thin spokes make it easy to associate marker, number and metric.
    for i in range(n):
        fig.add_trace(
            go.Scatterpolar(
                r=[perf_inner, marker_r[i]],
                theta=[theta[i], theta[i]],
                mode="lines",
                line=dict(color=performance_colors[i], width=9.5),
                opacity=0.82,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Performance markers: the primary quantitative encoding.
    marker_colors = performance_colors
    fig.add_trace(
        go.Scatterpolar(
            r=marker_r,
            theta=theta,
            mode="markers",
            marker=dict(
                size=21,
                color=marker_colors,
                line=dict(color="white", width=2.4),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[5]}</b><br>"
                "KPI: %{customdata[4]}<br>"
                "Raw value: %{customdata[0]:.2f}<br>"
                "Percentile: %{customdata[1]:.0f}<br>"
                "Z-score: %{customdata[2]:+.2f}<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Numerical result sits in the fixed KPI tile rather than collapsing toward centre.
    fig.add_trace(
        go.Scatterpolar(
            r=[82.0] * n,
            theta=theta,
            mode="text",
            text=[f"<b>{v}</b>" for v in value_text],
            textfont=dict(size=17, color="white"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # KPI group names are intentionally shown only in the legend.

    # Centre content.
    role_label = subtitle.split("|")[0].strip() if subtitle else ""
    centre_text = (
        f"<b>{player_name}</b>"
        + (f"<br><span style='font-size:11px'>{role_label}</span>" if role_label else "")
    )
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=centre_text,
        showarrow=False,
        align="center",
        font=dict(size=16, color="#17202A"),
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(120,130,140,0.22)",
        borderwidth=1,
        borderpad=11,
    )

    # Scale key in the lower-left, avoiding radial-axis clutter.
    key = "Percentile: 0–100" if scale_mode == "Percentile" else "Z-score: −2 to +2 · 0 = benchmark mean"
    fig.add_annotation(
        x=0.01, y=0.01,
        xref="paper", yref="paper",
        text=key,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=10, color="#6B7785"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            y=0.985,
            font=dict(size=20, color="#17202A"),
        ),
        template="plotly_white",
        height=920,
        margin=dict(l=180, r=180, t=132, b=125),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.025,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=10),
        ),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=[0, 96],
                visible=False,
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta,
                ticktext=labels,
                direction="clockwise",
                rotation=90,
                gridcolor="rgba(255,255,255,0)",
                tickfont=dict(size=16, color="#303A44"),
                showline=False,
            ),
        ),
        barmode="overlay",
    )

    return fig


def multi_player_profile_wheel(
    raw_table: pd.DataFrame,
    z_table: pd.DataFrame,
    percentile_table: pd.DataFrame,
    display_scale: str,
    players: List[str],
    profile_name: str,
    metrics: List[str],
    kpi_lookup: Dict[str, str],
) -> go.Figure:
    """
    Multi-player version of the fixed-radius scouting wheel.

    It deliberately mirrors the single-player profile:
    - identical KPI band architecture;
    - identical metric order and KPI gaps;
    - identical -2 to +2 z-score mapping;
    - multiple players shown as lines + markers rather than overlapping fills.
    """
    if not players or not metrics:
        return go.Figure()

    display_aliases = {
        "Save rate, %": "Save Rate %",
        "Prevented goals per 90": "Goals Prevented /90",
        "Conceded goals per 90": "Goals Conceded /90",
        "Shots against per 90": "Shots Faced /90",
        "xG against per 90": "xGA /90",
        "Exits per 90": "Exits /90",
        "Aerial duels per 90.1": "Aerial Duels /90",
        "Aerial duels per 90": "Aerial Duels /90",
        "Aerial duels won, %": "Aerial Duel Win %",
        "Passes per 90": "Passes /90",
        "Accurate passes, %": "Pass Accuracy %",
        "Long passes per 90": "Long Passes /90",
        "Accurate long passes, %": "Long Pass Accuracy %",
        "Back passes received as GK per 90": "GK Back Passes /90",
        "Forward passes per 90": "Forward Passes /90",
        "Accurate forward passes, %": "Forward Pass Accuracy %",
        "Progressive passes per 90": "Progressive Passes /90",
        "Accurate progressive passes, %": "Progressive Pass Accuracy %",
        "Passes to final third per 90": "Final Third Passes /90",
        "Received passes per 90": "Passes Received /90",
        "Progressive runs per 90": "Progressive Runs /90",
        "Interceptions per 90": "Interceptions /90",
        "PAdj Interceptions": "PAdj Interceptions",
        "Successful defensive actions per 90": "Defensive Actions /90",
        "Defensive duels per 90": "Defensive Duels /90",
        "Defensive duels won, %": "Defensive Duel Win %",
        "PAdj Sliding tackles": "PAdj Sliding Tackles",
        "Sliding tackles per 90": "Sliding Tackles /90",
        "Shots blocked per 90": "Shots Blocked /90",
        "Fouls per 90": "Fouls /90",
        "Yellow cards per 90": "Yellow Cards /90",
        "Dribbles per 90": "Dribbles /90",
        "Successful dribbles, %": "Dribble Success %",
        "Accelerations per 90": "Accelerations /90",
        "Crosses per 90": "Crosses /90",
        "Accurate crosses, %": "Cross Accuracy %",
        "Crosses to goalie box per 90": "Box Crosses /90",
        "Passes to penalty area per 90": "Penalty Area Passes /90",
        "Shot assists per 90": "Shot Assists /90",
        "xA per 90": "xA /90",
        "xG per 90": "xG /90",
        "Shots per 90": "Shots /90",
        "Touches in box per 90": "Box Touches /90",
        "Non-penalty goals per 90": "Non-Penalty Goals /90",
        "Successful attacking actions per 90": "Attacking Actions /90",
        "Smart passes per 90": "Smart Passes /90",
        "Key passes per 90": "Key Passes /90",
        "Deep completions per 90": "Deep Completions /90",
        "Through passes per 90": "Through Passes /90",
        "Offensive duels per 90": "Offensive Duels /90",
        "Offensive duels won, %": "Offensive Duel Win %",
        "Goal conversion, %": "Goal Conversion %",
        "Shots on target, %": "Shots on Target %",
        "Head goals per 90": "Headed Goals /90",
        "Received long passes per 90": "Long Passes Received /90",
        "Fouls suffered per 90": "Fouls Won /90",
    }

    labels = [display_aliases.get(m, m) for m in metrics]
    groups = []
    for m in metrics:
        g = kpi_lookup.get(m, "Profile")
        if g not in groups:
            groups.append(g)

    # Neutral categorical palette: colour = KPI family, NOT performance quality.
    # Avoid traffic-light red/green/orange semantics.
    # High-separation categorical palette for KPI families.
    # Red and green are deliberately excluded because those colours are reserved
    # for performance bands on the radial lines / endpoint dots.
    palette = [
        "#3478B8",  # blue
        "#8E5AB5",  # purple
        "#16A6B6",  # cyan / teal
        "#E58A2B",  # orange
        "#244A73",  # navy
        "#D4A72C",  # gold
        "#C44E8B",  # magenta
        "#687684",  # slate
    ]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    group_seq = [kpi_lookup.get(m, "Profile") for m in metrics]
    n = len(metrics)
    gap_units = 0.48
    boundaries = sum(1 for i in range(1, n) if group_seq[i] != group_seq[i - 1])
    unit = 360.0 / (n + boundaries * gap_units)

    theta = []
    cursor = 0.0
    for i in range(n):
        if i > 0 and group_seq[i] != group_seq[i - 1]:
            cursor += gap_units * unit
        theta.append(cursor)
        cursor += unit
    theta = np.array(theta, dtype=float)
    width = unit * 0.90

    fig = go.Figure()

    # Same KPI band as the single-player wheel.
    band_inner = 80.0
    band_outer = 90.0
    band_len = band_outer - band_inner
    for group in groups:
        idx = [i for i, g in enumerate(group_seq) if g == group]
        fig.add_trace(
            go.Barpolar(
                r=[band_len] * len(idx),
                base=[band_inner] * len(idx),
                theta=[float(theta[i]) for i in idx],
                width=[width] * len(idx),
                marker_color=group_colors[group],
                marker_line_color="white",
                marker_line_width=1.2,
                opacity=0.88,
                name=group,
                legendgroup=f"kpi_{group}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Same fixed-radius performance annulus as the single-player wheel.
    # Only the statistical scale changes; geometry/KPI structure stays identical.
    perf_inner = 20.0
    perf_outer = 76.0
    perf_span = perf_outer - perf_inner
    ring_theta = np.linspace(0, 360, 361)

    if display_scale == "Percentile":
        scale_ticks = [0, 25, 50, 75, 100]
        reference_tick = 50
        def map_to_radius(v: float) -> float:
            vv = float(np.clip(v, 0.0, 100.0))
            return perf_inner + (vv / 100.0) * perf_span
    else:
        scale_ticks = [-2, -1, 0, 1, 2]
        reference_tick = 0
        def map_to_radius(v: float) -> float:
            vv = float(np.clip(v, -2.0, 2.0))
            return perf_inner + ((vv + 2.0) / 4.0) * perf_span

    for tick in scale_ticks:
        rv = map_to_radius(tick)
        fig.add_trace(
            go.Scatterpolar(
                r=[rv] * len(ring_theta),
                theta=ring_theta,
                mode="lines",
                line=dict(
                    color="rgba(45,55,65,0.62)" if tick == reference_tick else "rgba(120,130,140,0.16)",
                    width=2.8 if tick == reference_tick else 1,
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Metric spokes stay neutral so several player traces remain readable.
    for angle in theta:
        fig.add_trace(
            go.Scatterpolar(
                r=[perf_inner, perf_outer],
                theta=[angle, angle],
                mode="lines",
                line=dict(color="rgba(120,130,140,0.16)", width=1.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Multi-player performance: no fills, only lines + markers.
    # Plotly assigns a distinct player color automatically.
    for player in players:
        if player not in z_table.index:
            continue

        z = pd.to_numeric(z_table.loc[player, metrics], errors="coerce").fillna(0.0)
        pct = pd.to_numeric(percentile_table.loc[player, metrics], errors="coerce").fillna(50.0)
        raw = pd.to_numeric(raw_table.loc[player, metrics], errors="coerce")

        display_values = pct if display_scale == "Percentile" else z
        marker_r = np.array([map_to_radius(v) for v in display_values], dtype=float)

        custom = np.column_stack([
            raw.to_numpy(dtype=float),
            z.to_numpy(dtype=float),
            pct.to_numpy(dtype=float),
            np.array([kpi_lookup.get(m, "Profile") for m in metrics], dtype=object),
            np.array(labels, dtype=object),
        ])

        fig.add_trace(
            go.Scatterpolar(
                r=list(marker_r) + [float(marker_r[0])],
                theta=list(theta) + [float(theta[0])],
                mode="lines+markers",
                name=player,
                line=dict(width=6.5),
                marker=dict(size=15, line=dict(color="white", width=2.2)),
                customdata=np.vstack([custom, custom[0]]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{customdata[4]}<br>"
                    "KPI: %{customdata[3]}<br>"
                    "Raw value: %{customdata[0]:.2f}<br>"
                    "Percentile: %{customdata[2]:.0f}<br>"
                    "Z-score: %{customdata[1]:+.2f}<extra></extra>"
                ),
            )
        )

    # Centre context mirrors the single-player chart.
    if display_scale == "Percentile":
        centre_text = "<b>Percentile</b><br><span style='font-size:11px'>Higher = better · 50 = median</span>"
    else:
        centre_text = "<b>Z-score</b><br><span style='font-size:11px'>0 = benchmark mean</span>"
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=centre_text,
        showarrow=False,
        align="center",
        font=dict(size=14, color="#17202A"),
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(120,130,140,0.22)",
        borderwidth=1,
        borderpad=8,
    )



    fig.update_layout(
        title=dict(
            text=f"<b>{profile_name}</b> — Multi-Player Role Comparison",
            x=0.5,
            xanchor="center",
            y=0.998,
            font=dict(size=22, color="#17202A"),
        ),
        template="plotly_white",
        height=1020,
        margin=dict(l=45, r=45, t=145, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.035,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=12),
        ),
        polar=dict(
            domain=dict(x=[0.03, 0.97], y=[0.015, 0.915]),
            bgcolor="white",
            radialaxis=dict(range=[0, 93], visible=False),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta,
                ticktext=labels,
                direction="clockwise",
                rotation=90,
                gridcolor="rgba(255,255,255,0)",
                tickfont=dict(size=16, color="#303A44"),
                showline=False,
            ),
        ),
        barmode="overlay",
    )

    # KPI legend, matching the single-player visual language, is kept separate
    # from the player legend so player colours remain unambiguous.
    kpi_text = "  ·  ".join(
        f"<span style='color:{group_colors[g]}'><b>■ {g}</b></span>" for g in groups
    )
    fig.add_annotation(
        x=0.5, y=0.982,
        xref="paper", yref="paper",
        text=kpi_text,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=13)
    )

    return fig



# =========================
# Calculated Metrics v1
# =========================
POSITION_FAMILIES = {
    "Goalkeeper": {"GK"},
    "Centre-Back": {"CB", "LCB", "RCB"},
    "Full-Back / Wing-Back": {"LB", "RB", "LWB", "RWB"},
    "Central Midfield": {"DMF", "LDMF", "RDMF", "CMF", "LCMF", "RCMF"},
    "Attacking Midfield": {"AMF"},
    "Winger": {"LW", "RW", "LAMF", "RAMF"},
    "Forward": {"CF", "LWF", "RWF"},
}

CALCULATED_METRICS = {
    "Calc: GK Shot Stopping": {"families": ["Goalkeeper"], "components": {"Save rate, %": 40, "Prevented goals per 90": 35, "Conceded goals per 90": 15, "xG against per 90": 10}},
    "Calc: GK Area Control": {"families": ["Goalkeeper"], "components": {"Exits per 90": 40, "Aerial duels per 90.1": 25, "Aerial duels won, %": 35}},
    "Calc: GK Distribution": {"families": ["Goalkeeper"], "components": {"Accurate passes, %": 20, "Accurate long passes, %": 30, "Accurate forward passes, %": 20, "Progressive passes per 90": 20, "Back passes received as GK per 90": 10}},
    "Calc: Defensive Activity": {"families": ["Centre-Back", "Full-Back / Wing-Back", "Central Midfield"], "components": {"Successful defensive actions per 90": 35, "PAdj Interceptions": 35, "PAdj Sliding tackles": 30}},
    "Calc: Duel Dominance": {"families": ["Centre-Back", "Full-Back / Wing-Back", "Central Midfield"], "components": {"Defensive duels won, %": 60, "Aerial duels won, %": 40}},
    "Calc: Aerial Impact": {"families": ["Centre-Back", "Central Midfield", "Forward"], "components": {"Aerial duels per 90": 45, "Aerial duels won, %": 55}},
    "Calc: Passing Progression": {"families": ["Centre-Back", "Full-Back / Wing-Back", "Central Midfield", "Attacking Midfield"], "components": {"Progressive passes per 90": 50, "Passes to final third per 90": 30, "Forward passes per 90": 20}},
    "Calc: Carrying Progression": {"families": ["Full-Back / Wing-Back", "Central Midfield", "Attacking Midfield", "Winger"], "components": {"Progressive runs per 90": 40, "Dribbles per 90": 30, "Accelerations per 90": 30}},
    "Calc: Wide Delivery": {"families": ["Full-Back / Wing-Back", "Winger"], "components": {"Crosses per 90": 25, "Accurate crosses, %": 30, "Crosses to goalie box per 90": 25, "Passes to penalty area per 90": 20}},
    "Calc: 1v1 Threat": {"families": ["Attacking Midfield", "Winger"], "components": {"Dribbles per 90": 35, "Successful dribbles, %": 35, "Offensive duels won, %": 30}},
    "Calc: Chance Creation": {"families": ["Full-Back / Wing-Back", "Central Midfield", "Attacking Midfield", "Winger", "Forward"], "components": {"xA per 90": 40, "Shot assists per 90": 30, "Key passes per 90": 20, "Deep completions per 90": 10}},
    "Calc: Box Threat": {"families": ["Attacking Midfield", "Winger", "Forward"], "components": {"xG per 90": 40, "Touches in box per 90": 30, "Shots per 90": 20, "Non-penalty goals per 90": 10}},
    "Calc: Finishing": {"families": ["Winger", "Forward"], "components": {"Non-penalty goals per 90": 35, "Goal conversion, %": 25, "Shots on target, %": 20, "xG per 90": 20}},
    "Calc: Link Play": {"families": ["Forward"], "components": {"Received passes per 90": 25, "Received long passes per 90": 20, "Accurate passes, %": 25, "Shot assists per 90": 15, "xA per 90": 15}},
}

EFFICIENCY_METRICS = [
    "Calc: Shot Quality (xG/Shot)",
    "Calc: Finishing Above xG /90",
    "Calc: Chance Quality (xA/Key Pass)",
]
CALCULATED_INDEX_NAMES = list(CALCULATED_METRICS.keys())

def calculated_position_family(value: object) -> str | None:
    s = str(value or "").strip().upper()
    for family, positions in POSITION_FAMILIES.items():
        if s in positions:
            return family
    return None

def add_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def n(col):
        return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(np.nan, index=out.index)
    shots = n("Shots per 90").replace(0, np.nan)
    key_passes = n("Key passes per 90").replace(0, np.nan)
    out["Calc: Shot Quality (xG/Shot)"] = n("xG per 90") / shots
    out["Calc: Finishing Above xG /90"] = n("Non-penalty goals per 90") - n("xG per 90")
    out["Calc: Chance Quality (xA/Key Pass)"] = n("xA per 90") / key_passes
    return out

@st.cache_data(show_spinner=False, max_entries=12)
def add_calculated_indices(score_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    out = add_efficiency_metrics(score_df)
    bench = add_efficiency_metrics(benchmark_df)
    families_score = out["Main Position"].map(calculated_position_family)
    families_bench = bench["Main Position"].map(calculated_position_family)
    active_score_families = set(families_score.dropna().unique().tolist())

    for index_name, cfg in CALCULATED_METRICS.items():
        out[index_name] = np.nan

        # Skip the whole composite when it is irrelevant to every player in the
        # current Players workspace. The output column still exists, preserving
        # the app's schema and selector logic.
        relevant_families = [
            family for family in cfg["families"]
            if family in active_score_families
        ]
        if not relevant_families:
            continue

        for family in relevant_families:
            score_mask = families_score.eq(family)
            bench_mask = families_bench.eq(family)
            if not score_mask.any() or not bench_mask.any():
                continue
            requested = list(cfg["components"].keys())
            resolved, _ = resolve_metrics_aliases(requested, bench.columns.tolist())
            req_to_col = {}
            for req in requested:
                rr, _ = resolve_metrics_aliases([req], bench.columns.tolist())
                if rr and rr[0] in out.columns:
                    req_to_col[req] = rr[0]
            cols = list(dict.fromkeys(req_to_col.values()))
            if not cols:
                continue
            B = bench.loc[bench_mask, cols].apply(pd.to_numeric, errors="coerce")
            sparse = sparse_mask(B, threshold=0.95)
            cols = [c for c in cols if not bool(sparse.get(c, False))]
            if not cols:
                continue
            stats = benchmark_metric_stats(B[cols])
            means, stds = stats["mean"], stats["std"].replace(0, np.nan)
            W = pd.Series({req_to_col[r]: float(w) for r, w in cfg["components"].items() if r in req_to_col and req_to_col[r] in cols})
            W = W.groupby(level=0).sum(); W = W / W.sum()
            Zb = (B[cols] - means) / stds
            flip = [c for c in cols if c in LOWER_IS_BETTER]
            if flip: Zb[flip] = -Zb[flip]
            avail_b = Zb.notna().mul(W, axis=1).sum(axis=1)
            score_b = Zb.fillna(0).mul(W, axis=1).sum(axis=1) / avail_b.replace(0, np.nan)
            score_b = score_b.where(avail_b >= 0.70).dropna()
            X = out.loc[score_mask, cols].apply(pd.to_numeric, errors="coerce")
            Z = (X - means) / stds
            if flip: Z[flip] = -Z[flip]
            avail = Z.notna().mul(W, axis=1).sum(axis=1)
            composite = Z.fillna(0).mul(W, axis=1).sum(axis=1) / avail.replace(0, np.nan)
            composite = composite.where(avail >= 0.70)
            out.loc[score_mask, index_name] = percentile_ranks_against_population_vectorized(
                score_b, composite, lower_is_better=False
            )
    return out

# =========================
# Cached benchmark scoring package
# =========================
@st.cache_data(show_spinner=False, max_entries=64)
def build_benchmark_scoring_package(
    benchmark_numeric: pd.DataFrame,
    metrics: Tuple[str, ...],
    weights: Tuple[float, ...],
    coverage_threshold: float,
) -> Dict[str, object]:
    """
    Build the reusable mathematical part of a weighted role/profile benchmark.

    The package contains the benchmark mean/std, direction-adjusted Z matrix,
    normalized metric weights, row-level coverage and weighted benchmark score
    distribution. Because all benchmark-defining inputs are arguments, Streamlit
    can safely reuse it only for the identical benchmark/metric/weight setup.
    """
    cols = [m for m in metrics if m in benchmark_numeric.columns]
    if not cols:
        return {
            "usable": [], "means": pd.Series(dtype=float),
            "stds": pd.Series(dtype=float), "weights": pd.Series(dtype=float),
            "benchmark_z": pd.DataFrame(index=benchmark_numeric.index),
            "benchmark_scores": pd.Series(np.nan, index=benchmark_numeric.index, dtype=float),
        }

    B = benchmark_numeric.loc[:, cols]
    # Preprocessing already makes Wyscout feature columns numeric. Keep a
    # defensive conversion only when an unexpected non-numeric column arrives.
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(B[c])]
    if non_numeric:
        B = B.copy()
        B[non_numeric] = B[non_numeric].apply(pd.to_numeric, errors="coerce")

    sparse = sparse_mask(B, threshold=0.95)
    usable = [m for m in cols if not bool(sparse.get(m, False))]
    if not usable:
        return {
            "usable": [], "means": pd.Series(dtype=float),
            "stds": pd.Series(dtype=float), "weights": pd.Series(dtype=float),
            "benchmark_z": pd.DataFrame(index=B.index),
            "benchmark_scores": pd.Series(np.nan, index=B.index, dtype=float),
        }

    weight_map = {m: float(w) for m, w in zip(metrics, weights)}
    W = pd.Series({m: weight_map.get(m, 0.0) for m in usable}, dtype=float)
    if float(W.sum()) <= 0:
        W[:] = 1.0
    W = W / float(W.sum())

    stats = benchmark_metric_stats(B[usable])
    means = stats["mean"]
    stds = stats["std"].replace(0, np.nan)

    Zb = (B[usable] - means) / stds
    flip = [m for m in usable if m in LOWER_IS_BETTER]
    if flip:
        Zb[flip] = -Zb[flip]

    avail_b = Zb.notna().mul(W, axis=1).sum(axis=1)
    score_b = (
        Zb.fillna(0.0).mul(W, axis=1).sum(axis=1)
        / avail_b.replace(0, np.nan)
    )
    score_b = score_b.where(avail_b >= float(coverage_threshold))

    return {
        "usable": usable,
        "means": means,
        "stds": stds,
        "weights": W,
        "benchmark_z": Zb,
        "benchmark_scores": score_b,
    }




# =========================
# League Strength / League Projection v1
# =========================
# Empirical calibration from consecutive-season cross-league movers in the
# supplied Wyscout dataset. 600+ minutes on both sides was selected as the V1
# balance between sample size and reliability. One Opta league-rating point is
# associated with ~0.020 role-score SD of translation difficulty.
LEAGUE_TRANSLATION_BETA = 0.020
OPTA_LEAGUE_META_PATH = Path(__file__).with_name("league-meta.json")

# Wyscout -> Opta naming exceptions. The matcher still validates these against
# the bundled Opta league metadata and falls back to conservative fuzzy matching.
OPTA_LEAGUE_ALIASES = {
    "england premier league": "English Premier League",
    "england championship": "English Championship",
    "england league one": "English League One",
    "england league two": "English League Two",
    "england national league": "English National League",
    "germany bundesliga": "German Bundesliga",
    "germany 2 bundesliga": "German 2. Bundesliga",
    "germany 3 liga": "German 3. Liga",
    "france ligue 1": "French Ligue 1",
    "france ligue 2": "French Ligue 2",
    "france championnat national": "French National",
    "italy serie a": "Italian Serie A",
    "italy serie b": "Italian Serie B",
    "spain la liga": "Spanish La Liga",
    "spain segunda": "Spanish Segunda División",
    "spain primera division rfef": "Spanish Primera Division RFEF",
    "netherlands eredivisie": "Dutch Eredivisie",
    "netherlands eerste divisie": "Dutch Eerste Divisie",
    "portugal primeira liga": "Portuguese Primeira Liga",
    "portugal segunda liga": "Portuguese Segunda Liga",
    "belgium first division a": "Belgian First Division A",
    "belgium first division b": "Belgian First Division B",
    "czech fortuna liga": "Czech First League",
    "czech chance national liga": "FNL",
    "denmark superliga": "Danish Superligaen",
    "denmark 1st division": "Danish 1st Division",
    "norway eliteserien": "Norwegian Eliteserien",
    "norway first division": "Norwegian 1. divisjon",
    "sweden allsvenskan": "Swedish Allsvenskan",
    "sweden superettan": "Swedish Superettan",
    "finland veikkausliiga": "Finnish Veikkausliiga",
    "latvia virsliga": "Latvian Virsliga",
    "usa mls": "US Major League Soccer",
    "usa mls next pro": "MLS Next Pro",
    "usa usl championship": "USL Championship",
    "usa usl league 1": "USL League One",
    "canada premier league": "Canadian Premier League",
}

def _league_identity(label: object) -> str:
    s = str(label or "").strip()
    s = re.sub(r"\s+(?:20\d{2}-\d{2}|20\d{2})$", "", s)
    return s.strip()

def _league_key(label: object) -> str:
    s = _league_identity(label).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

@st.cache_data(show_spinner=False, max_entries=2)
def load_opta_league_meta(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        out = pd.DataFrame(records)
        keep = [c for c in ["leagueId", "leagueName", "countryName", "globalRank", "seasonAverageRating", "lastWeekSeasonAverageRating"] if c in out.columns]
        out = out[keep].copy()
        out["seasonAverageRating"] = pd.to_numeric(out.get("seasonAverageRating"), errors="coerce")
        out["globalRank"] = pd.to_numeric(out.get("globalRank"), errors="coerce")
        out = out.dropna(subset=["leagueName", "seasonAverageRating"]).drop_duplicates("leagueId")
        out["_name_key"] = out["leagueName"].map(_league_key)
        out["_country_key"] = out.get("countryName", pd.Series(index=out.index, dtype=object)).map(_league_key)
        return out
    except Exception:
        return pd.DataFrame()

def match_wyscout_league_to_opta(wyscout_league: object, opta: pd.DataFrame) -> dict | None:
    if opta.empty:
        return None
    comp = _league_identity(wyscout_league)
    key = _league_key(comp)
    alias_name = OPTA_LEAGUE_ALIASES.get(key)
    if alias_name:
        exact = opta.loc[opta["leagueName"].str.casefold().eq(alias_name.casefold())]
        if not exact.empty:
            r = exact.iloc[0]
            return {"opta_name": r["leagueName"], "rating": float(r["seasonAverageRating"]), "rank": r.get("globalRank"), "confidence": 1.0}

    # Conservative fuzzy fallback. Country tokens are rewarded and a low score
    # is left unmatched rather than forcing a potentially wrong competition.
    first = key.split(" ")[0] if key else ""
    best = None
    for _, r in opta.iterrows():
        name_key = str(r.get("_name_key", ""))
        country_key = str(r.get("_country_key", ""))
        score = SequenceMatcher(None, key, name_key).ratio()
        if first and (first in name_key.split() or first in country_key.split()):
            score += 0.08
        # Also compare after removing the leading Wyscout country token.
        rest = " ".join(key.split(" ")[1:])
        if rest:
            score = max(score, SequenceMatcher(None, rest, name_key).ratio() + (0.05 if first in country_key.split() else 0.0))
        if best is None or score > best[0]:
            best = (score, r)
    if best is None or best[0] < 0.78:
        return None
    r = best[1]
    return {"opta_name": r["leagueName"], "rating": float(r["seasonAverageRating"]), "rank": r.get("globalRank"), "confidence": float(min(best[0], 0.99))}

@st.cache_data(show_spinner=False, max_entries=8)
def build_league_strength_lookup(league_labels: Tuple[str, ...], path_str: str) -> pd.DataFrame:
    opta = load_opta_league_meta(path_str)
    rows = []
    for league in league_labels:
        m = match_wyscout_league_to_opta(league, opta)
        if m:
            rows.append({"League": league, **m})
    return pd.DataFrame(rows)

def role_score_for_population(score_df: pd.DataFrame, benchmark_df: pd.DataFrame, role: str, coverage_threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    requested = list(PROFILES[role])
    resolved, _ = resolve_metrics_aliases(requested, benchmark_df.columns.tolist())
    if not resolved:
        return score_df.assign(_projection_role_score=np.nan), pd.Series(dtype=float)
    weights_pct = np.array(defaults_for_resolved(role, resolved), dtype=float)
    weights = normalize_weights(weights_pct)
    scored, _, _, _ = make_profile_score_vectorized(
        score_df, benchmark_df, resolved, weights, "_projection_role_score",
        coverage_threshold=coverage_threshold,
    )
    benchmark_scored, _, _, _ = make_profile_score_vectorized(
        benchmark_df, benchmark_df, resolved, weights, "_projection_role_score",
        coverage_threshold=coverage_threshold,
    )
    return scored, pd.to_numeric(benchmark_scored["_projection_role_score"], errors="coerce").dropna()



def build_projected_target_profile(
    player_row: pd.Series,
    target_benchmark: pd.DataFrame,
    role_name: str,
    rating_delta: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Project the player's role metrics against the actual target-league position population.

    V1 uses the player's unchanged raw Wyscout values, standardizes each metric against
    the target-league compatible-position population, then applies the empirically
    calibrated league-strength penalty to the direction-aware z-score. The adjusted
    z-score is converted to a 0-100 normal percentile for radar presentation.
    """
    requested = SINGLE_PLAYER_PROFILES[role_name]
    records, missing = [], []
    adjustment = LEAGUE_TRANSLATION_BETA * float(rating_delta)
    for group, requested_metric in requested:
        resolved, miss = resolve_metrics_aliases([requested_metric], target_benchmark.columns.tolist())
        if not resolved:
            missing.extend(miss or [requested_metric])
            continue
        metric = resolved[0]
        value = pd.to_numeric(pd.Series([player_row.get(metric, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        lower_better = metric in LOWER_IS_BETTER
        target_z = directional_z_against_population(target_benchmark[metric], value, lower_is_better=lower_better)
        if pd.isna(target_z):
            continue
        projected_z = float(target_z) - adjustment
        # Normal-CDF conversion avoids pretending that the translated raw value itself
        # was observed in the target league while retaining an intuitive percentile scale.
        projected_pct = 50.0 * (1.0 + math.erf(projected_z / math.sqrt(2.0)))
        records.append({
            "KPI Group": group, "Metric": metric, "Raw Value": float(value),
            "Percentile": float(np.clip(projected_pct, 0.0, 100.0)),
            "Z-score": projected_z,
        })
    return pd.DataFrame(records), sorted(set(missing))


def league_projection_radar(
    current_profile: pd.DataFrame,
    projected_profile: pd.DataFrame,
    player_name: str,
    current_label: str,
    target_label: str,
) -> go.Figure:
    """Single-Player-style scouting wheel overlaying current and projected percentiles."""
    if current_profile.empty or projected_profile.empty:
        return go.Figure()

    cur = current_profile[["KPI Group", "Metric", "Raw Value", "Percentile", "Z-score"]].copy()
    pro = projected_profile[["Metric", "Percentile", "Z-score"]].rename(
        columns={"Percentile": "Projected Percentile", "Z-score": "Projected Z-score"}
    )
    p = cur.merge(pro, on="Metric", how="inner").reset_index(drop=True)
    if p.empty:
        return go.Figure()

    groups = p["KPI Group"].drop_duplicates().tolist()
    palette = ["#3478B8", "#8E5AB5", "#16A6B6", "#E58A2B", "#244A73", "#D4A72C", "#C44E8B", "#687684"]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    display_aliases = {
        "Save rate, %": "Save Rate %", "Prevented goals per 90": "Goals Prevented /90",
        "Conceded goals per 90": "Goals Conceded /90 ↓", "Shots against per 90": "Shots Faced /90",
        "xG against per 90": "xGA /90", "Exits per 90": "Exits /90",
        "Aerial duels per 90.1": "Aerial Duels /90", "Aerial duels per 90": "Aerial Duels /90",
        "Aerial duels won, %": "Aerial Duel Win %", "Passes per 90": "Passes /90",
        "Accurate passes, %": "Pass Accuracy %", "Long passes per 90": "Long Passes /90",
        "Accurate long passes, %": "Long Pass Accuracy %", "Back passes received as GK per 90": "GK Back Passes /90",
        "Forward passes per 90": "Forward Passes /90", "Accurate forward passes, %": "Forward Pass Accuracy %",
        "Progressive passes per 90": "Progressive Passes /90", "Accurate progressive passes, %": "Progressive Pass Accuracy %",
        "Passes to final third per 90": "Final Third Passes /90", "Received passes per 90": "Passes Received /90",
        "Progressive runs per 90": "Progressive Runs /90", "Interceptions per 90": "Interceptions /90",
        "PAdj Interceptions": "PAdj Interceptions", "Successful defensive actions per 90": "Defensive Actions /90",
        "Defensive duels per 90": "Defensive Duels /90", "Defensive duels won, %": "Defensive Duel Win %",
        "PAdj Sliding tackles": "PAdj Sliding Tackles", "Sliding tackles per 90": "Sliding Tackles /90",
        "Shots blocked per 90": "Shots Blocked /90", "Fouls per 90": "Fouls /90 ↓",
        "Yellow cards per 90": "Yellow Cards /90 ↓", "Dribbles per 90": "Dribbles /90",
        "Successful dribbles, %": "Dribble Success %", "Accelerations per 90": "Accelerations /90",
        "Crosses per 90": "Crosses /90", "Accurate crosses, %": "Cross Accuracy %",
        "Crosses to goalie box per 90": "Box Crosses /90", "Passes to penalty area per 90": "Penalty Area Passes /90",
        "Shot assists per 90": "Shot Assists /90", "xA per 90": "xA /90", "xG per 90": "xG /90",
        "Shots per 90": "Shots /90", "Touches in box per 90": "Box Touches /90",
        "Non-penalty goals per 90": "Non-Penalty Goals /90", "Successful attacking actions per 90": "Attacking Actions /90",
        "Smart passes per 90": "Smart Passes /90", "Key passes per 90": "Key Passes /90",
        "Deep completions per 90": "Deep Completions /90", "Through passes per 90": "Through Passes /90",
        "Offensive duels per 90": "Offensive Duels /90", "Offensive duels won, %": "Offensive Duel Win %",
        "Goal conversion, %": "Goal Conversion %", "Shots on target, %": "Shots on Target %",
        "Head goals per 90": "Headed Goals /90", "Received long passes per 90": "Long Passes Received /90",
        "Fouls suffered per 90": "Fouls Won /90",
    }
    labels = [display_aliases.get(m, m) for m in p["Metric"]]
    p["Display Metric"] = labels

    n = len(p)
    group_seq = p["KPI Group"].tolist()
    gap_units = 0.48
    boundaries = sum(1 for i in range(1, n) if group_seq[i] != group_seq[i - 1])
    unit = 360.0 / (n + boundaries * gap_units)
    theta, cursor = [], 0.0
    for i in range(n):
        if i > 0 and group_seq[i] != group_seq[i - 1]:
            cursor += gap_units * unit
        theta.append(cursor)
        cursor += unit
    theta = np.array(theta, dtype=float)
    width = unit * 0.90

    band_inner, band_outer = 79.0, 89.0
    fig = go.Figure()
    for group in groups:
        idx = p.index[p["KPI Group"] == group].tolist()
        fig.add_trace(go.Barpolar(
            r=[band_outer-band_inner]*len(idx), base=[band_inner]*len(idx),
            theta=[float(theta[i]) for i in idx], width=[width]*len(idx),
            marker_color=group_colors[group], marker_line_color="white", marker_line_width=1.5,
            opacity=0.90, name=group, hoverinfo="skip", legendgroup="kpi",
        ))

    perf_inner, perf_outer = 20.0, 74.0
    perf_span = perf_outer - perf_inner
    cur_pct = np.clip(p["Percentile"].to_numpy(dtype=float), 0, 100)
    pro_pct = np.clip(p["Projected Percentile"].to_numpy(dtype=float), 0, 100)
    cur_r = perf_inner + cur_pct/100.0*perf_span
    pro_r = perf_inner + pro_pct/100.0*perf_span

    ring_theta = np.linspace(0, 360, 361)
    for tick in [0,25,50,75,100]:
        rv = perf_inner + tick/100.0*perf_span
        if tick == 50:
            col, lw = "rgba(45,55,65,0.55)", 2.6
        elif tick in (25,75):
            col, lw = "rgba(95,105,115,0.14)", 1.15
        else:
            col, lw = "rgba(120,130,140,0.10)", 1.0
        fig.add_trace(go.Scatterpolar(r=[rv]*len(ring_theta), theta=ring_theta, mode="lines",
                                      line=dict(color=col,width=lw), hoverinfo="skip", showlegend=False))

    # Close each profile around the wheel while preserving the KPI-boundary gaps.
    closed_theta = list(theta) + [float(theta[0])]
    current_color = "#303A44"
    projected_color = "#E56B3F"
    cur_custom = np.column_stack([p["Raw Value"], p["Percentile"], p["Z-score"], p["Display Metric"], p["KPI Group"]])
    pro_custom = np.column_stack([p["Raw Value"], p["Projected Percentile"], p["Projected Z-score"], p["Display Metric"], p["KPI Group"]])
    cur_custom_closed = np.vstack([cur_custom, cur_custom[0]])
    pro_custom_closed = np.vstack([pro_custom, pro_custom[0]])

    fig.add_trace(go.Scatterpolar(
        r=list(cur_r)+[cur_r[0]], theta=closed_theta, mode="lines+markers",
        line=dict(color=current_color,width=3.8), marker=dict(size=10,color=current_color,line=dict(color="white",width=1.7)),
        name=current_label, legendgroup="profiles", showlegend=False, customdata=cur_custom_closed,
        hovertemplate="<b>%{customdata[3]}</b><br>Current: %{customdata[1]:.0f}th percentile<br>Raw value: %{customdata[0]:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=list(pro_r)+[pro_r[0]], theta=closed_theta, mode="lines+markers",
        line=dict(color=projected_color,width=3.8,dash="dot"), marker=dict(size=10,color=projected_color,line=dict(color="white",width=1.7)),
        name=f"{target_label} (Projected)", legendgroup="profiles", showlegend=False, customdata=pro_custom_closed,
        hovertemplate="<b>%{customdata[3]}</b><br>Projected: %{customdata[1]:.0f}th percentile<br>Current raw value: %{customdata[0]:.2f}<extra></extra>",
    ))

    # Outer tiles show current → projected percentile, making the translation readable without hover.
    tile_text = [f"<b>{int(round(c))}→{int(round(q))}</b>" for c,q in zip(cur_pct,pro_pct)]
    fig.add_trace(go.Scatterpolar(r=[84.0]*n, theta=theta, mode="text", text=tile_text,
                                  textfont=dict(size=11,color="white"), hoverinfo="skip", showlegend=False))

    # Compact profile legend below the subtitle, separate from the KPI-category legend.
    fig.add_annotation(x=0.39, y=1.045, xref="paper", yref="paper",
                       text=f"<span style='color:{current_color}'>━━●</span> {current_label}",
                       showarrow=False, xanchor="right", font=dict(size=11, color="#303A44"))
    fig.add_annotation(x=0.61, y=1.045, xref="paper", yref="paper",
                       text=f"<span style='color:{projected_color}'>┈┈●</span> {target_label} (Projected)",
                       showarrow=False, xanchor="left", font=dict(size=11, color="#303A44"))

    fig.add_annotation(x=0.5,y=0.5,xref="paper",yref="paper",
                       text=f"<b>{player_name}</b><br><span style='font-size:11px'>Current → Projected</span>",
                       showarrow=False,align="center",font=dict(size=16,color="#17202A"),
                       bgcolor="rgba(255,255,255,0.94)",bordercolor="rgba(120,130,140,0.22)",borderwidth=1,borderpad=11)
    fig.add_annotation(x=0.01,y=0.01,xref="paper",yref="paper",text="Percentile: 0–100 · outer tiles = current → projected",
                       showarrow=False,xanchor="left",yanchor="bottom",font=dict(size=10,color="#6B7785"))

    fig.update_layout(
        title=dict(text=f"<b>Projected performance of {player_name}</b><br><sup>{current_label} → {target_label}</sup>",
                   x=0.5,xanchor="center",y=0.985,font=dict(size=20,color="#17202A")),
        template="plotly_white",height=920,margin=dict(l=180,r=180,t=150,b=125),paper_bgcolor="white",plot_bgcolor="white",
        legend=dict(orientation="v", yanchor="top", y=1.015, xanchor="left", x=0.08, title_text="KPI groups", font=dict(size=10), bgcolor="rgba(255,255,255,0.72)"),
        polar=dict(bgcolor="white",radialaxis=dict(range=[0,99],visible=False),
                   angularaxis=dict(tickmode="array",tickvals=theta,ticktext=labels,direction="clockwise",rotation=90,
                                    gridcolor="rgba(255,255,255,0)",tickfont=dict(size=16,color="#303A44"),showline=False)),
        barmode="overlay",
    )
    return fig

# =========================
# Role Fit v1
# =========================
ROLE_FIT_POSITION_MAP: Dict[str, List[str]] = {
    "GK": ["Classic Goalkeeper", "Sweeper Keeper", "Build-Up Keeper"],
    "CB": ["Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle Pin CB", "Wide CB (in 3)"],
    "DM": ["Defensive Midfielder #6", "Deep-Lying Playmaker", "Box-to-Box Midfielder", "Playmaker #10"],
    "CM": ["Defensive Midfielder #6", "Deep-Lying Playmaker", "Box-to-Box Midfielder", "Playmaker #10"],
    "AM": ["Deep-Lying Playmaker", "Box-to-Box Midfielder", "Playmaker #10"],
    "LB": ["Full-Back", "Wing-Back", "Inverted Full-Back"],
    "RB": ["Full-Back", "Wing-Back", "Inverted Full-Back"],
    "LWB": ["Full-Back", "Wing-Back", "Inverted Full-Back"],
    "RWB": ["Full-Back", "Wing-Back", "Inverted Full-Back"],
    "LW": ["Classic Winger", "Inverted Winger", "Wide Forward / Inside 9"],
    "RW": ["Classic Winger", "Inverted Winger", "Wide Forward / Inside 9"],
    "LWF": ["Classic Winger", "Inverted Winger", "Wide Forward / Inside 9"],
    "RWF": ["Classic Winger", "Inverted Winger", "Wide Forward / Inside 9"],
    "CF": ["Target Man #9", "Poacher", "Pressing Forward", "Creative Forward / False 9", "Wide Forward / Inside 9"],
    "ST": ["Target Man #9", "Poacher", "Pressing Forward", "Creative Forward / False 9", "Wide Forward / Inside 9"],
}


def normalize_main_position_for_role_fit(value: object) -> str:
    """Normalize Wyscout-style position labels to a Role Fit family."""
    s = str(value or "").strip().upper()
    if not s:
        return ""
    # Most specific wide/keeper/forward labels first.
    aliases = [
        ("GOALKEEPER", "GK"), ("GK", "GK"),
        ("RWB", "RWB"), ("LWB", "LWB"),
        ("RIGHT WINGBACK", "RWB"), ("LEFT WINGBACK", "LWB"),
        ("RIGHT BACK", "RB"), ("LEFT BACK", "LB"),
        ("RB", "RB"), ("LB", "LB"),
        ("CENTRE BACK", "CB"), ("CENTER BACK", "CB"), ("CB", "CB"),
        ("DEFENSIVE MID", "DM"), ("DMF", "DM"), ("CDM", "DM"), ("DM", "DM"),
        ("ATTACKING MID", "AM"), ("AMF", "AM"), ("CAM", "AM"), ("AM", "AM"),
        ("CENTRAL MID", "CM"), ("CMF", "CM"), ("CM", "CM"),
        ("RIGHT WINGER", "RW"), ("LEFT WINGER", "LW"),
        ("RWF", "RWF"), ("LWF", "LWF"), ("RW", "RW"), ("LW", "LW"),
        ("CENTRE FORWARD", "CF"), ("CENTER FORWARD", "CF"), ("CF", "CF"),
        ("STRIKER", "ST"), ("ST", "ST"),
    ]
    for token, family in aliases:
        if token in s:
            return family
    return s


def compatible_roles_for_position(value: object) -> List[str]:
    return ROLE_FIT_POSITION_MAP.get(normalize_main_position_for_role_fit(value), [])


@st.cache_data(show_spinner=False, max_entries=96)
def role_fit_for_player(
    player_row: pd.Series,
    role_name: str,
    benchmark_df: pd.DataFrame,
    coverage_threshold: float,
) -> Dict[str, object]:
    """Evaluate one player under one canonical built-in role."""
    requested = PROFILES[role_name]
    resolved, missing = resolve_metrics_aliases(requested, benchmark_df.columns.tolist())
    if not resolved:
        return {
            "Role": role_name, "Role Score": np.nan, "Role Percentile": np.nan,
            "Coverage %": 0.0, "Benchmark N": len(benchmark_df), "KPI Scores": {},
            "Resolved Metrics": [], "Missing Metrics": missing,
        }

    requested_to_resolved: Dict[str, str] = {}
    resolved_weights: List[float] = []
    for req in requested:
        rr, _ = resolve_metrics_aliases([req], benchmark_df.columns.tolist())
        if rr:
            requested_to_resolved[req] = rr[0]
            resolved_weights.append(float(DEFAULT_WEIGHTS[role_name][req]))

    resolved = [requested_to_resolved[r] for r in requested if r in requested_to_resolved]
    # Deduplicate while retaining canonical order.
    resolved = list(dict.fromkeys(resolved))
    weight_by_resolved = {}
    for req, col in requested_to_resolved.items():
        weight_by_resolved[col] = weight_by_resolved.get(col, 0.0) + float(DEFAULT_WEIGHTS[role_name][req])
    weights = np.array([weight_by_resolved[m] for m in resolved], dtype=float)

    B = benchmark_df[resolved]
    package = build_benchmark_scoring_package(
        B,
        tuple(resolved),
        tuple(float(weight_by_resolved[m]) for m in resolved),
        float(coverage_threshold),
    )
    usable = list(package["usable"])
    if not usable:
        return {
            "Role": role_name, "Role Score": np.nan, "Role Percentile": np.nan,
            "Coverage %": 0.0, "Benchmark N": len(benchmark_df), "KPI Scores": {},
            "Resolved Metrics": resolved, "Missing Metrics": missing,
        }

    W = package["weights"]
    means = package["means"]
    stds = package["stds"]
    Zb = package["benchmark_z"]
    score_b = package["benchmark_scores"]

    player_vals = pd.Series(
        {m: pd.to_numeric(pd.Series([player_row.get(m, np.nan)]), errors="coerce").iloc[0] for m in usable},
        dtype=float,
    )
    z = (player_vals - means) / stds
    flip = [m for m in usable if m in LOWER_IS_BETTER]
    if flip:
        z[flip] = -z[flip]

    available_weight = float(W[z.notna()].sum())
    coverage = available_weight * 100.0
    score = float((z.fillna(0.0) * W).sum() / available_weight) if available_weight > 0 else np.nan
    if available_weight < coverage_threshold:
        score = np.nan

    role_pct = percentile_rank_against_population(
        score_b, score, lower_is_better=False
    ) if pd.notna(score) else np.nan

    # KPI decomposition for this player.
    kpi_scores: Dict[str, float] = {}
    for kpi, metric_weights in PROFILE_CONFIG[role_name].items():
        cols = []
        kws = []
        for req, default_w in metric_weights.items():
            col = requested_to_resolved.get(req)
            if col in usable:
                cols.append(col)
                kws.append(float(default_w))
        if not cols:
            continue
        kz = z[cols]
        kw = pd.Series(kws, index=cols, dtype=float)
        kavail = float(kw[kz.notna()].sum())
        if kavail > 0:
            kpi_scores[kpi] = float((kz.fillna(0.0) * kw).sum() / kavail)

    return {
        "Role": role_name,
        "Role Score": score,
        "Role Percentile": role_pct,
        "Coverage %": coverage,
        "Benchmark N": int(score_b.notna().sum()),
        "KPI Scores": kpi_scores,
        "Resolved Metrics": usable,
        "Missing Metrics": missing,
        "Player Z": z,
        "Benchmark Z": Zb,
        "Benchmark Raw": B,
    }


def single_metric_display_label(metric: str) -> str:
    """Compact display label used by Role Fit detail wheels."""
    aliases = {
        "Save rate, %": "Save Rate %",
        "Prevented goals per 90": "Goals Prevented /90",
        "Conceded goals per 90": "Goals Conceded /90 ↓",
        "Shots against per 90": "Shots Faced /90",
        "xG against per 90": "xGA /90",
        "Exits per 90": "Exits /90",
        "Aerial duels per 90.1": "Aerial Duels /90",
        "Aerial duels per 90": "Aerial Duels /90",
        "Aerial duels won, %": "Aerial Duel Win %",
        "Passes per 90": "Passes /90",
        "Accurate passes, %": "Pass Accuracy %",
        "Long passes per 90": "Long Passes /90",
        "Accurate long passes, %": "Long Pass Accuracy %",
        "Back passes received as GK per 90": "GK Back Passes /90",
        "Forward passes per 90": "Forward Passes /90",
        "Accurate forward passes, %": "Forward Pass Accuracy %",
        "Progressive passes per 90": "Progressive Passes /90",
        "Accurate progressive passes, %": "Progressive Pass Accuracy %",
        "Passes to final third per 90": "Final Third Passes /90",
        "Received passes per 90": "Passes Received /90",
        "Progressive runs per 90": "Progressive Runs /90",
        "Interceptions per 90": "Interceptions /90",
        "PAdj Interceptions": "PAdj Interceptions",
        "Successful defensive actions per 90": "Defensive Actions /90",
        "Defensive duels per 90": "Defensive Duels /90",
        "Defensive duels won, %": "Defensive Duel Win %",
        "PAdj Sliding tackles": "PAdj Sliding Tackles",
        "Sliding tackles per 90": "Sliding Tackles /90",
        "Shots blocked per 90": "Shots Blocked /90",
        "Fouls per 90": "Fouls /90 ↓",
        "Yellow cards per 90": "Yellow Cards /90 ↓",
        "Dribbles per 90": "Dribbles /90",
        "Successful dribbles, %": "Dribble Success %",
        "Accelerations per 90": "Accelerations /90",
        "Crosses per 90": "Crosses /90",
        "Accurate crosses, %": "Cross Accuracy %",
        "Crosses to goalie box per 90": "Box Crosses /90",
        "Passes to penalty area per 90": "Penalty Area Passes /90",
        "Shot assists per 90": "Shot Assists /90",
        "xA per 90": "xA /90",
        "xG per 90": "xG /90",
        "Shots per 90": "Shots /90",
        "Touches in box per 90": "Box Touches /90",
        "Non-penalty goals per 90": "Non-Penalty Goals /90",
        "Successful attacking actions per 90": "Attacking Actions /90",
        "Smart passes per 90": "Smart Passes /90",
        "Key passes per 90": "Key Passes /90",
        "Deep completions per 90": "Deep Completions /90",
        "Through passes per 90": "Through Passes /90",
        "Offensive duels per 90": "Offensive Duels /90",
        "Offensive duels won, %": "Offensive Duel Win %",
        "Goal conversion, %": "Goal Conversion %",
        "Shots on target, %": "Shots on Target %",
        "Head goals per 90": "Headed Goals /90",
        "Received long passes per 90": "Long Passes Received /90",
        "Fouls suffered per 90": "Fouls Won /90",
    }
    return aliases.get(metric, metric)


def role_fit_detail_wheel(
    player_name: str,
    role_name: str,
    fit_result: Dict[str, object],
    player_row: pd.Series,
    display_scale: str,
) -> go.Figure:
    """
    Role Fit detail uses the canonical Single Player wheel renderer.

    This keeps metric order, KPI grouping, category colours, angular gaps,
    reference rings, performance colours, typography, centre panel and hover
    behaviour identical across Single Player and Role Fit.
    """
    metrics = list(fit_result.get("Resolved Metrics", []))
    if not metrics:
        return go.Figure()

    benchmark_raw = fit_result.get("Benchmark Raw", pd.DataFrame())
    player_z = fit_result.get("Player Z", pd.Series(dtype=float))

    # Resolve the canonical role KPI order against the actual usable metrics.
    resolved_to_kpi: Dict[str, str] = {}
    canonical_order: List[str] = []
    for kpi, requested_metric in SINGLE_PLAYER_PROFILES[role_name]:
        rr, _ = resolve_metrics_aliases([requested_metric], list(player_row.index))
        if not rr:
            continue
        metric = rr[0]
        if metric in metrics and metric not in canonical_order:
            canonical_order.append(metric)
            resolved_to_kpi[metric] = kpi

    # Retain any usable metric that was not represented above, while keeping
    # canonical profile metrics first.
    for metric in metrics:
        if metric not in canonical_order:
            canonical_order.append(metric)
            resolved_to_kpi.setdefault(metric, "Profile")

    rows = []
    for metric in canonical_order:
        raw_value = pd.to_numeric(
            pd.Series([player_row.get(metric, np.nan)]), errors="coerce"
        ).iloc[0]
        z_value = pd.to_numeric(
            pd.Series([player_z.get(metric, np.nan)]), errors="coerce"
        ).iloc[0]

        pct_value = np.nan
        if isinstance(benchmark_raw, pd.DataFrame) and metric in benchmark_raw.columns:
            pct_value = percentile_rank_against_population(
                benchmark_raw[metric],
                raw_value,
                lower_is_better=(metric in LOWER_IS_BETTER),
            )

        rows.append({
            "KPI Group": resolved_to_kpi.get(metric, "Profile"),
            "Metric": metric,
            "Raw Value": raw_value,
            "Percentile": pct_value,
            "Z-score": z_value,
        })

    profile_df = pd.DataFrame(rows)
    if profile_df.empty:
        return go.Figure()

    # Match the Single Player subtitle structure while retaining Role Fit context.
    subtitle = f"{role_name} | Role Fit benchmark"
    return single_player_wheel(
        profile_df=profile_df,
        player_name=player_name,
        subtitle=subtitle,
        scale_mode=display_scale,
    )


# =========================
# Upload
# =========================
st.sidebar.header("Upload")
# =========================
# Workspace navigation
# =========================
APP_PAGES = ["Players", "Single Player", "Multi Player", "Role Fit", "League Projection"]
app_page = st.sidebar.radio("Workspace", APP_PAGES, index=0, key="app_workspace")
st.sidebar.markdown("---")

uploaded = st.sidebar.file_uploader("Upload your Football Data CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload your football data CSV (e.g., Wyscout export).")
    st.stop()

file_bytes = uploaded.getvalue()

# Stability guard: fail cleanly before large allocations can destabilize a
# constrained Streamlit deployment.
MAX_UPLOAD_MB = 200
_upload_mb = len(file_bytes) / (1024 * 1024)
if not file_bytes:
    st.error("The uploaded CSV file is empty.")
    st.stop()
if _upload_mb > MAX_UPLOAD_MB:
    st.error(
        f"The uploaded CSV is {_upload_mb:.1f} MB. "
        f"For stability, this app accepts files up to {MAX_UPLOAD_MB} MB."
    )
    st.stop()

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

try:
    df_all = preprocess(df_raw)
except Exception as e:
    st.error(
        "The CSV was read, but preprocessing failed. "
        "Check that the Wyscout export has consistent columns and a valid League field."
    )
    with st.expander("Technical details"):
        st.code(str(e))
    st.stop()

if df_all.empty:
    st.error("The uploaded CSV contains no usable player rows after preprocessing.")
    st.stop()

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

# Season — prefer 2026 on a fresh upload. If 2026 is not present, fall back to
# the latest available season group so the app never opens on an empty season.
season_groups = sorted([x for x in df_all["Season group"].dropna().unique().tolist()])
season_default = [2026] if 2026 in season_groups else ([season_groups[-1]] if season_groups else [])
selected_groups = st.sidebar.multiselect(
    "Season group (start year)",
    season_groups,
    default=season_default,
)

use_exact_label = st.sidebar.checkbox("Also filter by exact Season label", value=False)
selected_labels = None
if use_exact_label:
    season_labels = sorted(df_all["Season label"].dropna().unique().tolist())
    selected_labels = st.sidebar.multiselect("Season label(s)", season_labels, default=season_labels)

season_mask = (
    df_all["Season group"].isin(selected_groups)
    if selected_groups
    else pd.Series(False, index=df_all.index)
)
if use_exact_label and selected_labels:
    season_mask = season_mask & df_all["Season label"].isin(selected_labels)

df_season = df_all.loc[season_mask]

# League — deliberately blank by default. The normal workflow is to choose a
# small group of leagues manually; ALL remains available as an explicit choice.
leagues = sorted(df_season["League"].dropna().unique().tolist())
selected_leagues, _ = multiselect_all(
    "League(s)",
    leagues,
    default_all=False,
    help="Choose the leagues you want to analyse. Select ALL manually when needed.",
)

if not selected_leagues:
    st.sidebar.info("Select at least one league to continue.")
    st.info("Select at least one league in the sidebar to continue.")
    st.stop()

df_league = df_season.loc[df_season["League"].isin(selected_leagues)]
if df_league.empty:
    st.warning("No players found for selected seasons/leagues.")
    st.stop()

# Team / Position — unchanged. After leagues are chosen, Team defaults to ALL.
teams = sorted(df_league["Team"].dropna().unique().tolist())
positions = sorted(df_league["Main Position"].dropna().unique().tolist())
selected_teams, _ = multiselect_all("Team(s)", teams, default_all=True)
# A stable widget key allows Apply profile to set role-eligible Main Positions.
# Apply any queued profile-driven selection BEFORE the widget is instantiated.
_position_widget_options = [ALL_TOKEN] + positions

_pending_positions = st.session_state.pop("_pending_main_position_filter", None)
if _pending_positions is not None:
    _pending_clean = [p for p in _pending_positions if p in positions]
    st.session_state["main_position_filter"] = _pending_clean or [ALL_TOKEN]
elif "main_position_filter" in st.session_state:
    # Remove stale values if season/league changes make positions unavailable.
    _clean_positions = [
        p for p in st.session_state["main_position_filter"]
        if p in _position_widget_options
    ]
    st.session_state["main_position_filter"] = _clean_positions or [ALL_TOKEN]

selected_positions, _ = multiselect_all(
    "Main Position(s)",
    positions,
    default_all=True,
    key="main_position_filter",
)

# Age — quick scouting presets plus the existing manual range slider.
age_series = df_league["Age"]
_age_valid = age_series.dropna()
if _age_valid.empty:
    age_min, age_max = 0, 100
else:
    age_min = int(_age_valid.min())
    age_max = int(_age_valid.max())
if age_max < age_min:
    age_min, age_max = 0, 100

age_preset = st.sidebar.radio(
    "Age preset",
    ["All", "U21", "U23", "U25"],
    index=0,
    horizontal=True,
    key="age_preset",
)

age_caps = {"U21": 21, "U23": 23, "U25": 25}
preset_age_max = min(age_caps.get(age_preset, age_max), age_max)
preset_age_max = max(age_min, preset_age_max)

# Reset the manual range when the preset changes; otherwise retain manual slider edits.
if st.session_state.get("_last_age_preset") != age_preset:
    st.session_state["age_range"] = (age_min, preset_age_max)
    st.session_state["_last_age_preset"] = age_preset

# Keep session-state range valid if league/season changes alter available ages.
current_age_range = st.session_state.get("age_range", (age_min, preset_age_max))
valid_age_low = max(age_min, min(int(current_age_range[0]), age_max))
valid_age_high = max(valid_age_low, min(int(current_age_range[1]), age_max))
st.session_state["age_range"] = (valid_age_low, valid_age_high)

age_range = st.sidebar.slider(
    "Age range",
    age_min,
    age_max,
    key="age_range",
) if age_max >= age_min else (0, 0)

# Minutes — quick presets plus the existing slider. Default remains 500.
min_minutes = 0
if "Minutes played" in df_league.columns:
    mm = df_league["Minutes played"]
    _mm_max = mm.max(skipna=True)
    min_minutes_max = int(_mm_max) if pd.notna(_mm_max) else 0

    minute_preset = st.sidebar.radio(
        "Minutes preset",
        [0, 300, 500, 900, 1500],
        index=2,
        horizontal=True,
        key="minutes_preset",
        format_func=lambda x: f"{x}",
    )

    preset_minutes = min(int(minute_preset), max(0, min_minutes_max))
    if st.session_state.get("_last_minutes_preset") != minute_preset:
        st.session_state["min_minutes_slider"] = preset_minutes
        st.session_state["_last_minutes_preset"] = minute_preset

    current_minutes = int(st.session_state.get("min_minutes_slider", preset_minutes))
    st.session_state["min_minutes_slider"] = max(0, min(current_minutes, max(0, min_minutes_max)))

    min_minutes = st.sidebar.slider(
        "Minimum minutes this season",
        0,
        max(0, min_minutes_max),
        key="min_minutes_slider",
    )

# Plot-only option moved out of the primary scouting filters.
with st.sidebar.expander("Advanced / Plot settings", expanded=False):
    remove_outliers = st.checkbox(
        "Remove outliers (|Z| > 3) — for plots only",
        value=False,
        key="remove_plot_outliers",
    )

mask = (
    df_league["Team"].isin(selected_teams)
    & df_league["Main Position"].isin(selected_positions)
    & df_league["Age"].between(age_range[0], age_range[1])
)
if "Minutes played" in df_league.columns:
    mask = mask & (df_league["Minutes played"] >= min_minutes)

filtered_base = df_league.loc[mask]
st.sidebar.markdown(f"**Players matching filters: {len(filtered_base)}**")
if filtered_base.empty:
    st.warning("No players match the selected filters.")
    st.stop()

numeric_cols_base = get_numeric_columns(filtered_base)


# =========================
# Benchmark population
# =========================
st.sidebar.header("Benchmark population")
benchmark_mode = st.sidebar.radio(
    "Benchmark basis",
    ["Role position", "Selected positions", "All positions"],
    index=0,
    help=(
        "Benchmarking is independent from team and age search filters. "
        "Role position is recommended for built-in profiles."
    ),
)
min_benchmark_n = st.sidebar.number_input(
    "Minimum benchmark sample",
    min_value=10,
    max_value=100,
    value=20,
    step=5,
    help="The app warns when the role benchmark falls below this sample size.",
)
coverage_threshold_pct = st.sidebar.slider(
    "Minimum Profile Score coverage",
    min_value=50,
    max_value=100,
    value=70,
    step=5,
    help="Scores below this percentage of available profile weight are suppressed rather than treating missing data as average.",
)

benchmark_source_global = df_league
if "Minutes played" in benchmark_source_global.columns:
    benchmark_source_global = benchmark_source_global.loc[
        benchmark_source_global["Minutes played"] >= min_minutes
    ]

if benchmark_source_global.empty:
    st.warning(
        "The current league/season/minutes selection leaves no benchmark population. "
        "Lower the minutes threshold or broaden the selected leagues."
    )
    st.stop()

# =========================
# Profile builder
# =========================
with st.sidebar.expander("⚙️ Advanced profile settings", expanded=False):
    st.header("Player profiles (z-score)")
    if BUILTIN_PROFILE_FRAMEWORK_ISSUES:
        st.error("Built-in profile configuration error: " + " | ".join(BUILTIN_PROFILE_FRAMEWORK_ISSUES))

    if st.button("Clear active profile"):
        st.session_state["active_profile"] = None
        st.rerun()

    mode = st.radio("Profile mode", ["Built-in", "Custom"], index=0, horizontal=True)

    if mode == "Built-in":
        profile_name = st.selectbox("Choose profile", list(PROFILES.keys()))
        requested_metrics = PROFILES[profile_name]
        resolved_metrics, missing_names = resolve_metrics_aliases(requested_metrics, filtered_base.columns.tolist())

        preset_key = f"preset::{profile_name}"
        defaults = defaults_for_resolved(profile_name, resolved_metrics)
        if preset_key not in st.session_state:
            st.session_state[preset_key] = {"metrics": resolved_metrics[:], "weights": defaults[:]}

        state = st.session_state[preset_key]
        if state.get("metrics") != resolved_metrics:
            old_map = {m: int(w) for m, w in zip(state.get("metrics", []), state.get("weights", []))}
            state = {
                "metrics": resolved_metrics[:],
                "weights": [int(old_map.get(m, d)) for m, d in zip(resolved_metrics, defaults)],
            }
            st.session_state[preset_key] = state

        if st.button("Reset weights to defaults", key=f"reset::{profile_name}"):
            st.session_state[preset_key] = {"metrics": resolved_metrics[:], "weights": defaults[:]}
            for m, d in zip(resolved_metrics, defaults):
                st.session_state[safe_widget_key("w", profile_name, m)] = int(d)
            st.rerun()

        weights_pct = []
        resolved_to_requested = {}
        for req in requested_metrics:
            rr, _ = resolve_metrics_aliases([req], filtered_base.columns.tolist())
            if rr:
                resolved_to_requested[rr[0]] = req

        for kpi, metric_weights in PROFILE_CONFIG[profile_name].items():
            st.markdown(f"**{kpi}**")
            for requested_metric in metric_weights:
                rr, _ = resolve_metrics_aliases([requested_metric], filtered_base.columns.tolist())
                if not rr:
                    continue
                m = rr[0]
                default_w = int(DEFAULT_WEIGHTS[profile_name][requested_metric])
                slider_key = safe_widget_key("w", profile_name, m)
                if slider_key not in st.session_state:
                    st.session_state[slider_key] = default_w
                w = st.slider(f"{m}", min_value=0, max_value=100, step=1, key=slider_key)
                weights_pct.append(int(w))

        total_weight = int(sum(weights_pct))
        st.metric("Total profile weight", f"{total_weight}%")
        if total_weight != 100:
            st.warning("Weights must total exactly 100% before the profile can be applied.")

        if missing_names:
            st.caption("Unavailable metrics: " + ", ".join(missing_names))

        if st.button(
            "Apply profile",
            key=f"apply::{profile_name}",
            disabled=(not resolved_metrics or total_weight != 100),
            type="primary",
        ):
            st.session_state["active_profile"] = {
                "calc_col": f"Score: {profile_name}",
                "profile_name": profile_name,
                "metrics": resolved_metrics[:],
                "weights_pct": [int(x) for x in weights_pct],
            }

            # Automatically focus the Main Position filter on players eligible
            # for the applied built-in profile. The user can edit it afterward.
            eligible_positions = PROFILE_POSITION_ELIGIBILITY.get(profile_name, [])
            available_eligible_positions = [p for p in eligible_positions if p in positions]
            if available_eligible_positions:
                # The Main Position widget has already been instantiated on this
                # Streamlit run, so do not mutate its key here. Queue the change
                # and apply it before widget creation on the next rerun.
                st.session_state["_pending_main_position_filter"] = available_eligible_positions

            # Synchronize role selectors across workspaces on the next rerun.
            # Users may still manually change a workspace role afterward.
            st.session_state["_pending_workspace_profile_sync"] = profile_name

            st.rerun()

    else:
        custom_name = st.text_input("Profile name", value="Custom Profile").strip() or "Custom Profile"
        custom_metrics = st.multiselect(
            "Pick metrics to include",
            options=numeric_cols_base,
            default=numeric_cols_base[:5],
        )
        weights_pct = []
        if custom_metrics:
            default_pct = max(1, int(100 / len(custom_metrics)))
            for m in custom_metrics:
                weights_pct.append(
                    int(st.slider(
                        m, 0, 100, default_pct, 1,
                        key=safe_widget_key("w_custom", custom_name, m),
                    ))
                )
        custom_total = int(sum(weights_pct))
        st.metric("Total profile weight", f"{custom_total}%")
        if custom_metrics and custom_total != 100:
            st.warning("Weights must total exactly 100% before the profile can be applied.")

        if st.button(
            "Apply custom profile",
            disabled=(not custom_metrics or custom_total != 100),
            type="primary",
        ):
            st.session_state["active_profile"] = {
                "calc_col": f"Score: {custom_name}",
                "profile_name": None,
                "metrics": custom_metrics[:],
                "weights_pct": [int(x) for x in weights_pct],
            }
            st.rerun()


# =========================
# Apply active profile on EVERY rerun (fixes disappearing score)
# =========================
active = st.session_state.get("active_profile")
calc_col_name: str | None = None
profile_metrics_in_use: List[str] = []
active_benchmark_df = benchmark_source_global
active_kpi_cols: List[str] = []

if active and active.get("metrics"):
    calc_col_name = str(active.get("calc_col", "Score"))
    active_role = active.get("profile_name")
    profile_metrics_in_use = list(active.get("metrics", []))
    weights_pct_arr = np.array(active.get("weights_pct", []), dtype=float)

    if len(weights_pct_arr) != len(profile_metrics_in_use) or len(profile_metrics_in_use) == 0:
        weights = np.ones(len(profile_metrics_in_use), dtype=float) / max(1, len(profile_metrics_in_use))
    else:
        weights = weights_pct_arr / 100.0

    active_benchmark_df = benchmark_population_for_role(
        benchmark_source_global,
        active_role,
        benchmark_mode,
        selected_positions,
    )

    if len(active_benchmark_df) < int(min_benchmark_n):
        st.warning(
            f"Benchmark sample is small ({len(active_benchmark_df)} players; recommended minimum {int(min_benchmark_n)}). "
            "Scores and percentiles may be unstable. Consider broadening the benchmark."
        )

    filtered, usable_metrics, skipped_sparse, score_z = make_profile_score_vectorized(
        filtered_base,
        active_benchmark_df,
        profile_metrics_in_use,
        weights,
        calc_col_name,
        coverage_threshold=float(coverage_threshold_pct) / 100.0,
    )

    coverage_col = f"{calc_col_name} Coverage %"

    # KPI sub-scores for built-in profiles.
    if active_role in PROFILE_CONFIG and not score_z.empty:
        requested_to_resolved = {}
        for requested_metric in PROFILES[active_role]:
            rr, _ = resolve_metrics_aliases([requested_metric], filtered.columns.tolist())
            if rr:
                requested_to_resolved[requested_metric] = rr[0]
        custom_weight_map = {
            m: float(w) for m, w in zip(profile_metrics_in_use, weights_pct_arr)
        }
        kpi_df = role_kpi_subscores(
            score_z,
            active_role,
            requested_to_resolved,
            custom_weights_pct=custom_weight_map,
        )
        for c in kpi_df.columns:
            filtered[c] = kpi_df[c]
            active_kpi_cols.append(c)

    if skipped_sparse:
        st.caption("⚠️ Skipped sparse benchmark metrics (≥95% zeros/NaNs): " + ", ".join(skipped_sparse))

    st.caption(
        f"Benchmark: {benchmark_mode} · selected leagues · {min_minutes}+ min · "
        f"N={len(active_benchmark_df):,} · score coverage threshold={coverage_threshold_pct}%"
    )
else:
    filtered = filtered_base

# Calculated Metrics v1.
# Step 3: enrich only the Players workspace. Compare / Single Player / Multi Player /
# Role Fit / League Projection do not use Calc: columns, so they should not pay the cost of building the
# 14 positional composites and 3 efficiency metrics on every widget rerun.
CALCULATED_OUTPUT_COLUMNS = CALCULATED_INDEX_NAMES + EFFICIENCY_METRICS
calc_metric_cols = []
current_families = []
expected_family_indices = []
populated_family_indices = []

if app_page == "Players":
    # Benchmark remains independent from team/age filters and follows the selected
    # season/league/minutes universe. Step 2 caching still applies inside this call.
    filtered = add_calculated_indices(filtered, benchmark_source_global)

    # Calculated outputs are numeric by construction. Avoid reparsing every
    # calculated column on each rerun.
    calc_metric_cols = [
        c for c in CALCULATED_OUTPUT_COLUMNS
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])
    ]

    # Runtime contract for Calculated Metrics v1.1, displayed only in Players.
    current_families = sorted({
        f for f in filtered["Main Position"].map(calculated_position_family).dropna().tolist()
    }) if "Main Position" in filtered.columns else []
    expected_family_indices = [
        name for name, cfg in CALCULATED_METRICS.items()
        if any(f in cfg["families"] for f in current_families)
    ]
    populated_family_indices = [
        c for c in expected_family_indices
        if c in filtered.columns and filtered[c].notna().any()
    ]

# Build the final numeric pool after any page-specific enrichment.
# On non-Players workspaces this is intentionally just the raw/profile-score pool.
raw_numeric_cols = list(get_numeric_columns(filtered))
numeric_cols = list(dict.fromkeys(raw_numeric_cols + calc_metric_cols))

# Two headline calculated indicators shown by default for each canonical profile.
# Presentation only: role scores, weights, benchmarks and KPI calculations are unchanged.
PROFILE_TABLE_CALC_DEFAULTS = {
    "Classic Goalkeeper": ["Calc: GK Shot Stopping", "Calc: GK Area Control"],
    "Sweeper Keeper": ["Calc: GK Area Control", "Calc: GK Distribution"],
    "Build-Up Keeper": ["Calc: GK Distribution", "Calc: GK Shot Stopping"],
    "Ball-Playing CB": ["Calc: Passing Progression", "Calc: Defensive Activity"],
    "Combative CB / Stopper": ["Calc: Duel Dominance", "Calc: Defensive Activity"],
    "Libero / Middle Pin CB": ["Calc: Passing Progression", "Calc: Defensive Activity"],
    "Wide CB (in 3)": ["Calc: Carrying Progression", "Calc: Duel Dominance"],
    "Defensive Midfielder #6": ["Calc: Defensive Activity", "Calc: Passing Progression"],
    "Deep-Lying Playmaker": ["Calc: Passing Progression", "Calc: Chance Creation"],
    "Box-to-Box Midfielder": ["Calc: Carrying Progression", "Calc: Defensive Activity"],
    "Playmaker #10": ["Calc: Chance Creation", "Calc: Carrying Progression"],
    "Full-Back": ["Calc: Defensive Activity", "Calc: Wide Delivery"],
    "Wing-Back": ["Calc: Carrying Progression", "Calc: Wide Delivery"],
    "Inverted Full-Back": ["Calc: Passing Progression", "Calc: Defensive Activity"],
    "Classic Winger": ["Calc: 1v1 Threat", "Calc: Wide Delivery"],
    "Inverted Winger": ["Calc: 1v1 Threat", "Calc: Box Threat"],
    "Target Man #9": ["Calc: Aerial Impact", "Calc: Link Play"],
    "Poacher": ["Calc: Finishing", "Calc: Box Threat"],
    "Pressing Forward": ["Calc: Defensive Activity", "Calc: Box Threat"],
    "Creative Forward / False 9": ["Calc: Chance Creation", "Calc: Link Play"],
    "Wide Forward / Inside 9": ["Calc: Box Threat", "Calc: 1v1 Threat"],
}

# Synchronize built-in role selectors after Apply profile, before those widgets
# are instantiated. This avoids Streamlit's post-instantiation session-state error.
_pending_workspace_profile = st.session_state.pop("_pending_workspace_profile_sync", None)
if _pending_workspace_profile in PROFILES:
    if _pending_workspace_profile in SINGLE_PLAYER_PROFILES:
        st.session_state["single_profile_role"] = _pending_workspace_profile
    st.session_state["multi_radar_profile"] = _pending_workspace_profile
    # Role Fit has a detail selector whose options depend on the selected player.
    # Store the preferred role separately and apply it only when compatible.
    st.session_state["_preferred_role_fit_detail"] = _pending_workspace_profile

# =========================
# Main workspaces — exactly one renders
# =========================
if app_page == "Players":
    # =========================
    # Top-N table
    # =========================
    st.subheader("Filtered Player Data")

    _status_ok = len(calc_metric_cols) == len(CALCULATED_OUTPUT_COLUMNS)
    _status_icon = "✅" if _status_ok else "❌"
    st.caption(
        f"{_status_icon} Calculated Metrics Engine v1.1 · "
        f"{len(calc_metric_cols)}/{len(CALCULATED_OUTPUT_COLUMNS)} outputs attached · "
        f"families in current selection: {', '.join(current_families) if current_families else 'none'} · "
        f"relevant composites populated: {len(populated_family_indices)}/{len(expected_family_indices)}"
    )
    if not _status_ok:
        missing_outputs = [c for c in CALCULATED_OUTPUT_COLUMNS if c not in filtered.columns]
        st.error("Calculated metric columns missing from the enriched dataframe: " + ", ".join(missing_outputs))

    # Optional competition-strength normalization for cross-league player ranking.
    # The original Profile Score is never overwritten. The adjusted score expresses
    # every player on the selected reference league's competition-strength scale.
    league_adjustment_on = False
    adjusted_score_col = None
    reference_league = None
    reference_rating = None
    league_strength_players = pd.DataFrame()

    if calc_col_name and calc_col_name in filtered.columns and "League" in filtered.columns:
        league_strength_players = build_league_strength_lookup(
            tuple(sorted(filtered["League"].dropna().astype(str).unique().tolist())),
            str(OPTA_LEAGUE_META_PATH),
        )
        if not league_strength_players.empty:
            league_adjustment_on = st.toggle(
                "Apply league adjustment",
                value=False,
                key="players_apply_league_adjustment",
                help=(
                    "Keeps the original Profile Score and adds a competition-adjusted score using "
                    "β = 0.020 per Opta league-rating point. Choose the league whose strength should "
                    "be the common comparison reference."
                ),
            )
            if league_adjustment_on:
                mapped_leagues = league_strength_players.sort_values(["rating", "League"], ascending=[False, True])["League"].tolist()
                reference_league = st.selectbox(
                    "Reference league for adjusted scores",
                    options=mapped_leagues,
                    index=0,
                    key="players_league_adjustment_reference",
                    help="Adjusted Score = Profile Score + 0.020 × (player league rating − reference league rating).",
                )
                ref_row = league_strength_players.loc[league_strength_players["League"].eq(reference_league)].iloc[0]
                reference_rating = float(ref_row["rating"])
                rating_map = league_strength_players.set_index("League")["rating"]
                rank_map = league_strength_players.set_index("League")["rank"]
                filtered["Opta League Rating"] = filtered["League"].map(rating_map)
                filtered["Opta Global Rank"] = filtered["League"].map(rank_map)
                adjusted_score_col = "League-Adjusted Score"
                filtered[adjusted_score_col] = (
                    pd.to_numeric(filtered[calc_col_name], errors="coerce")
                    + LEAGUE_TRANSLATION_BETA
                    * (pd.to_numeric(filtered["Opta League Rating"], errors="coerce") - reference_rating)
                )
                filtered["League Adjustment"] = filtered[adjusted_score_col] - pd.to_numeric(filtered[calc_col_name], errors="coerce")
                for _c in ["Opta League Rating", "Opta Global Rank", adjusted_score_col, "League Adjustment"]:
                    if _c not in numeric_cols:
                        numeric_cols.append(_c)
                st.caption(
                    f"League adjustment ON · reference: {reference_league} (Opta {reference_rating:.2f}) · "
                    f"β={LEAGUE_TRANSLATION_BETA:.3f}. Original Profile Score remains unchanged."
                )
        else:
            st.caption("League adjustment unavailable: none of the currently filtered leagues are mapped to Opta league strength.")

    # Position-aware default table columns. All source columns remain available
    # in the selector; this only controls what is shown initially.
    _gk_only_selection = bool(current_families) and set(current_families) == {"Goalkeeper"}

    # Default order: identity/context -> Age -> Minutes -> active role Score ->
    # position-relevant headline metrics -> other defaults -> KPI sub-scores.
    ID_COLS = [
        "Season label",
        "Player",
        "Team",
        "League",
        "Main Position",
        "Age",
        "Minutes played",
    ]

    exclude_cols = {"Market value"} if "Market value (M€)" in filtered.columns else set()
    display_options = [c for c in filtered.columns if c not in exclude_cols]

    default_cols = [c for c in ID_COLS if c in filtered.columns]

    # Put the active weighted role Z-score immediately after Minutes played.
    # Coverage remains available in the selector but is not shown by default.
    if calc_col_name and calc_col_name in filtered.columns:
        default_cols.append(calc_col_name)
        if league_adjustment_on and adjusted_score_col in filtered.columns:
            default_cols.extend(["Opta League Rating", adjusted_score_col])

    if _gk_only_selection:
        # GK-specific raw indicators replace attacking output columns in the default view.
        default_cols.extend([c for c in ["Save rate, %", "Prevented goals per 90"] if c in filtered.columns])
    else:
        default_cols.extend([c for c in ["Goals", "Assists", "xG", "xA"] if c in filtered.columns])

    # Exactly two calculated indicators relevant to the active profile.
    # Use the currently applied built-in profile stored in session state.
    # Custom profiles have profile_name=None and therefore simply receive no
    # automatic calculated-metric table defaults.
    active_profile_name = active.get("profile_name") if isinstance(active, dict) else None
    profile_calc_defaults = PROFILE_TABLE_CALC_DEFAULTS.get(active_profile_name, [])
    default_cols.extend([
        c for c in profile_calc_defaults
        if c in filtered.columns and filtered[c].notna().any()
    ])

    if calc_col_name and calc_col_name in filtered.columns:
        default_cols.extend([c for c in active_kpi_cols if c in filtered.columns])

    # Defensive dedupe while preserving the deliberate display order above.
    default_cols = list(dict.fromkeys(default_cols))

    selected_display_cols = st.multiselect("Columns to display", options=display_options, default=default_cols)

    # The active profile score is the primary scouting ranking. Calculated
    # indicators remain available but no longer override the role score default.
    rank_candidates = [calc_col_name] + CALCULATED_OUTPUT_COLUMNS + ["Assists per 90", "Goals per 90", "xA per 90", "xG per 90", "xA", "xG", "Minutes played"]
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

        # Compact, one-frame scouting table.
        # Keep source dataframe names unchanged; only display labels/widths are shortened.
        table_column_config = {}

        compact_labels = {
            "Season label": "Season",
            "Main Position": "Pos",
            "Minutes played": "Min",
            "Save rate, %": "Save %",
            "Prevented goals per 90": "Prev G/90",
            "KPI: Shot Stopping": "Shot Stop",
            "KPI: Area Control": "Area Ctrl",
            "KPI: Distribution": "Distrib.",
            "KPI: Build-Up": "Build-Up",
            "Calc: GK Shot Stopping": "GK Shot Stop",
            "Calc: GK Area Control": "GK Area Ctrl",
            "Calc: GK Distribution": "GK Distrib.",
            "Calc: Defensive Activity": "Def. Activity",
            "Calc: Duel Dominance": "Duel Dom.",
            "Calc: Aerial Impact": "Aerial Impact",
            "Calc: Passing Progression": "Pass Prog.",
            "Calc: Carrying Progression": "Carry Prog.",
            "Calc: Wide Delivery": "Wide Delivery",
            "Calc: 1v1 Threat": "1v1 Threat",
            "Calc: Chance Creation": "Chance Create",
            "Calc: Box Threat": "Box Threat",
            "Calc: Finishing": "Finishing",
            "Calc: Link Play": "Link Play",
            "Opta League Rating": "League Rating",
            "Opta Global Rank": "Opta Rank",
            "League-Adjusted Score": "Adjusted Score",
            "League Adjustment": "Adjustment",
        }

        for c in table_df.columns:
            c_low = str(c).lower()
            display_label = compact_labels.get(c, c)

            # Role score keeps its real dataframe name internally but uses a compact header.
            if c.startswith("Score:"):
                display_label = "Profile Score"

            if c == "Season label":
                table_column_config[c] = st.column_config.TextColumn(display_label, width=65)
            elif c == "Player":
                table_column_config[c] = st.column_config.TextColumn(display_label, width=105)
            elif c == "Team":
                table_column_config[c] = st.column_config.TextColumn(display_label, width=115)
            elif c == "League":
                table_column_config[c] = st.column_config.TextColumn(display_label, width=180)
            elif c == "Main Position":
                table_column_config[c] = st.column_config.TextColumn(display_label, width=50)
            elif c == "Age":
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=45, format="%d")
            elif c in {"Minutes played", "Matches played", "Goals", "Assists"}:
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=60, format="%d")
            elif c.startswith("Score:"):
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=90, format="%.2f")
            elif c in {"League-Adjusted Score", "League Adjustment", "Opta League Rating"}:
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=92, format="%.2f")
            elif c == "Opta Global Rank":
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=72, format="%d")
            elif c == "Save rate, %":
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=65, format="%.2f")
            elif c == "Prevented goals per 90":
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=72, format="%.2f")
            elif c.startswith("KPI:"):
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=78, format="%.2f")
            elif c.startswith("Calc:"):
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=85, format="%.2f")
            elif "%" in str(c) or c_low.endswith("rate"):
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=65, format="%.2f")
            elif pd.api.types.is_numeric_dtype(table_df[c]):
                table_column_config[c] = st.column_config.NumberColumn(display_label, width=65, format="%.2f")
            else:
                table_column_config[c] = st.column_config.TextColumn(display_label, width=90)

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
            column_config=table_column_config,
        )

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

    with st.expander("Calculated Metrics Methodology", expanded=False):
        _available_calc = [c for c in CALCULATED_OUTPUT_COLUMNS if c in filtered.columns]
        _populated_calc = [c for c in _available_calc if filtered[c].notna().any()]
        st.caption(f"Calculated outputs in current dataset: {len(_available_calc)} · populated for current filtered players: {len(_populated_calc)}")
        st.markdown("**Composite Indices** are positional-family benchmark percentiles (0–100). Components are standardized as direction-aware z-scores before weighting; 70% component-weight coverage is required.")
        for name, cfg in CALCULATED_METRICS.items():
            parts = " · ".join(f"{metric} {weight}%" for metric, weight in cfg["components"].items())
            st.markdown(f"**{name}** — {', '.join(cfg['families'])}  \n{parts}")
        st.markdown("**Efficiency Metrics** are direct calculations: Calc: Shot Quality (xG/Shot) = xG/90 ÷ Shots/90; Calc: Finishing Above xG /90 = NPG/90 − xG/90; Calc: Chance Quality (xA/Key Pass) = xA/90 ÷ Key Passes/90.")

    # =========================
    # Scatter plot
    # =========================
    st.subheader("Player Performance Visualization")

    plot_metrics = [c for c in numeric_cols if c not in {"Age", "Market value"}]
    # Final selector contract: explicitly append every attached calculated output.
    # This makes Scatter independent of any upstream numeric-column inference.
    plot_metrics = list(dict.fromkeys(plot_metrics + calc_metric_cols))
    if calc_metric_cols:
        st.caption(f"Scatter calculated metrics available: {len(calc_metric_cols)} (type `Calc:` in X/Y to filter them)")
    if not plot_metrics:
        st.warning("No numerical metrics available for plotting.")
    else:
        # Profile-aware scatter defaults: pair the active weighted role score
        # with the most relevant calculated indicator for that role.
        PROFILE_SCATTER_X_DEFAULT = {
            "Classic Goalkeeper": "Calc: GK Shot Stopping",
            "Sweeper Keeper": "Calc: GK Area Control",
            "Build-Up Keeper": "Calc: GK Distribution",
            "Ball-Playing CB": "Calc: Passing Progression",
            "Combative CB / Stopper": "Calc: Defensive Activity",
            "Libero / Middle Pin CB": "Calc: Passing Progression",
            "Wide CB (in 3)": "Calc: Carrying Progression",
            "Defensive Midfielder #6": "Calc: Defensive Activity",
            "Deep-Lying Playmaker": "Calc: Passing Progression",
            "Box-to-Box Midfielder": "Calc: Carrying Progression",
            "Playmaker #10": "Calc: Chance Creation",
            "Full-Back": "Calc: Passing Progression",
            "Wing-Back": "Calc: Wide Delivery",
            "Inverted Full-Back": "Calc: Passing Progression",
            "Classic Winger": "Calc: Wide Delivery",
            "Inverted Winger": "Calc: Box Threat",
            "Target Man #9": "Calc: Aerial Impact",
            "Poacher": "Calc: Finishing",
            "Pressing Forward": "Calc: Defensive Activity",
            "Creative Forward / False 9": "Calc: Chance Creation",
            "Wide Forward / Inside 9": "Calc: Box Threat",
        }
        _profile_x = PROFILE_SCATTER_X_DEFAULT.get(active_role) if active else None
        if _profile_x in plot_metrics and filtered[_profile_x].notna().any():
            x_default = _profile_x
        elif _gk_only_selection and "Calc: GK Shot Stopping" in plot_metrics:
            x_default = "Calc: GK Shot Stopping"
        elif "Goals per 90" in plot_metrics:
            x_default = "Goals per 90"
        else:
            x_default = plot_metrics[0]

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
            _color_options = [o for o in ["Season label", "Main Position", "Team", "League", "Foot", "None"] if o == "None" or o in filtered.columns]
            _color_default = _color_options.index("Team") if "Team" in _color_options else 0
            color_by = st.selectbox(
                "Color by",
                options=_color_options,
                index=_color_default,
            )

        _size_options = [o for o in ["None", "Minutes played", "Market value (M€)", "Age", "Matches played"] if o == "None" or o in filtered.columns]
        size_by = st.selectbox(
            "Size by",
            options=_size_options,
            index=_size_options.index("None"),
        )

        rank_axis = st.radio("Sort Top-N players by", ["X-axis", "Y-axis"], index=1, horizontal=True)
        sort_metric = y_axis if rank_axis == "Y-axis" else x_axis
        plot_limit = st.slider(f"Number of players to plot (Top-N by {sort_metric})", 1, min(30, len(filtered)), min(10, len(filtered)))

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

        show_labels = st.checkbox("Show player labels on chart", value=True)

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

        # When Profile Score is the Y-axis and league adjustment is enabled,
        # overlay each player's adjusted score at the same X coordinate. The
        # lighter point preserves the original team/category colour, while a
        # thin connector makes the size and direction of the adjustment visible.
        if (
            league_adjustment_on
            and adjusted_score_col in plot_df.columns
            and y_axis == calc_col_name
            and plot_df[adjusted_score_col].notna().any()
        ):
            import plotly.graph_objects as go

            original_color_by = None if color_by == "None" else color_by
            color_lookup = {}
            for tr in fig.data:
                if getattr(tr, "mode", None) and "markers" in tr.mode:
                    color_lookup[str(tr.name)] = getattr(tr.marker, "color", None)
            fallback_color = "#636EFA"

            # Connector segments are drawn first so both endpoints stay crisp.
            for _, r in plot_df.dropna(subset=[x_axis, y_axis, adjusted_score_col]).iterrows():
                category = str(r[original_color_by]) if original_color_by else ""
                base_color = color_lookup.get(category, fallback_color)
                fig.add_trace(go.Scatter(
                    x=[r[x_axis], r[x_axis]],
                    y=[r[y_axis], r[adjusted_score_col]],
                    mode="lines",
                    line=dict(color=base_color, width=1.5),
                    opacity=0.28,
                    hoverinfo="skip",
                    showlegend=False,
                ))

            if original_color_by:
                grouped = plot_df.groupby(original_color_by, dropna=False, sort=False)
            else:
                grouped = [("Adjusted", plot_df)]

            for category, g in grouped:
                g = g.dropna(subset=[x_axis, adjusted_score_col])
                if g.empty:
                    continue
                category_key = str(category) if original_color_by else ""
                base_color = color_lookup.get(category_key, fallback_color)
                custom = np.column_stack([
                    g["Player"].astype(str) if "Player" in g.columns else pd.Series([""] * len(g), index=g.index),
                    pd.to_numeric(g[y_axis], errors="coerce"),
                    pd.to_numeric(g[adjusted_score_col], errors="coerce"),
                    pd.to_numeric(g.get("Opta League Rating", np.nan), errors="coerce"),
                    g["League"].astype(str) if "League" in g.columns else pd.Series([""] * len(g), index=g.index),
                    pd.to_numeric(g.get("League Adjustment", np.nan), errors="coerce"),
                ])
                fig.add_trace(go.Scatter(
                    x=g[x_axis],
                    y=g[adjusted_score_col],
                    mode="markers",
                    name="Adjusted" if not original_color_by else f"{category} · adjusted",
                    legendgroup=f"adjusted_{category_key}",
                    showlegend=False,
                    marker=dict(
                        size=16,
                        color=base_color,
                        opacity=0.35,
                        line=dict(width=1.5, color=base_color),
                    ),
                    customdata=custom,
                    hovertemplate=(
                        "Player: %{customdata[0]}<br>"
                        + x_axis + ": %{x:.2f}<br>"
                        + "Original Profile Score: %{customdata[1]:.2f}<br>"
                        + "Adjusted Score: %{customdata[2]:.2f}<br>"
                        + "League: %{customdata[4]}<br>"
                        + "Opta League Rating: %{customdata[3]:.2f}<br>"
                        + "League Adjustment: %{customdata[5]:+.2f}<br>"
                        + f"Reference: {reference_league} ({reference_rating:.2f})"
                        + "<extra>League-adjusted</extra>"
                    ),
                ))

            fig.update_layout(
                title=f"{y_axis} vs. {x_axis} by Player · original + league-adjusted",
            )
        elif league_adjustment_on and y_axis != calc_col_name:
            st.caption("League-adjusted overlay appears when the active Profile Score is selected on the Y-axis.")

        st.plotly_chart(fig, width="stretch")


elif app_page in ("Single Player", "Multi Player"):
    # =========================
    # Player profile / comparison visualizations
    # =========================
    st.subheader(
        "Single-Player Profile"
        if app_page == "Single Player"
        else "Multi-Player Comparison"
    )

    comparison_mode = (
        "Single-Player Profile"
        if app_page == "Single Player"
        else "Multi-Player Radar"
    )

    if comparison_mode == "Multi-Player Radar":
        player_options = sorted(filtered["Player"].dropna().unique().tolist()) if "Player" in filtered.columns else []
        compare_players = st.multiselect(
            "Players to compare (max 5 recommended)",
            options=player_options,
            default=[],
            key="multi_compare_players",
        )

        # Keep the multi-player comparison tied to the same canonical role framework
        # as the weighted Profile Score and single-player radar.
        active_builtin_profile = None
        if active and isinstance(active.get("calc_col"), str):
            active_label = str(active.get("calc_col"))
            if active_label.startswith("Score: "):
                candidate = active_label.replace("Score: ", "", 1)
                if candidate in PROFILES:
                    active_builtin_profile = candidate

        radar_metric_mode = st.radio(
            "Radar metric mode",
            ["Profile metrics", "Custom metrics"],
            horizontal=True,
            key="multi_radar_metric_mode",
        )

        comp_metric_choices = get_numeric_columns(filtered)
        selected_radar_profile = None
        radar_kpi_lookup: Dict[str, str] = {}

        if radar_metric_mode == "Profile metrics":
            profile_names = list(PROFILES.keys())
            default_profile_index = profile_names.index(active_builtin_profile) if active_builtin_profile in profile_names else 0
            selected_radar_profile = st.selectbox(
                "Comparison profile",
                options=profile_names,
                index=default_profile_index,
                key="multi_radar_profile",
                help="Uses the same 15 metrics as the selected built-in Profile Score and single-player role profile.",
            )

            requested_compare_metrics = PROFILES[selected_radar_profile]
            comp_metrics, missing_compare_metrics = resolve_metrics_aliases(
                requested_compare_metrics,
                filtered.columns.tolist(),
            )
            radar_kpi_lookup = {
                metric: kpi for kpi, metric in SINGLE_PLAYER_PROFILES[selected_radar_profile]
            }

            st.caption(
                f"{selected_radar_profile}: {len(comp_metrics)} of 15 profile metrics available. "
                "The radar uses direction-aware z-scores against the current filtered player population."
            )
            if missing_compare_metrics:
                st.warning("Missing profile metrics: " + ", ".join(missing_compare_metrics))
        else:
            default_custom_metrics = [m for m in profile_metrics_in_use if m in comp_metric_choices]
            if not default_custom_metrics:
                default_custom_metrics = comp_metric_choices[: min(6, len(comp_metric_choices))]
            comp_metrics = st.multiselect(
                "Custom metrics for comparison table & radar",
                options=comp_metric_choices,
                default=default_custom_metrics,
                key="multi_compare_custom_metrics",
            )

        if compare_players and "Player" in filtered.columns and comp_metrics:
            comp_rows = filtered.loc[filtered["Player"].isin(compare_players)].copy()

            # If a player has more than one row after filtering, keep the row with most minutes.
            if "Minutes played" in comp_rows.columns:
                comp_rows["_cmp_minutes"] = pd.to_numeric(comp_rows["Minutes played"], errors="coerce").fillna(0)
                comp_rows = comp_rows.sort_values("_cmp_minutes", ascending=False).drop_duplicates("Player")
                comp_rows = comp_rows.drop(columns="_cmp_minutes")
            else:
                comp_rows = comp_rows.drop_duplicates("Player")

            comp_df = comp_rows.set_index("Player")
            available_players = [p for p in compare_players if p in comp_df.index]

            # Raw-value comparison table.
            show_table = comp_df[comp_metrics].copy()
            for c in show_table.columns:
                show_table[c] = pd.to_numeric(show_table[c], errors="coerce").round(2)

            # Use the same explicit benchmark logic as Profile Score / Single Player.
            multi_role = selected_radar_profile if radar_metric_mode == "Profile metrics" else None
            multi_benchmark_df = benchmark_population_for_role(
                benchmark_source_global,
                multi_role,
                benchmark_mode,
                selected_positions,
            )
            if len(multi_benchmark_df) < int(min_benchmark_n):
                st.warning(
                    f"Multi-player benchmark sample is small ({len(multi_benchmark_df)} players; "
                    f"recommended minimum {int(min_benchmark_n)})."
                )
            baseX = multi_benchmark_df[comp_metrics].apply(pd.to_numeric, errors="coerce")
            means = baseX.mean(axis=0)
            stds = baseX.std(axis=0, ddof=0).replace(0, np.nan)

            z_table = pd.DataFrame(index=show_table.index, columns=comp_metrics, dtype=float)
            for player in available_players:
                row = show_table.loc[player, comp_metrics]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                row = row.apply(pd.to_numeric, errors="coerce")
                z = ((row - means) / stds).fillna(0.0)
                for m in comp_metrics:
                    if m in LOWER_IS_BETTER:
                        z[m] = -z[m]
                z_table.loc[player, comp_metrics] = z.values

            # Direction-aware percentiles against exactly the same benchmark population.
            percentile_table = pd.DataFrame(index=show_table.index, columns=comp_metrics, dtype=float)
            for player in available_players:
                for metric in comp_metrics:
                    value = pd.to_numeric(pd.Series([show_table.loc[player, metric]]), errors="coerce").iloc[0]
                    percentile_table.loc[player, metric] = percentile_rank_against_population(
                        baseX[metric],
                        value,
                        lower_is_better=(metric in LOWER_IS_BETTER),
                    )

            # For built-in role comparisons, show the default weighted role score
            # using exactly the same 15 metrics as the radar.
            profile_score_row = None
            if radar_metric_mode == "Profile metrics" and selected_radar_profile:
                default_weight_map = DEFAULT_WEIGHTS[selected_radar_profile]
                resolved_weight_map: Dict[str, float] = {}
                for requested_metric, pct in default_weight_map.items():
                    resolved, _missing = resolve_metrics_aliases(
                        [requested_metric],
                        filtered.columns.tolist(),
                    )
                    if resolved and resolved[0] in comp_metrics:
                        resolved_weight_map[resolved[0]] = float(pct)

                total_pct = sum(resolved_weight_map.values())
                if total_pct > 0:
                    profile_score_row = {}
                    for player in available_players:
                        z = pd.to_numeric(z_table.loc[player, comp_metrics], errors="coerce").fillna(0.0)
                        score = 0.0
                        for metric, pct in resolved_weight_map.items():
                            score += float(z.get(metric, 0.0)) * (pct / total_pct)
                        profile_score_row[player] = round(score, 2)

            # Friendly labels while keeping raw Wyscout columns internally.
            radar_display_aliases = {
                "Save rate, %": "Save Rate %",
                "Prevented goals per 90": "Goals Prevented /90",
                "Conceded goals per 90": "Goals Conceded /90",
                "Shots against per 90": "Shots Faced /90",
                "xG against per 90": "xGA /90",
                "Exits per 90": "Exits /90",
                "Aerial duels per 90.1": "Aerial Duels /90",
                "Aerial duels per 90": "Aerial Duels /90",
                "Aerial duels won, %": "Aerial Duel Win %",
                "Passes per 90": "Passes /90",
                "Accurate passes, %": "Pass Accuracy %",
                "Long passes per 90": "Long Passes /90",
                "Accurate long passes, %": "Long Pass Accuracy %",
                "Back passes received as GK per 90": "GK Back Passes /90",
                "Forward passes per 90": "Forward Passes /90",
                "Accurate forward passes, %": "Forward Pass Accuracy %",
                "Progressive passes per 90": "Progressive Passes /90",
                "Accurate progressive passes, %": "Progressive Pass Accuracy %",
                "Passes to final third per 90": "Final Third Passes /90",
                "Received passes per 90": "Passes Received /90",
                "Progressive runs per 90": "Progressive Runs /90",
                "Interceptions per 90": "Interceptions /90",
                "PAdj Interceptions": "PAdj Interceptions",
                "Successful defensive actions per 90": "Defensive Actions /90",
                "Defensive duels per 90": "Defensive Duels /90",
                "Defensive duels won, %": "Defensive Duel Win %",
                "PAdj Sliding tackles": "PAdj Sliding Tackles",
                "Sliding tackles per 90": "Sliding Tackles /90",
                "Shots blocked per 90": "Shots Blocked /90",
                "Fouls per 90": "Fouls /90",
                "Yellow cards per 90": "Yellow Cards /90",
                "Dribbles per 90": "Dribbles /90",
                "Successful dribbles, %": "Dribble Success %",
                "Accelerations per 90": "Accelerations /90",
                "Crosses per 90": "Crosses /90",
                "Accurate crosses, %": "Cross Accuracy %",
                "Crosses to goalie box per 90": "Box Crosses /90",
                "Passes to penalty area per 90": "Penalty Area Passes /90",
                "Shot assists per 90": "Shot Assists /90",
                "xA per 90": "xA /90",
                "xG per 90": "xG /90",
                "Shots per 90": "Shots /90",
                "Touches in box per 90": "Box Touches /90",
                "Non-penalty goals per 90": "Non-Penalty Goals /90",
                "Successful attacking actions per 90": "Attacking Actions /90",
                "Smart passes per 90": "Smart Passes /90",
                "Key passes per 90": "Key Passes /90",
                "Deep completions per 90": "Deep Completions /90",
                "Through passes per 90": "Through Passes /90",
                "Offensive duels per 90": "Offensive Duels /90",
                "Offensive duels won, %": "Offensive Duel Win %",
                "Goal conversion, %": "Goal Conversion %",
                "Shots on target, %": "Shots on Target %",
                "Head goals per 90": "Headed Goals /90",
                "Received long passes per 90": "Long Passes Received /90",
                "Fouls suffered per 90": "Fouls Won /90",
            }

            table_for_display = show_table.copy()
            table_for_display.index.name = None
            table_for_display = table_for_display.rename(columns=radar_display_aliases).T

            if profile_score_row is not None:
                score_df = pd.DataFrame(
                    {player: [profile_score_row.get(player, np.nan)] for player in available_players},
                    index=[f"Profile Score: {selected_radar_profile}"],
                )
                table_for_display = pd.concat([score_df, table_for_display], axis=0)

            st.dataframe(table_for_display, width="stretch")

            comparison_scale = st.radio(
                "Comparison scale",
                ["Percentile", "Z-score"],
                horizontal=True,
                key="multi_compare_scale",
                help="Switches only the comparison visual. Profile Score remains the weighted direction-aware z-score composite.",
            )
            st.caption(
                f"Benchmark: {benchmark_mode} · selected leagues · {min_minutes}+ min · "
                f"N={len(multi_benchmark_df):,}"
            )

            # Multi-player role wheel: same architecture as the single-player profile.
            if radar_metric_mode == "Profile metrics" and selected_radar_profile:
                fig_radar = multi_player_profile_wheel(
                    raw_table=show_table,
                    z_table=z_table,
                    percentile_table=percentile_table,
                    display_scale=comparison_scale,
                    players=available_players,
                    profile_name=selected_radar_profile,
                    metrics=comp_metrics,
                    kpi_lookup=radar_kpi_lookup,
                )
            else:
                # Custom mode has no canonical KPI groups, so use one neutral group.
                custom_kpi_lookup = {m: "Custom Metrics" for m in comp_metrics}
                fig_radar = multi_player_profile_wheel(
                    raw_table=show_table,
                    z_table=z_table,
                    percentile_table=percentile_table,
                    display_scale=comparison_scale,
                    players=available_players,
                    profile_name="Custom Metrics",
                    metrics=comp_metrics,
                    kpi_lookup=custom_kpi_lookup,
                )

            st.plotly_chart(fig_radar, width="stretch")

            st.caption(
                "Multi-player comparison uses the same fixed-radius scouting-wheel architecture as the single-player profile: "
                "the same 15 metrics, KPI bands, metric order and KPI gaps. Percentile (0–100) is the default scout-facing view; "
                "direction-aware Z-score (−2 to +2) remains available as an advanced view. Both use the same benchmark population. Hover always shows raw value, percentile and z-score. "
                "Profile Score weighting remains a direction-aware weighted z-score composite and is separate from wheel geometry."
            )

            csv_export = show_table.copy()
            if profile_score_row is not None:
                csv_export.insert(
                    0,
                    f"Profile Score: {selected_radar_profile}",
                    pd.Series(profile_score_row),
                )
            csv_buf2 = StringIO()
            csv_export.to_csv(csv_buf2)
            st.download_button(
                "⬇️ Download comparison (CSV)",
                data=csv_buf2.getvalue(),
                file_name="player_comparison.csv",
                mime="text/csv",
            )

        elif compare_players and not comp_metrics:
            st.info("No valid comparison metrics are available for the selected profile.")
        else:
            st.info("Select players above to compare their stats and see the role radar.")

    else:
        st.markdown("#### Single-Player Role Profile")
        if SINGLE_PLAYER_METRIC_POLICY_ISSUES:
            st.warning(
                "Single-player profile metric policy issue: "
                + "; ".join(SINGLE_PLAYER_METRIC_POLICY_ISSUES)
            )
        st.caption(
            "15 role-specific normalized metrics grouped by KPI family. Volume metrics are per 90; efficiency metrics are percentages; PAdj metrics remain possession-adjusted. "
            "Built-in Profile Scores use these same 15 role metrics with reviewed weights; the wheel itself remains unweighted. Percentiles and z-scores are direction-aware."
        )

        single_player_options = sorted(filtered["Player"].dropna().unique().tolist()) if "Player" in filtered.columns else []

        if not single_player_options:
            st.info("No players are available for the current filters.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                single_player = st.selectbox(
                    "Player",
                    options=single_player_options,
                    key="single_profile_player",
                )
            with c2:
                single_role = st.selectbox(
                    "Role / profile",
                    options=list(SINGLE_PLAYER_PROFILES.keys()),
                    key="single_profile_role",
                )

            player_rows = filtered.loc[filtered["Player"] == single_player].copy()
            if "Minutes played" in player_rows.columns:
                player_rows["_single_minutes"] = pd.to_numeric(player_rows["Minutes played"], errors="coerce").fillna(0)
                player_rows = player_rows.sort_values("_single_minutes", ascending=False)
            player_row = player_rows.iloc[0]

            scale_mode = st.radio(
                "Performance scale",
                ["Percentile", "Z-score"],
                horizontal=True,
                key="single_profile_scale",
                help="Percentile is the default scout-facing view (0–100). Z-score remains available as an advanced analytical view.",
            )

            single_benchmark_choice = st.radio(
                "Single-player benchmark",
                ["Use global benchmark", "Same Main Position"],
                horizontal=True,
                key="single_benchmark_mode",
            )

            if single_benchmark_choice == "Same Main Position" and "Main Position" in benchmark_source_global.columns:
                player_pos = player_row.get("Main Position")
                benchmark_df = benchmark_source_global.loc[
                    benchmark_source_global["Main Position"] == player_pos
                ].copy()
                benchmark_desc = f"{player_pos} · selected leagues · {min_minutes}+ min"
            else:
                benchmark_df = benchmark_population_for_role(
                    benchmark_source_global,
                    single_role,
                    benchmark_mode,
                    selected_positions,
                )
                benchmark_desc = f"{benchmark_mode} · selected leagues · {min_minutes}+ min"

            if len(benchmark_df) < int(min_benchmark_n):
                st.warning(
                    f"Benchmark sample is small ({len(benchmark_df)} players; recommended minimum {int(min_benchmark_n)}). "
                    "Consider broadening the benchmark."
                )

            profile_df, missing_single_metrics = build_single_player_profile(
                player_row=player_row,
                benchmark_df=benchmark_df,
                role_name=single_role,
            )

            team = str(player_row.get("Team", "")).strip()
            league = str(player_row.get("League", "")).strip()
            season = str(player_row.get("Season label", "")).strip()
            player_position = str(player_row.get("Main Position", "")).strip()

            header_bits = [x for x in [team, player_position] if x and x.lower() != "nan"]

            info1, info2, info3 = st.columns(3)
            info1.metric("Benchmark players", f"{len(benchmark_df):,}")
            info2.metric("Metrics displayed", f"{len(profile_df)}")
            info3.metric("Role", single_role)

            if missing_single_metrics:
                st.caption(
                    "Unavailable in this dataset and skipped: "
                    + ", ".join(missing_single_metrics)
                )

            if profile_df.empty:
                st.warning("None of the selected role metrics contain usable values for this player/benchmark.")
            else:
                subtitle_parts = [single_role, benchmark_desc]
                if season and season.lower() != "none" and season.lower() != "nan":
                    subtitle_parts.append(f"Season {season}")
                subtitle = " | ".join(subtitle_parts)

                fig_single = single_player_wheel(
                    profile_df=profile_df.reset_index(drop=True),
                    player_name=single_player,
                    subtitle=subtitle,
                    scale_mode=scale_mode,
                )
                st.plotly_chart(fig_single, width="stretch")

                profile_table = profile_df.copy()
                profile_table["Raw Value"] = profile_table["Raw Value"].round(2)
                profile_table["Percentile"] = profile_table["Percentile"].round(0).astype(int)
                profile_table["Z-score"] = profile_table["Z-score"].round(2)
                st.dataframe(profile_table, width="stretch", hide_index=True)

                csv_single = StringIO()
                profile_table.to_csv(csv_single, index=False)
                st.download_button(
                    "⬇️ Download single-player profile (CSV)",
                    data=csv_single.getvalue(),
                    file_name=f"{safe_widget_key(single_player, single_role)}_percentile_profile.csv",
                    mime="text/csv",
                )



elif app_page == "Role Fit":
    # =========================
    # ROLE FIT v1
    # =========================
    st.markdown("---")
    st.header("🧭 Role Fit")
    st.caption(
        "Evaluate one player across position-compatible canonical roles. "
        "Each role is scored against its own role-relevant benchmark using the same weights, "
        "coverage rules and direction-aware methodology as Profile Score."
    )

    role_fit_pool = filtered_base
    if not role_fit_pool.empty and "Player" in role_fit_pool.columns:
        rf_c1, rf_c2 = st.columns([2, 1])
        with rf_c1:
            role_fit_player = st.selectbox(
                "Player",
                sorted(role_fit_pool["Player"].dropna().astype(str).unique().tolist()),
                key="role_fit_player",
            )

        player_candidates = role_fit_pool.loc[role_fit_pool["Player"].astype(str) == str(role_fit_player)].copy()
        # If duplicate player rows survive filters, use the row with the most minutes.
        if "Minutes played" in player_candidates.columns:
            player_candidates["_rf_minutes"] = pd.to_numeric(player_candidates["Minutes played"], errors="coerce").fillna(0)
            player_candidates = player_candidates.sort_values("_rf_minutes", ascending=False)
        rf_player_row = player_candidates.iloc[0]

        main_pos = rf_player_row.get("Main Position", "")
        compatible_roles = compatible_roles_for_position(main_pos)

        with rf_c2:
            st.metric("Main Position", str(main_pos) if pd.notna(main_pos) else "—")

        info_cols = st.columns(4)
        info_cols[0].metric("Team", str(rf_player_row.get("Team", "—")))
        age_val = pd.to_numeric(pd.Series([rf_player_row.get("Age", np.nan)]), errors="coerce").iloc[0]
        info_cols[1].metric("Age", f"{age_val:.0f}" if pd.notna(age_val) else "—")
        min_val = pd.to_numeric(pd.Series([rf_player_row.get("Minutes played", np.nan)]), errors="coerce").iloc[0]
        info_cols[2].metric("Minutes", f"{min_val:,.0f}" if pd.notna(min_val) else "—")
        info_cols[3].metric("Compatible roles", str(len(compatible_roles)))

        if not compatible_roles:
            st.info(
                f"No automatic Role Fit mapping is defined for Main Position '{main_pos}'. "
                "Use the existing Single-Player Profile to evaluate a role manually."
            )
        else:
            fit_results = []
            fit_detail: Dict[str, Dict[str, object]] = {}

            for role in compatible_roles:
                role_benchmark = benchmark_population_for_role(
                    benchmark_source_global,
                    role,
                    "Role position",  # Role Fit intentionally normalizes every role to its own family.
                    selected_positions,
                )
                result = role_fit_for_player(
                    rf_player_row,
                    role,
                    role_benchmark,
                    coverage_threshold=float(coverage_threshold_pct) / 100.0,
                )
                fit_detail[role] = result
                fit_results.append({
                    "Role": role,
                    "Role Score": result["Role Score"],
                    "Role Percentile": result["Role Percentile"],
                    "Coverage %": result["Coverage %"],
                    "Benchmark N": result["Benchmark N"],
                })

            fit_df = pd.DataFrame(fit_results)
            fit_df = fit_df.sort_values(
                ["Role Percentile", "Role Score"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)

            st.subheader("Role Fit overview")
            st.caption(
                "Role Percentile is the player's weighted Role Score percentile inside that role's own benchmark. "
                "It is the cleaner cross-role reference; raw Role Scores are still shown for transparency."
            )

            # Horizontal percentile chart with the same four performance bands
            # used by the Single Player wheel.
            chart_df = fit_df.dropna(subset=["Role Percentile"]).copy()
            if not chart_df.empty:
                chart_df = chart_df.sort_values("Role Percentile", ascending=True)
                chart_df["Performance Band"] = pd.cut(
                    pd.to_numeric(chart_df["Role Percentile"], errors="coerce"),
                    bins=[-np.inf, 25, 50, 75, np.inf],
                    labels=["<25", "25–49", "50–74", "75+"],
                    right=False,
                )

                role_band_colors = {
                    "<25": "#8B1E1E",
                    "25–49": "#E57373",
                    "50–74": "#81C784",
                    "75+": "#1B5E20",
                }

                role_fig = px.bar(
                    chart_df,
                    x="Role Percentile",
                    y="Role",
                    orientation="h",
                    text="Role Percentile",
                    color="Performance Band",
                    color_discrete_map=role_band_colors,
                    category_orders={"Performance Band": ["<25", "25–49", "50–74", "75+"]},
                    hover_data={
                        "Role Score": ":.2f",
                        "Coverage %": ":.0f",
                        "Benchmark N": True,
                        "Role Percentile": ":.0f",
                        "Performance Band": False,
                    },
                    range_x=[0, 100],
                    title="Role percentile by compatible profile",
                )
                role_fig.update_traces(
                    texttemplate="%{text:.0f}",
                    textposition="outside",
                    cliponaxis=False,
                )
                role_fig.update_layout(
                    height=max(330, 75 * len(chart_df)),
                    xaxis_title="Role Percentile",
                    yaxis_title="",
                    showlegend=False,
                    margin=dict(l=20, r=45, t=55, b=35),
                )
                st.plotly_chart(role_fig, width="stretch")

            # Ranked audit table. Role Percentile is the primary cross-role
            # comparison; raw weighted z-score remains visible for transparency.
            display_fit = fit_df.copy()
            display_fit["Role Percentile"] = pd.to_numeric(
                display_fit["Role Percentile"], errors="coerce"
            )
            display_fit = display_fit.sort_values(
                ["Role Percentile", "Role Score"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)
            display_fit.insert(0, "Rank", np.arange(1, len(display_fit) + 1))

            for c in ["Role Score", "Role Percentile", "Coverage %"]:
                display_fit[c] = pd.to_numeric(display_fit[c], errors="coerce").round(
                    2 if c == "Role Score" else 0
                )
            if "Benchmark N" in display_fit.columns:
                display_fit["Benchmark N"] = pd.to_numeric(
                    display_fit["Benchmark N"], errors="coerce"
                ).round(0).astype("Int64")

            display_fit = display_fit[
                ["Rank", "Role", "Role Percentile", "Role Score", "Coverage %", "Benchmark N"]
            ]

            st.dataframe(
                display_fit,
                width="stretch",
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d", width=55),
                    "Role": st.column_config.TextColumn("Role", width="medium"),
                    "Role Percentile": st.column_config.NumberColumn(
                        "Role Percentile", format="%.0f", width="small"
                    ),
                    "Role Score": st.column_config.NumberColumn(
                        "Role Score", format="%.2f", width="small"
                    ),
                    "Coverage %": st.column_config.NumberColumn(
                        "Coverage %", format="%.0f", width="small"
                    ),
                    "Benchmark N": st.column_config.NumberColumn(
                        "Benchmark N", format="%d", width="small"
                    ),
                },
            )

            low_n_roles = fit_df.loc[fit_df["Benchmark N"] < int(min_benchmark_n), "Role"].tolist()
            if low_n_roles:
                st.warning(
                    "Small benchmark sample for: " + ", ".join(low_n_roles) +
                    f". Recommended minimum is {int(min_benchmark_n)}."
                )

            selectable_roles = (
                fit_df.assign(
                    _role_pct_sort=pd.to_numeric(fit_df["Role Percentile"], errors="coerce")
                )
                .sort_values("_role_pct_sort", ascending=False, na_position="last")["Role"]
                .tolist()
            )

            preferred_role_fit = st.session_state.pop("_preferred_role_fit_detail", None)
            if preferred_role_fit in selectable_roles:
                st.session_state["role_fit_detail_role"] = preferred_role_fit
            elif (
                "role_fit_detail_role" in st.session_state
                and st.session_state["role_fit_detail_role"] not in selectable_roles
            ):
                # Prevent stale role values when the selected player's compatible
                # role set changes.
                del st.session_state["role_fit_detail_role"]

            detail_role = st.selectbox(
                "Role detail",
                selectable_roles,
                key="role_fit_detail_role",
            )
            detail = fit_detail[detail_role]

            st.subheader(detail_role)
            dcols = st.columns(4)
            dscore = detail["Role Score"]
            dpct = detail["Role Percentile"]
            dcov = detail["Coverage %"]
            dn = detail["Benchmark N"]
            dcols[0].metric("Role Score", f"{dscore:+.2f}" if pd.notna(dscore) else "—")
            dcols[1].metric("Role Percentile", f"{dpct:.0f}" if pd.notna(dpct) else "—")
            dcols[2].metric("Coverage", f"{dcov:.0f}%")
            dcols[3].metric("Benchmark N", f"{int(dn):,}")

            kpi_scores = detail.get("KPI Scores", {})
            if kpi_scores:
                st.markdown("#### KPI decomposition")
                kpi_cols = st.columns(min(4, len(kpi_scores)))
                for i, (kpi, score) in enumerate(kpi_scores.items()):
                    kpi_cols[i % len(kpi_cols)].metric(kpi, f"{score:+.2f}")

            rf_scale = st.radio(
                "Role detail scale",
                ["Percentile", "Z-score"],
                horizontal=True,
                key="role_fit_scale",
            )
            wheel = role_fit_detail_wheel(
                str(role_fit_player),
                detail_role,
                detail,
                rf_player_row,
                rf_scale,
            )
            st.plotly_chart(wheel, width="stretch")

            st.markdown("#### Underlying role metrics")
            metric_rows = []
            for kpi, requested_metric in SINGLE_PLAYER_PROFILES[detail_role]:
                rr, _ = resolve_metrics_aliases([requested_metric], role_fit_pool.columns.tolist())
                if not rr:
                    continue
                m = rr[0]
                raw = pd.to_numeric(pd.Series([rf_player_row.get(m, np.nan)]), errors="coerce").iloc[0]
                zval = detail.get("Player Z", pd.Series(dtype=float)).get(m, np.nan)
                bench_raw = detail.get("Benchmark Raw", pd.DataFrame())
                pctval = (
                    percentile_rank_against_population(
                        bench_raw[m], raw, lower_is_better=(m in LOWER_IS_BETTER)
                    )
                    if isinstance(bench_raw, pd.DataFrame) and m in bench_raw.columns else np.nan
                )
                metric_rows.append({
                    "KPI": kpi,
                    "Metric": single_metric_display_label(m),
                    "Raw Value": raw,
                    "Z-score": zval,
                    "Percentile": pctval,
                    "Weight %": DEFAULT_WEIGHTS[detail_role].get(requested_metric, 0),
                })
            metric_df = pd.DataFrame(metric_rows)
            if not metric_df.empty:
                metric_df["Raw Value"] = pd.to_numeric(metric_df["Raw Value"], errors="coerce").round(2)
                metric_df["Z-score"] = pd.to_numeric(metric_df["Z-score"], errors="coerce").round(2)
                metric_df["Percentile"] = pd.to_numeric(metric_df["Percentile"], errors="coerce").round(0)
                st.dataframe(metric_df, width="stretch", hide_index=True)

            st.caption(
                "Role Fit does not assign qualitative labels such as Excellent/Good/Poor. "
                "Role Percentile, Role Score, coverage, KPI decomposition and the metric wheel are shown separately."
            )
    else:
        st.info("Upload/filter player data to use Role Fit.")

elif app_page == "League Projection":
    st.header("🌍 League Projection")
    st.caption(
        "Estimate how a player's role performance may translate to another competition. "
        "V1 uses the Opta league season-average rating and the empirically calibrated β = 0.020. "
        "The player's underlying Wyscout metrics are not multiplied or altered."
    )

    league_strength = build_league_strength_lookup(
        tuple(sorted(df_all["League"].dropna().astype(str).unique().tolist())),
        str(OPTA_LEAGUE_META_PATH),
    )
    if league_strength.empty:
        st.error(
            "League Projection requires league-meta.json beside football_scout_app.py. "
            "Add the supplied Opta league metadata file to the repository."
        )
    else:
        # Player pool follows the scouting filters, while projection benchmarks
        # deliberately ignore team and age filters.
        projection_pool = filtered_base.copy()
        projection_pool = projection_pool.loc[projection_pool["League"].isin(set(league_strength["League"]))]
        if projection_pool.empty:
            st.info("No players in the current filters have a matched Opta league rating.")
        else:
            player_options = []
            player_index = {}
            for idx, row in projection_pool.iterrows():
                label = f"{row.get('Player', '')} — {row.get('Team', '')} | {row.get('League', '')}"
                # Keep duplicate labels selectable without relying on Player name as an ID.
                if label in player_index:
                    label = f"{label} [{idx}]"
                player_options.append(label)
                player_index[label] = idx

            c1, c2 = st.columns([1.35, 1])
            with c1:
                selected_player_label = st.selectbox("Player", player_options, key="projection_player")
            selected_idx = player_index[selected_player_label]
            player_row = projection_pool.loc[selected_idx]
            current_league = str(player_row["League"])
            current_match = league_strength.loc[league_strength["League"].eq(current_league)].iloc[0]

            compatible_roles = [
                r for r, eligible in PROFILE_POSITION_ELIGIBILITY.items()
                if str(player_row.get("Main Position", "")) in eligible
            ]
            if not compatible_roles:
                st.warning(f"No built-in projection role is mapped to Main Position {player_row.get('Main Position', '')}.")
            else:
                preferred = st.session_state.get("active_profile", {}) or {}
                preferred_role = preferred.get("profile_name")
                default_role_idx = compatible_roles.index(preferred_role) if preferred_role in compatible_roles else 0
                with c2:
                    projection_role = st.selectbox("Role", compatible_roles, index=default_role_idx, key="projection_role")

                # Target competitions are taken from the same season group(s) selected
                # in the sidebar, but do not need to be among the current league filters.
                target_labels = sorted(df_season["League"].dropna().astype(str).unique().tolist())
                target_strength = league_strength.loc[league_strength["League"].isin(target_labels)].copy()
                target_strength = target_strength.loc[~target_strength["League"].eq(current_league)]
                if target_strength.empty:
                    st.info("No other matched target competitions are available in the selected season group.")
                else:
                    target_display = {
                        f"{r.League}  ·  Opta {r.rating:.2f}": r.League
                        for r in target_strength.itertuples()
                    }
                    target_label = st.selectbox(
                        "Target league",
                        list(target_display.keys()),
                        key="projection_target_league",
                    )
                    target_league = target_display[target_label]
                    target_match = target_strength.loc[target_strength["League"].eq(target_league)].iloc[0]

                    eligible_positions = PROFILE_POSITION_ELIGIBILITY[projection_role]
                    current_benchmark = df_all.loc[
                        df_all["League"].eq(current_league)
                        & df_all["Main Position"].isin(eligible_positions)
                    ]
                    target_benchmark = df_all.loc[
                        df_all["League"].eq(target_league)
                        & df_all["Main Position"].isin(eligible_positions)
                    ]
                    if "Minutes played" in df_all.columns:
                        current_benchmark = current_benchmark.loc[current_benchmark["Minutes played"] >= min_minutes]
                        target_benchmark = target_benchmark.loc[target_benchmark["Minutes played"] >= min_minutes]

                    one_player = pd.DataFrame([player_row])
                    scored_player, current_distribution = role_score_for_population(
                        one_player, current_benchmark, projection_role, float(coverage_threshold_pct) / 100.0
                    )
                    _, target_distribution = role_score_for_population(
                        target_benchmark, target_benchmark, projection_role, float(coverage_threshold_pct) / 100.0
                    )
                    current_score = pd.to_numeric(scored_player["_projection_role_score"], errors="coerce").iloc[0]

                    current_rating = float(current_match["rating"])
                    target_rating = float(target_match["rating"])
                    rating_delta = target_rating - current_rating
                    projected_score = current_score - LEAGUE_TRANSLATION_BETA * rating_delta if pd.notna(current_score) else np.nan
                    current_pct = percentile_rank_against_population(current_distribution, current_score) if pd.notna(current_score) else np.nan
                    projected_pct = percentile_rank_against_population(target_distribution, projected_score) if pd.notna(projected_score) else np.nan

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Current Role Score", f"{current_score:+.2f}" if pd.notna(current_score) else "—")
                    m2.metric("Current Role Percentile", f"{current_pct:.0f}" if pd.notna(current_pct) else "—")
                    m3.metric("Projected Role Score", f"{projected_score:+.2f}" if pd.notna(projected_score) else "—", delta=f"{(projected_score-current_score):+.2f}" if pd.notna(projected_score) and pd.notna(current_score) else None)
                    m4.metric("Projected Target Percentile", f"{projected_pct:.0f}" if pd.notna(projected_pct) else "—")

                    left, right = st.columns(2)
                    with left:
                        st.subheader("Competition translation")
                        summary = pd.DataFrame([
                            {"Competition": "Current", "League": current_league, "Opta League": current_match["opta_name"], "Opta Rating": current_rating, "Global Rank": current_match["rank"]},
                            {"Competition": "Target", "League": target_league, "Opta League": target_match["opta_name"], "Opta Rating": target_rating, "Global Rank": target_match["rank"]},
                        ])
                        st.dataframe(summary, width="stretch", hide_index=True)
                    with right:
                        st.subheader("Projection")
                        direction = "stronger" if rating_delta > 0 else "weaker" if rating_delta < 0 else "equivalent"
                        st.write(f"**League-strength change:** {rating_delta:+.2f} Opta points ({direction} target).")
                        st.write(f"**Translation adjustment:** {-LEAGUE_TRANSLATION_BETA * rating_delta:+.2f} Role Score.")
                        st.write(f"**Target benchmark N:** {len(target_distribution):,} usable {projection_role} observations.")

                    # Current vs projected role-metric radar. The current shape is benchmarked
                    # against the player's present competition; the projected shape uses the actual
                    # target-league compatible-position population plus the calibrated league penalty.
                    current_profile, _ = build_single_player_profile(player_row, current_benchmark, projection_role)
                    projected_profile, _ = build_projected_target_profile(
                        player_row, target_benchmark, projection_role, rating_delta
                    )
                    if not current_profile.empty and not projected_profile.empty:
                        st.subheader("Current vs projected performance profile")
                        st.caption(
                            f"Percentile profile for {projection_role}. Current values are relative to {current_league}; "
                            f"projected values are relative to actual {target_league} players in compatible Main Positions, "
                            f"with the β = {LEAGUE_TRANSLATION_BETA:.3f} competition-strength adjustment applied."
                        )
                        fig = league_projection_radar(
                            current_profile, projected_profile, str(player_row.get("Player", "")),
                            current_league, target_league,
                        )
                        st.plotly_chart(fig, width="stretch")

                        radar_table = current_profile[["KPI Group", "Metric", "Percentile"]].rename(columns={"Percentile": "Current %ile"})
                        radar_table = radar_table.merge(
                            projected_profile[["Metric", "Percentile"]].rename(columns={"Percentile": "Projected %ile"}),
                            on="Metric", how="inner"
                        )
                        radar_table["Change"] = radar_table["Projected %ile"] - radar_table["Current %ile"]
                        for c in ["Current %ile", "Projected %ile", "Change"]:
                            radar_table[c] = radar_table[c].round(0)
                        with st.expander("Projected metric detail", expanded=False):
                            st.dataframe(radar_table, width="stretch", hide_index=True)

                    st.caption(
                        "V1 projection: Projected Role Score = Current Role Score − 0.020 × "
                        "(Target Opta rating − Current Opta rating). The coefficient was calibrated from "
                        "cross-league movers; it is an estimate, not a guarantee of future performance. "
                        "Projected percentile compares the translated score with the target league's actual role-score distribution."
                    )


st.markdown("---")
with st.expander("Methodology & score reliability", expanded=False):
    st.markdown(
        """
**Canonical roles.** Each built-in role is defined once as KPI group → metric → default weight.
The Profile Score, single-player wheel and multi-player wheel are derived from that same definition.

**Benchmarking.** Search filters determine which players you see. Benchmark controls determine who
those players are compared against. Team and age filters do not narrow the benchmark. The selected
league/season universe and minimum-minutes threshold still apply.

**Profile Score.** Metrics are standardized as z-scores against the selected benchmark. Metrics where
lower is better are direction-reversed. Default/custom weights must total exactly 100%.

**Missing data and coverage.** Missing player values are not treated as average. The score is calculated
from the available weighted metrics and the available weights are renormalized for that player. The
Coverage % column shows how much of the intended profile weight was actually observed. Scores below the
selected coverage threshold are suppressed.

**KPI sub-scores.** Built-in profiles also expose weighted direction-aware z-score sub-scores for each
KPI family. They use the same benchmark and the same metric weights as the overall Profile Score.

**Percentiles.** Percentile (0–100) is the default presentation scale because it is faster to interpret.
Higher always means better after direction reversal. Z-score remains available as an advanced analytical
view. Both use the same benchmark population; the weighted Profile Score remains z-score based.

**Visual colour.** KPI colours are categorical only. They identify KPI families and deliberately avoid a
traffic-light red/green/orange performance meaning. Performance is communicated by radial length and the
numeric percentile, not by colour.

**League Projection.** The V1 translation model keeps raw Wyscout metrics unchanged. It adjusts the
player's role score by the difference between the current and target Opta `seasonAverageRating`, using
β = 0.020 role-score units per Opta rating point. Projected percentile is then evaluated against the
target league's actual role-score distribution for the selected role. This is a translation estimate,
not a deterministic forecast.

**Sample size.** Small benchmark populations make both z-scores and percentiles less stable. The app
warns when the benchmark falls below the configured minimum sample size.
        """
    )

st.markdown("Developed with ❤️ using Streamlit & Plotly | Season-aware ✨")
