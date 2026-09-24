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



@st.cache_data(show_spinner=False)
def benchmark_metric_stats(benchmark_numeric: pd.DataFrame) -> pd.DataFrame:
    """Cache benchmark means/SDs/counts; input is already numeric and role-filtered."""
    return pd.DataFrame({
        "mean": benchmark_numeric.mean(axis=0),
        "std": benchmark_numeric.std(axis=0, ddof=0),
        "count": benchmark_numeric.count(axis=0),
    })


def benchmark_population_for_role(
    source_df: pd.DataFrame,
    role_name: str | None,
    mode: str,
    selected_positions: List[str],
) -> pd.DataFrame:
    """Benchmark is intentionally independent from age/team search filters."""
    bench = source_df.copy()
    if "Main Position" not in bench.columns:
        return bench
    if mode == "Role position" and role_name in PROFILES:
        return bench.loc[position_family_mask(bench["Main Position"], role_name)].copy()
    if mode == "Selected positions" and selected_positions:
        return bench.loc[bench["Main Position"].isin(selected_positions)].copy()
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

    palette = [
        "#4E79A7", "#59A14F", "#F28E2B", "#E15759",
        "#76B7B2", "#B07AA1", "#EDC948", "#9C755F",
    ]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    # Presentation aliases only; underlying Wyscout columns remain untouched.
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
    width = unit * 0.90

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

    # Neutral reference rings across the performance annulus.
    ring_theta = np.linspace(0, 360, 361)
    for rv in scale_r:
        fig.add_trace(
            go.Scatterpolar(
                r=[rv] * len(ring_theta),
                theta=ring_theta,
                mode="lines",
                line=dict(color="rgba(120,130,140,0.16)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Stronger neutral reference in z-score mode.
    if scale_mode == "Z-score":
        zero_r = perf_inner + 0.5 * perf_span
        fig.add_trace(
            go.Scatterpolar(
                r=[zero_r] * len(ring_theta),
                theta=ring_theta,
                mode="lines",
                line=dict(color="rgba(45,55,65,0.68)", width=2.8),
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
                line=dict(color=group_colors[group_seq[i]], width=3.5),
                opacity=0.62,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Performance markers: the primary quantitative encoding.
    marker_colors = [group_colors[g] for g in group_seq]
    fig.add_trace(
        go.Scatterpolar(
            r=marker_r,
            theta=theta,
            mode="markers",
            marker=dict(
                size=15,
                color=marker_colors,
                line=dict(color="white", width=2),
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
            textfont=dict(size=11, color="white"),
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
                tickfont=dict(size=11, color="#566270"),
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

    palette = [
        "#4E79A7", "#59A14F", "#F28E2B", "#E15759",
        "#76B7B2", "#B07AA1", "#EDC948", "#9C755F",
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
                line=dict(color="rgba(120,130,140,0.20)", width=1),
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
                line=dict(width=3),
                marker=dict(size=9, line=dict(color="white", width=1.4)),
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
        centre_text = "<b>Percentile</b><br><span style='font-size:11px'>50 = benchmark median</span>"
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
                tickfont=dict(size=13, color="#566270"),
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

benchmark_source_global = df_league.copy()
if "Minutes played" in benchmark_source_global.columns:
    benchmark_source_global = benchmark_source_global.loc[
        pd.to_numeric(benchmark_source_global["Minutes played"], errors="coerce") >= min_minutes
    ].copy()

# =========================
# Profile builder
# =========================
st.sidebar.header("Player profiles (z-score)")
if BUILTIN_PROFILE_FRAMEWORK_ISSUES:
    st.sidebar.error("Built-in profile configuration error: " + " | ".join(BUILTIN_PROFILE_FRAMEWORK_ISSUES))

if st.sidebar.button("Clear active profile"):
    st.session_state["active_profile"] = None
    st.rerun()

mode = st.sidebar.radio("Profile mode", ["Built-in", "Custom"], index=0, horizontal=True)

if mode == "Built-in":
    profile_name = st.sidebar.selectbox("Choose profile", list(PROFILES.keys()))
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

    if st.sidebar.button("Reset weights to defaults", key=f"reset::{profile_name}"):
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
        st.sidebar.markdown(f"**{kpi}**")
        for requested_metric in metric_weights:
            rr, _ = resolve_metrics_aliases([requested_metric], filtered_base.columns.tolist())
            if not rr:
                continue
            m = rr[0]
            default_w = int(DEFAULT_WEIGHTS[profile_name][requested_metric])
            slider_key = safe_widget_key("w", profile_name, m)
            if slider_key not in st.session_state:
                st.session_state[slider_key] = default_w
            w = st.sidebar.slider(f"{m}", 0, 100, int(st.session_state[slider_key]), 1, key=slider_key)
            weights_pct.append(int(w))

    total_weight = int(sum(weights_pct))
    st.sidebar.metric("Total profile weight", f"{total_weight}%")
    if total_weight != 100:
        st.sidebar.warning("Weights must total exactly 100% before the profile can be applied.")

    if missing_names:
        st.sidebar.caption("Unavailable metrics: " + ", ".join(missing_names))

    if st.sidebar.button(
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
        st.rerun()

else:
    custom_name = st.sidebar.text_input("Profile name", value="Custom Profile").strip() or "Custom Profile"
    custom_metrics = st.sidebar.multiselect(
        "Pick metrics to include",
        options=numeric_cols_base,
        default=numeric_cols_base[:5],
    )
    weights_pct = []
    if custom_metrics:
        default_pct = max(1, int(100 / len(custom_metrics)))
        for m in custom_metrics:
            weights_pct.append(
                int(st.sidebar.slider(
                    m, 0, 100, default_pct, 1,
                    key=safe_widget_key("w_custom", custom_name, m),
                ))
            )
    custom_total = int(sum(weights_pct))
    st.sidebar.metric("Total profile weight", f"{custom_total}%")
    if custom_metrics and custom_total != 100:
        st.sidebar.warning("Weights must total exactly 100% before the profile can be applied.")

    if st.sidebar.button(
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
active_benchmark_df = benchmark_source_global.copy()
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
    coverage_col = f"{calc_col_name} Coverage %"
    if coverage_col in filtered.columns:
        default_cols.append(coverage_col)
    default_cols.extend([c for c in active_kpi_cols if c in filtered.columns])

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
# Player comparison visualizations
# =========================
st.subheader("Player Comparison")

comparison_mode = st.radio(
    "Visualization mode",
    ["Multi-Player Radar", "Single-Player Profile"],
    horizontal=True,
    key="comparison_visualization_mode",
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

        st.dataframe(table_for_display, use_container_width=True)

        comparison_scale = st.radio(
            "Comparison scale",
            ["Z-score", "Percentile"],
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

        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption(
            "Multi-player comparison uses the same fixed-radius scouting-wheel architecture as the single-player profile: "
            "the same 15 metrics, KPI bands, metric order and KPI gaps. Switch between direction-aware Z-score (−2 to +2) "
            "and Percentile (0–100) using the same current filtered benchmark population. Hover always shows raw value, percentile and z-score. "
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
            help="Percentile shows 0–100 rank. Z-score shows standard deviations from the benchmark mean, displayed from −2 to +2, and is direction-aware.",
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
            st.plotly_chart(fig_single, use_container_width=True)

            profile_table = profile_df.copy()
            profile_table["Raw Value"] = profile_table["Raw Value"].round(2)
            profile_table["Percentile"] = profile_table["Percentile"].round(0).astype(int)
            profile_table["Z-score"] = profile_table["Z-score"].round(2)
            st.dataframe(profile_table, use_container_width=True, hide_index=True)

            csv_single = StringIO()
            profile_table.to_csv(csv_single, index=False)
            st.download_button(
                "⬇️ Download single-player profile (CSV)",
                data=csv_single.getvalue(),
                file_name=f"{safe_widget_key(single_player, single_role)}_percentile_profile.csv",
                mime="text/csv",
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

**Percentiles.** Percentile wheels use the same benchmark population as z-scores. Higher always means
better after direction reversal. Percentile and z-score views change the visual scale only; the weighted
Profile Score remains z-score based.

**Sample size.** Small benchmark populations make both z-scores and percentiles less stable. The app
warns when the benchmark falls below the configured minimum sample size.
        """
    )

st.markdown("Developed with ❤️ using Streamlit & Plotly | Season-aware ✨")