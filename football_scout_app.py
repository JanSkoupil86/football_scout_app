import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    layout="wide",
    page_title="Advanced Football Scouting App",
    page_icon="⚽",
)

st.title("⚽ Advanced Football Player Scouting App — Improved")
st.markdown(
    "Upload your football data CSV to analyze player metrics. "
    "Caching, robust parsing, ALL filters, drag-and-drop ordering, rounding to 2 decimals, "
    "downloads, a radar chart (z-score standardized), and calculated player profile scores (incl. Custom Profile)."
)

# ---------------------------
# Utilities
# ---------------------------

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
    # Try utf-8, fallback to latin-1
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding="latin-1")

# Known non-feature columns we generally don't treat as numeric
NON_FEATURE_COLUMNS = {
    'Column1', 'Player', 'Team', 'Team within selected timeframe', 'Position', 'Birth country',
    'Passport country', 'Foot', 'On loan', 'Contract expires', 'League', 'Main Position'
}

PCT_SUFFIX = ", %"  # dataset columns use a comma before %

def parse_market_value(series: pd.Series) -> pd.Series:
    """Parse market value strings into float (millions of EUR)."""
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
    # Special-case market value
    if 'Market value' in df.columns:
        df['Market value (M€)'] = parse_market_value(df['Market value'])
    # Coerce other numerics
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

def _zscore(series: pd.Series) -> pd.Series:
    """Z-score with safe zero-std handling."""
    m = series.mean()
    s = series.std(ddof=0)
    if pd.isna(s) or s == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - m) / s

def normalize_weights(pcts: np.ndarray) -> np.ndarray:
    if pcts.sum() == 0:
        return np.ones_like(pcts) / len(pcts)
    return pcts / pcts.sum()

def make_profile_score(df: pd.DataFrame, metrics: list[str], weights: np.ndarray, new_col: str) -> pd.DataFrame:
    """
    Add a weighted z-score composite column to df for the given metrics.
    Uses z-score across the CURRENT df (filtered view).
    Only uses present metrics; weights align to those.
    """
    present = [m for m in metrics if m in df.columns]
    if not present:
        df[new_col] = 0.0
        return df

    # Align weights (already normalized to sum=1 before calling)
    w_map = {m: w for m, w in zip(metrics, weights)}
    used_weights = np.array([w_map[m] for m in present], dtype=float)
    used_weights = normalize_weights(used_weights)

    z_cols = []
    for m in present:
        z = _zscore(df[m].astype(float))
        z_cols.append(z.values.reshape(-1, 1))
    Z = np.hstack(z_cols)  # (n_players, n_metrics)

    score = (Z * used_weights.reshape(1, -1)).sum(axis=1)
    df[new_col] = np.round(score, 2)
    return df

# -------- Metric alias resolver (e.g., "Save rate %" -> "Save rate, %") --------
def _norm(s: str) -> str:
    return re.sub(r'[\s,%%]+', '', s).lower()

def resolve_metrics_aliases(requested: list[str], columns: list[str]) -> tuple[list[str], list[str]]:
    """
    Try to resolve requested metric names to existing df column names.
    Returns (resolved_list, missing_list).
    Rules:
      - exact match
      - try inserting ', %' before % if missing
      - fallback to normalization-based match ignoring spaces/commas/% and case
    """
    col_norm_map = {_norm(c): c for c in columns}
    resolved, missing = [], []
    for name in requested:
        if name in columns:
            resolved.append(name); continue
        alt = name.replace(' %', ', %').replace('%', ', %') if '%' in name and ', %' not in name else name
        if alt in columns:
            resolved.append(alt); continue
        key = _norm(name)
        if key in col_norm_map:
            resolved.append(col_norm_map[key]); continue
        missing.append(name)
    return resolved, missing

# ---------------------------
# Sidebar — File uploader
# ---------------------------
uploaded = st.sidebar.file_uploader("Upload your Football Data CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload your football data CSV to begin. Include columns like 'Player', 'Team', 'Main Position', 'Age', 'Goals per 90', etc.")
    st.stop()

# Load & clean
try:
    df_raw = load_csv(uploaded)
except pd.errors.EmptyDataError:
    st.error("The uploaded CSV file is empty.")
    st.stop()
except pd.errors.ParserError:
    st.error("Could not parse the CSV file. Please ensure it is a valid CSV.")
    st.stop()

# Basic sanity checks
required_cols = ['Player', 'Team', 'Main Position', 'Age', 'League']
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing critical column(s): {', '.join(missing)}. Please check your CSV.")
    st.stop()

# Drop 'Column1' if it looks like an index
if 'Column1' in df_raw.columns and df_raw['Column1'].nunique() == len(df_raw):
    df_raw = df_raw.drop(columns=['Column1'])

# Coerce types
df = coerce_numeric(df_raw)

# Fill NaNs in numeric columns and round to 2 decimals for consistent display & exports
for col in get_numeric_columns(df):
    df[col] = df[col].fillna(0).round(2)

# ---------------------------
# Sidebar — Filters
# ---------------------------
st.sidebar.header("Filters")

# League first
leagues = sorted(df['League'].dropna().unique().tolist())
selected_leagues, _ = multiselect_all("League(s)", leagues, default_all=True, help="Choose specific leagues or use ALL")
filtered = df[df['League'].isin(selected_leagues)].copy()

if filtered.empty:
    st.warning("No players found for selected leagues. Adjust filters.")
    st.stop()

# Teams and positions depend on league filter
teams = sorted(filtered['Team'].dropna().unique().tolist())
positions = sorted(filtered['Main Position'].dropna().unique().tolist())

selected_teams, _ = multiselect_all("Team(s)", teams, default_all=True, help="Pick teams or ALL")
selected_positions, _ = multiselect_all("Main Position(s)", positions, default_all=True, help="Pick positions or ALL")

# Age slider
age_min, age_max = int(filtered['Age'].min()), int(filtered['Age'].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

# Market value slider — use parsed 'Market value (M€)' if available, else fallback
if 'Market value (M€)' in filtered.columns:
    mv_col = 'Market value (M€)'
else:
    mv_col = 'Market value' if 'Market value' in filtered.columns else None

if mv_col is not None:
    mv_min = float(filtered[mv_col].replace([np.inf, -np.inf], np.nan).dropna().min())
    mv_max = float(filtered[mv_col].replace([np.inf, -np.inf], np.nan).dropna().max())
    if np.isfinite(mv_min) and np.isfinite(mv_max):
        mv_range = st.sidebar.slider(
            "Market value range" + (" (M€)" if mv_col == 'Market value (M€)' else ""),
            float(np.floor(mv_min)), float(np.ceil(mv_max)),
            (float(np.floor(mv_min)), float(np.ceil(mv_max)))
        )
    else:
        mv_col = None  # disable if bad values

# Minutes played threshold
if 'Minutes played' in filtered.columns:
    min_minutes_max = int(filtered['Minutes played'].max())
    min_minutes = st.sidebar.slider("Minimum minutes played", 0, max(0, min_minutes_max), min(500, max(0, min_minutes_max)))
else:
    min_minutes = 0

# Optional: outlier removal using z-score on selected plotting metrics later
remove_outliers = st.sidebar.checkbox("Remove outliers (Z-score > 3)", value=False, help="Applied to the selected X/Y metrics only")

# Apply remaining filters
mask = (
    filtered['Team'].isin(selected_teams) &
    filtered['Main Position'].isin(selected_positions) &
    (filtered['Age'].between(age_range[0], age_range[1]))
)
if mv_col is not None:
    mask &= filtered[mv_col].between(mv_range[0], mv_range[1])
if 'Minutes played' in filtered.columns:
    mask &= filtered['Minutes played'] >= min_minutes

filtered = filtered.loc[mask].copy()

# Round again after filtering (safe if transforms introduced decimals)
for col in get_numeric_columns(filtered):
    filtered[col] = filtered[col].round(2)

st.sidebar.markdown(f"**Players matching filters: {len(filtered)}**")

if filtered.empty:
    st.warning("No players match the selected filters. Please adjust your criteria.")
    st.stop()

# ---------------------------
# Player Profiles (Calculated Fields) — Built-ins + Custom Profile
# ---------------------------
calc_col_name = None

# Built-in goalkeeper profiles (keep)
PROFILES = {
    "Classic Goalkeeper": [
        "Save rate %",                # dataset: "Save rate, %"
        "Prevented goals per 90",
        "Conceded goals per 90",
        "Exits per 90",
        "Aerial duels won %",         # dataset: "Aerial duels won, %"
        "Accurate long passes %",     # dataset: "Accurate long passes, %"
    ],
    "Sweeper Keeper": [
        "Exits per 90",
        "Passes per 90",
        "Accurate passes %",          # dataset: "Accurate passes, %"
        "Progressive passes per 90",
        "Forward passes per 90",
        "Accurate forward passes %",  # dataset: "Accurate forward passes, %"
        "Accurate long passes %",     # dataset: "Accurate long passes, %"
    ],
    "All-Round Keeper": [
        "Prevented goals per 90",
        "Save rate %",                # dataset: "Save rate, %"
        "Exits per 90",
        "Aerial duels won %",         # dataset: "Aerial duels won, %"
        "Passes per 90",
        "Accurate passes %",          # dataset: "Accurate passes, %"
        "Accurate long passes %",     # dataset: "Accurate long passes, %"
    ]
}

with st.sidebar.expander("Player profiles (calculated z-score)", expanded=True):
    st.caption("Scores are weighted sums of z-scored metrics across the currently filtered players.")
    mode = st.radio("Profile mode", ["Built-in", "Custom"], index=1, horizontal=True)

    if mode == "Built-in":
        profile = st.selectbox("Choose profile", list(PROFILES.keys()))
        requested_metrics = PROFILES[profile]
        resolved_metrics, missing = resolve_metrics_aliases(requested_metrics, filtered.columns.tolist())
        if missing:
            st.info("Some profile metrics were not found in this dataset and will be skipped: " + ", ".join(missing))

        if resolved_metrics:
            # Equal default percentages
            default_pct = max(1, int(100 / len(resolved_metrics)))
            weights_pct = []
            for i, m in enumerate(resolved_metrics, start=1):
                weights_pct.append(st.slider(f"Weight %: {m}", 0, 100, default_pct, 1, key=f"w_{profile}_{i}"))
            weights_pct = np.array(weights_pct, dtype=float)
            weights = normalize_weights(weights_pct)
            calc_col_name = f"Score: {profile}"
            filtered = make_profile_score(filtered, resolved_metrics, weights, calc_col_name)
            st.caption(f"✅ Added column **{calc_col_name}** to the dataset.")
        else:
            st.warning("No valid metrics for this profile in the current dataset.")

    else:
        # Custom Profile Builder
        st.subheader("Custom Profile")
        custom_name = st.text_input("Profile name", value="Custom Profile")
        numeric_cols = get_numeric_columns(filtered)
        # let user pick any metrics
        custom_metrics = st.multiselect("Pick metrics to include", options=numeric_cols, default=numeric_cols[:5])
        if custom_metrics:
            default_pct = max(1, int(100 / len(custom_metrics)))
            weights_pct = []
            for i, m in enumerate(custom_metrics, start=1):
                weights_pct.append(st.slider(f"Weight %: {m}", 0, 100, default_pct, 1, key=f"w_custom_{i}"))
            weights_pct = np.array(weights_pct, dtype=float)
            weights = normalize_weights(weights_pct)

            calc_col_name = f"Score: {custom_name}"
            filtered = make_profile_score(filtered, custom_metrics, weights, calc_col_name)
            st.caption(f"✅ Added column **{calc_col_name}** to the dataset.")
        else:
            st.info("Select at least one metric to build a custom profile.")

# ---------------------------
# Data table (select columns)
# ---------------------------
st.subheader("Filtered Player Data")

num_cols_for_rank = get_numeric_columns(filtered)
_default_rank = next(
    (m for m in ['Assists per 90','Goals per 90','xA per 90','xG per 90','xA','xG', calc_col_name] if m and m in filtered.columns),
    (num_cols_for_rank[0] if num_cols_for_rank else 'Minutes played')
)
rank_metric = st.session_state.get('rank_metric', _default_rank)

def reorder_pills(items: list, *, key: str, direction: str = "horizontal") -> list:
    """Try optional drag-and-drop libs; fall back silently."""
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
            return items  # silent fallback

default_cols = [c for c in [
    'Player', 'Team', 'League', 'Main Position', 'Age',
    'Market value (M€)' if 'Market value (M€)' in filtered.columns else 'Market value',
    'Goals', 'Assists', 'xG', 'xA', 'Minutes played',
    calc_col_name if calc_col_name and calc_col_name in filtered.columns else None
] if c and c in filtered.columns]

# Build selection options (exclude redundant/raw columns when parsed versions exist)
exclude_cols = set()
if 'Market value (M€)' in filtered.columns:
    exclude_cols.add('Market value')

display_options = [c for c in filtered.columns if c not in exclude_cols]
selected_display_cols = st.multiselect(
    "Columns to display",
    options=display_options,
    default=default_cols,
)

ordered_display_cols = reorder_pills(selected_display_cols, key="order_display_cols")

if selected_display_cols:
    # Allow user to choose which metric to rank rows by
    rank_by = st.selectbox("Sort Top-N rows by", options=num_cols_for_rank,
                           index=num_cols_for_rank.index(rank_metric) if rank_metric in num_cols_for_rank else 0)
    row_limit = st.slider(f"Number of rows to show (Top-N by {rank_by})", 1, 30, 15)
    rank_metric = rank_by
    st.dataframe(
        filtered.sort_values(by=rank_metric, ascending=False)[ordered_display_cols]
            .reset_index(drop=True)
            .head(row_limit),
        use_container_width=True,
    )
else:
    st.info("Please select at least one column to display.")

# Download filtered data
csv_buf = StringIO()
filtered[ordered_display_cols or default_cols].to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered data (CSV)", data=csv_buf.getvalue(),
                   file_name="filtered_players.csv", mime="text/csv")

# ---------------------------
# Scatter plot
# ---------------------------
st.subheader("Player Performance Visualization")
# Recompute numeric columns (if profile added)
num_cols = get_numeric_columns(filtered)
plot_metrics = [c for c in num_cols if c not in {'Age', 'Market value'}]

# Helpful defaults
x_default = 'Goals per 90' if 'Goals per 90' in plot_metrics else (plot_metrics[0] if plot_metrics else None)
# Prefer calculated score as Y if available
if calc_col_name and calc_col_name in plot_metrics:
    y_default = calc_col_name
else:
    y_default = 'Assists per 90' if 'Assists per 90' in plot_metrics else (plot_metrics[1] if len(plot_metrics) > 1 else x_default)

if x_default is None or y_default is None:
    st.warning("No numerical metrics available for plotting.")
else:
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-axis", plot_metrics, index=plot_metrics.index(x_default))
    with c2:
        y_axis = st.selectbox("Y-axis", plot_metrics, index=plot_metrics.index(y_default))

    color_by = st.selectbox(
        "Color by",
        options=[o for o in ['Main Position', 'Team', 'League', 'Foot', 'None'] if o == 'None' or o in filtered.columns],
        index=0
    )
    size_by = st.selectbox(
        "Size by",
        options=[o for o in ['None', 'Minutes played', 'Market value (M€)', 'Age', 'Matches played'] if o == 'None' or o in filtered.columns],
        index=1
    )

    # Choose whether to rank Top-N by X or Y axis
    rank_axis = st.radio("Sort Top-N players by", ["X-axis", "Y-axis"], index=1, horizontal=True)
    sort_metric = y_axis if rank_axis == "Y-axis" else x_axis

    # Limit how many players are rendered in the chart
    plot_limit = st.slider(f"Number of players to plot (Top-N by {sort_metric})",
                           1, min(30, len(filtered)), min(15, len(filtered)))
    plot_df = filtered.sort_values(by=sort_metric, ascending=False).head(plot_limit).copy()
    if remove_outliers:
        # z-score on selected axes (based on currently plotted subset)
        for ax in [x_axis, y_axis]:
            if plot_df[ax].std(ddof=0) > 0:
                z = (plot_df[ax] - plot_df[ax].mean()) / plot_df[ax].std(ddof=0)
                plot_df = plot_df[np.abs(z) <= 3]

    # Ensure the plotted columns are rounded to exactly 2 decimals
    plot_df[x_axis] = plot_df[x_axis].round(2)
    plot_df[y_axis] = plot_df[y_axis].round(2)

    # Option to always show labels without hovering
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

# ---------------------------
# Player comparison + Radar chart (z-score normalized)
# ---------------------------
st.subheader("Compare Selected Players")
compare_players = st.multiselect(
    "Players to compare (max 5 recommended)",
    options=sorted(filtered['Player'].dropna().unique().tolist()),
    default=[]
)

if compare_players:
    comp_df = filtered[filtered['Player'].isin(compare_players)].set_index('Player')

    comp_metric_choices = [c for c in num_cols if not c.endswith(PCT_SUFFIX)]
    default_comp = [c for c in [
        'Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90',
        'Successful defensive actions per 90', 'Duels won, %'
    ] if c in comp_metric_choices]

    comp_metrics = st.multiselect(
        "Metrics for comparison table & radar",
        options=comp_metric_choices,
        default=default_comp if default_comp else comp_metric_choices[:6]
    )

    if comp_metrics:
        # Optional drag reorder
        def reorder_pills_inner(items: list, *, key: str, direction: str = "horizontal") -> list:
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
        ordered_comp_metrics = reorder_pills_inner(comp_metrics, key="order_comp_metrics")
        comp_df = comp_df.round(2)

        st.dataframe(
            comp_df[ordered_comp_metrics].transpose().round(2).style.format("{:.2f}").highlight_max(axis=1, color='#C8E6C9'),
            use_container_width=True
        )

        # Download comparison CSV
        csv_buf2 = StringIO()
        comp_df[ordered_comp_metrics].to_csv(csv_buf2)
        st.download_button("⬇️ Download comparison (CSV)", data=csv_buf2.getvalue(),
                           file_name="player_comparison.csv", mime="text/csv")

        # Radar chart — z-score across the CURRENT filtered set
        mm_base = filtered.set_index('Player')  # pool for stats equals the "Players matching filters" set

        # Mean and std for each metric across filtered players
        metric_means = {m: mm_base[m].mean() for m in ordered_comp_metrics}
        metric_stds  = {m: mm_base[m].std(ddof=0) for m in ordered_comp_metrics}

        def zscore(val, mean, std):
            if pd.isna(val) or pd.isna(mean) or pd.isna(std) or std == 0:
                return 0.0
            return float((val - mean) / std)

        theta = ordered_comp_metrics
        fig_radar = go.Figure()
        for player in compare_players:
            row = comp_df.loc[player, ordered_comp_metrics]
            r = [zscore(float(row[m]), metric_means[m], metric_stds[m]) if pd.notna(row[m]) else 0.0
                 for m in ordered_comp_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill='toself',
                name=player,
                text=[f"{player}: z={val:.2f}" for val in r] + [f"{player}: z={r[0]:.2f}"],
                hoverinfo="text"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),  # clip to ±3σ
            showlegend=True,
            template='plotly_white',
            height=640
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select metrics to compare players.")
else:
    st.info("Select players above to compare their stats and see a radar chart.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Plotly | Enhanced edition ✨")
