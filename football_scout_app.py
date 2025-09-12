import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    layout="wide",
    page_title="Advanced Football Scouting App",
    page_icon="⚽",
)

st.title("⚽ Advanced Football Player Scouting App — Improved")
st.markdown("Upload your football data CSV to analyze player metrics. Caching, robust parsing, ALL filters, drag‑and‑drop ordering, rounding to 2 decimals, downloads, and a radar chart included.")

# ---------------------------
# Utilities
# ---------------------------

def multiselect_all(label: str, options: list, *, default_all: bool = True, help: str | None = None, key: str | None = None):
    """A multiselect with an 'ALL' option that maps to all items.
    Returns the resolved selection list (without 'ALL') and whether ALL is active.
    """
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

PCT_SUFFIX = ", %"  # columns that end with this are percentages (e.g., "Accurate passes, %")


def parse_market_value(series: pd.Series) -> pd.Series:
    """Parse market value strings like '€12.5m', '€800k', '12,000,000' into float (EUR).
    Returns float in millions of EUR for easier sliders.
    """
    if series.dtype.kind in 'iuf':
        s = series.astype(float)
        if s.max() > 1e6:
            return s / 1e6
        return s

    def to_float(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace('€', '').replace(',', '').lower()
        mult = 1.0
        if s.endswith('m'):
            mult = 1_000_000.0
            s = s[:-1]
        elif s.endswith('k'):
            mult = 1_000.0
            s = s[:-1]
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
    # Try to coerce all other columns that look numeric or end with percentage marker
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
    mv_min, mv_max = float(filtered[mv_col].replace([np.inf, -np.inf], np.nan).dropna().min()), float(filtered[mv_col].replace([np.inf, -np.inf], np.nan).dropna().max())
    if np.isfinite(mv_min) and np.isfinite(mv_max):
        mv_range = st.sidebar.slider(
            "Market value range" + (" (M€)" if mv_col == 'Market value (M€)' else ""),
            float(np.floor(mv_min)), float(np.ceil(mv_max)), (float(np.floor(mv_min)), float(np.ceil(mv_max)))
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
# Data table (select columns)
# ---------------------------
st.subheader("Filtered Player Data")

# Drag & drop helper (optional community components)

def reorder_pills(items: list, *, key: str, direction: str = "horizontal") -> list:
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
            st.caption("💡 Install `streamlit-sortable` to enable drag-and-drop reordering.")
            return items

default_cols = [c for c in [
    'Player', 'Team', 'League', 'Main Position', 'Age',
    'Market value (M€)' if 'Market value (M€)' in filtered.columns else 'Market value',
    'Goals', 'Assists', 'xG', 'xA', 'Minutes played'
] if c in filtered.columns]

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
    st.dataframe(
        filtered[ordered_display_cols].sort_values(by="Player").reset_index(drop=True),
        use_container_width=True,
    )
else:
    st.info("Please select at least one column to display.")

# Download filtered data
csv_buf = StringIO()
filtered[ordered_display_cols or default_cols].to_csv(csv_buf, index=False)
st.download_button("⬇️ Download filtered data (CSV)", data=csv_buf.getvalue(), file_name="filtered_players.csv", mime="text/csv")

# ---------------------------
# Key Metrics Averages
# ---------------------------
st.subheader("Key Metrics Averages (Filtered Players)")
num_cols = get_numeric_columns(filtered)
# Exclude obvious filters from averages
exclude_avg = {'Age', 'Minutes played', 'Matches played'}
metric_choices = [c for c in num_cols if c not in exclude_avg]

default_avg = [c for c in ['Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90', 'Accurate passes, %', 'Duels won, %'] if c in metric_choices]

selected_avg_metrics = st.multiselect(
    "Metrics for average summary",
    options=metric_choices,
    default=default_avg if default_avg else metric_choices[:4]
)

ordered_avg_metrics = reorder_pills(selected_avg_metrics, key="order_avg_metrics")

if selected_avg_metrics:
    n_cols = min(4, len(selected_avg_metrics))
    cols = st.columns(n_cols)
    for i, m in enumerate(ordered_avg_metrics):
        val = filtered[m].mean().round(2)
        suffix = "%" if m.endswith(PCT_SUFFIX) else ""
        cols[i % n_cols].metric(f"Avg {m}", f"{val:.2f}{suffix}")
else:
    st.info("Select metrics above to see averages.")

# ---------------------------
# Scatter plot
# ---------------------------
st.subheader("Player Performance Visualization")
plot_metrics = [c for c in num_cols if c not in {'Age', 'Market value'}]

# Helpful defaults
x_default = 'Goals per 90' if 'Goals per 90' in plot_metrics else (plot_metrics[0] if plot_metrics else None)
y_default = 'Assists per 90' if 'Assists per 90' in plot_metrics else (plot_metrics[1] if len(plot_metrics) > 1 else x_default)

if x_default is None or y_default is None:
    st.warning("No numerical metrics available for plotting.")
else:
    c1, c2 = st.columns(2)
    with c1:
        x_axis = st.selectbox("X-axis", plot_metrics, index=plot_metrics.index(x_default))
    with c2:
        y_axis = st.selectbox("Y-axis", plot_metrics, index=plot_metrics.index(y_default))

    color_by = st.selectbox("Color by", options=[o for o in ['Main Position', 'Team', 'League', 'Foot', 'None'] if o == 'None' or o in filtered.columns], index=0)
    size_by = st.selectbox("Size by", options=[o for o in ['None', 'Minutes played', 'Market value (M€)', 'Age', 'Matches played'] if o == 'None' or o in filtered.columns], index=1)

    plot_df = filtered.copy()
    if remove_outliers:
        # z-score on selected axes
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
# Player comparison + Radar chart
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
    default_comp = [c for c in ['Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90', 'Successful defensive actions per 90', 'Duels won, %'] if c in comp_metric_choices]

    comp_metrics = st.multiselect(
        "Metrics for comparison table & radar",
        options=comp_metric_choices,
        default=default_comp if default_comp else comp_metric_choices[:6]
    )

    if comp_metrics:
        ordered_comp_metrics = reorder_pills(comp_metrics, key="order_comp_metrics")
        comp_df = comp_df.round(2)
        st.dataframe(comp_df[ordered_comp_metrics].transpose().round(2).style.format("{:.2f}").highlight_max(axis=1, color='#C8E6C9'), use_container_width=True)

        # Download comparison
        csv_buf2 = StringIO()
        comp_df[ordered_comp_metrics].to_csv(csv_buf2)
        st.download_button("⬇️ Download comparison (CSV)", data=csv_buf2.getvalue(), file_name="player_comparison.csv", mime="text/csv")

        # Radar chart (normalize across the filtered set for comparability)
        # Min-max per metric using only players currently filtered (not just selected)
        mm_base = filtered.set_index('Player')
        mm = {}
        for m in ordered_comp_metrics:
            series = mm_base[m].replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                mm[m] = (0.0, 1.0)
            else:
                lo, hi = float(series.min()), float(series.max())
                if hi - lo < 1e-9:
                    hi = lo + 1.0
                mm[m] = (lo, hi)

        def scale(val, lo, hi):
            return float((val - lo) / (hi - lo))

        theta = ordered_comp_metrics
        fig_radar = go.Figure()
        for player in compare_players:
            row = comp_df.loc[player, ordered_comp_metrics]
            r = [scale(float(row[m]), *mm[m]) if pd.notna(row[m]) else 0.0 for m in ordered_comp_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill='toself',
                name=player,
                text=[f"{player}: {val:.2f}" for val in r] + [f"{player}: {r[0]:.2f}"],
                hoverinfo="text"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
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
