"""Recruitment Market Finder, integrated into Wyscout scouting workspace.
Only enriches/filter rows; never changes score or translation calculations.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _load_tiers(path: str):
    d = pd.read_csv(path)
    d['Sporting tier'] = pd.to_numeric(d['Sporting tier'], errors='coerce').astype('Int64')
    d['Opta rating'] = pd.to_numeric(d['Opta rating'], errors='coerce')
    return d.drop_duplicates('Wyscout league (exact)')

def render_recruitment_market(players: pd.DataFrame, scored_players: pd.DataFrame | None = None):
    st.header('🎯 Recruitment Market Finder')
    st.caption('Fixed Wyscout-only Jenks 10-tier classification v8.0. Tiers organize recruitment markets; they do not rate individual player quality.')
    ref_path = Path(__file__).resolve().parent / 'wyscout_league_tiers_v8_reference.csv'
    if not ref_path.exists():
        st.error('Missing wyscout_league_tiers_v8_reference.csv beside the app.'); return
    ref = _load_tiers(str(ref_path))
    needed = {'League','Player','Team','Main Position'}
    if not needed.issubset(players.columns):
        st.error('Missing Wyscout columns: ' + ', '.join(sorted(needed-set(players.columns)))); return
    # Keep the app's current season selection, but do not inherit position/team
    # filters that could accidentally suppress recruitment markets.
    work = players.copy()
    idx = ref.set_index('Wyscout league (exact)')
    for c, out in [('Sporting tier','Source tier'), ('Opta rating','Source Opta rating'), ('Competition type','Competition type')]:
        work[out] = work['League'].map(idx[c])
    if scored_players is not None:
        score_cols = [c for c in scored_players.columns if c not in work.columns and ('score' in c.lower() or 'fit' in c.lower()) and pd.api.types.is_numeric_dtype(scored_players[c])]
        if score_cols:
            for c in score_cols:
                work[c] = scored_players[c].reindex(work.index)
    destinations = (ref.loc[ref['Sporting tier'].notna()]
                    .sort_values(['Sporting tier','Opta rating'], ascending=[True,False])
                    .drop_duplicates('Wyscout competition (season removed)'))
    labels = destinations['Wyscout competition (season removed)'].tolist()
    if not labels:
        st.error('No rated destination competitions found.'); return
    left, right = st.columns([1,2])
    with left:
        target = st.selectbox('Destination league', labels, key='rm_destination')
        row = destinations.loc[destinations['Wyscout competition (season removed)'].eq(target)].iloc[0]
        target_tier = int(row['Sporting tier']); target_rating = float(row['Opta rating'])
        st.metric('Target tier', target_tier)
        st.metric('Target Opta rating', f'{target_rating:.2f}')
        defaults = [t for t in range(1,11) if abs(t-target_tier)<=2]
        tiers = st.multiselect('Source tiers', list(range(1,11)), default=defaults, key='rm_source_tiers')
        positions = st.multiselect('Main Position', sorted(work['Main Position'].dropna().astype(str).unique()), key='rm_positions')
        ages = pd.to_numeric(work['Age'], errors='coerce') if 'Age' in work else pd.Series(dtype=float)
        age_range = None
        if ages.notna().any():
            amin, amax = int(ages.min()), int(ages.max())
            if amin < amax:
                age_range = st.slider('Age', amin, amax, (amin,amax), key='rm_age')
        minutes = st.number_input('Minimum minutes', min_value=0, value=900, step=100, key='rm_minutes')
        available_leagues = sorted(work.loc[work['Source tier'].isin(tiers), 'League'].dropna().unique())
        source_leagues = st.multiselect('Source leagues (optional)', available_leagues, key='rm_leagues')
    subset = work.loc[work['Source tier'].isin(tiers)].copy()
    if positions: subset = subset.loc[subset['Main Position'].isin(positions)]
    if age_range is not None: subset = subset.loc[pd.to_numeric(subset['Age'],errors='coerce').between(*age_range)]
    if 'Minutes played' in subset: subset = subset.loc[pd.to_numeric(subset['Minutes played'],errors='coerce').ge(minutes)]
    if source_leagues: subset = subset.loc[subset['League'].isin(source_leagues)]
    subset['Tier difference (source - target)'] = pd.to_numeric(subset['Source tier'],errors='coerce') - target_tier
    subset['Opta difference (source - target)'] = pd.to_numeric(subset['Source Opta rating'],errors='coerce') - target_rating
    subset['Market relationship'] = subset['Tier difference (source - target)'].map(lambda v: 'Same tier' if v == 0 else ('Stronger source league' if v < 0 else 'Weaker source league'))
    with right:
        a,b,c = st.columns(3)
        a.metric('Player rows', len(subset)); b.metric('Source leagues', subset['League'].nunique()); c.metric('Target tier', target_tier)
        st.caption('Negative tier difference = stronger source tier. Positive Opta difference = stronger source league.')
        scores = [c for c in subset.columns if ('score' in c.lower() or 'fit' in c.lower()) and pd.api.types.is_numeric_dtype(subset[c])]
        selected_score = st.selectbox('Optional existing score for sorting', ['None']+scores, key='rm_score_sort')
        if selected_score != 'None': subset = subset.sort_values(selected_score, ascending=False, kind='stable')
        else: subset = subset.sort_values(['Opta difference (source - target)','Minutes played'] if 'Minutes played' in subset else ['Opta difference (source - target)'], ascending=False, kind='stable')
        columns = ['Player','Team','League','Main Position','Age','Minutes played','Source tier','Source Opta rating','Tier difference (source - target)','Opta difference (source - target)','Market relationship']
        columns = [c for c in columns if c in subset.columns]
        if selected_score != 'None': columns.append(selected_score)
        display_limit = st.slider('Rows to display', 10, 500, 100, 10, key='rm_limit')
        st.dataframe(subset[columns].head(display_limit), use_container_width=True, hide_index=True)
        st.download_button('⬇️ Export recruitment shortlist (CSV)', subset[columns].to_csv(index=False).encode('utf-8-sig'), 'recruitment_shortlist.csv', 'text/csv', key='rm_export')
    with st.expander('Source market overview'):
        if not subset.empty:
            summary = (subset.groupby(['League','Source tier'], dropna=False)
                       .agg(Player_rows=('Player','size'), Opta_rating=('Source Opta rating','first'))
                       .reset_index().sort_values(['Source tier','Opta_rating'],ascending=[True,False]))
            st.dataframe(summary,use_container_width=True,hide_index=True)
    missing = work.loc[work['Source tier'].isna(), 'League'].dropna().unique()
    if len(missing):
        st.info(f'{len(missing)} competition labels are unrated or separately classified and excluded from tier filtering.')
    st.caption('Opta league ratings are a fixed snapshot, not season-specific. Tier classifications do not predict player adaptation. Existing player scoring and adjustment formulas are unchanged.')
