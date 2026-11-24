from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def load_reference(path):
    return pd.read_csv(path).drop_duplicates('Wyscout league (exact)').set_index('Wyscout league (exact)')

def render_recruitment_market(players, profiles, eligibility, score_fn, benchmark_fn, strength_fn, strength_path, club_fn, audit_path, snapshot_path, season_team, league_beta, club_beta):
    st.header('🎯 Recruitment Market Finder')
    st.caption('Club → Profile → Source tiers → Adjusted shortlist | Fixed Jenks v8.0 tiers')
    try:
        ref = load_reference(str(Path(__file__).with_name('wyscout_league_tiers_v8_reference.csv')))
    except (OSError, ValueError) as e:
        st.error(f'League reference unavailable: {e}'); return
    if players.empty:
        st.info('Select a season containing players.'); return
    tier = players['League'].map(ref['Sporting tier'])
    team_col = season_team if season_team in players.columns else 'Team'
    teams = players[team_col].fillna('').astype(str)
    clubs = players.loc[tier.notna() & teams.ne(''), ['League', team_col]].drop_duplicates()
    options = sorted(clubs.itertuples(index=False, name=None), key=lambda x:(str(x[1]).casefold(),str(x[0])))
    if not options:
        st.info('No rated clubs in the selected seasons.'); return
    left, right = st.columns([1,2])
    with left:
        target_league, target_club = st.selectbox('1. Recruiting club', options, format_func=lambda x:f'{x[1]} — {x[0]}', key='rm9_club')
        target_tier = int(ref.loc[target_league,'Sporting tier'])
        st.metric('Target tier',target_tier)
        role = st.selectbox('2. Player profile', list(profiles), key='rm9_role')
        depth = st.selectbox('Profile depth',['Core 5','Balanced 10','Complete 15'],index=2,key='rm9_depth')
        tiers = st.multiselect('3. Source tiers', list(range(1,11)), default=[t for t in range(1,11) if abs(t-target_tier)<=1],key='rm9_tiers')
        league_on = st.toggle('Apply league adjustment',value=True,key='rm9_league')
        club_on = st.toggle('Apply club adjustment',value=True,key='rm9_club_adj')
        age = st.slider('Age', 15, 60, (15,60),key='rm9_age')
        minutes = st.number_input('Minimum minutes',min_value=0,value=900,step=100,key='rm9_minutes')
        coverage = st.slider('Minimum profile coverage (%)',50,100,70,5,key='rm9_coverage')/100
        exclude = st.checkbox('Exclude recruiting club’s own players',value=True,key='rm9_exclude')
    with right:
        eligible = eligibility.get(role, [])
        if not eligible:
            st.warning('No position eligibility defined for this profile.'); return
        mins = pd.to_numeric(players['Minutes played'],errors='coerce')
        ages = pd.to_numeric(players['Age'],errors='coerce')
        # Stable role benchmark: all selected seasons, independent of source tiers and age.
        bench = benchmark_fn(players.loc[mins.ge(minutes)],role,'Role position',[])
        mask = tier.isin(tiers) & players['Main Position'].isin(eligible) & mins.ge(minutes) & ages.between(*age)
        if exclude: mask &= ~(players['League'].eq(target_league) & teams.eq(target_club))
        pool = players.loc[mask].copy()
        st.caption(f'Eligible positions: {", ".join(eligible)} | Role benchmark: {len(bench):,} rows across selected seasons')
        if pool.empty or bench.empty:
            st.info('No eligible candidates or benchmark rows. Broaden the filters.'); return
        if len(bench)<20: st.warning('Small role benchmark; interpret scores cautiously.')
        with st.spinner('Calculating existing profile scores…'):
            result,_ = score_fn(pool,bench,role,depth,coverage)
        result = result.rename(columns={'_shadow_role_score':'Profile Score'})
        result['Source tier'] = result['League'].map(ref['Sporting tier'])
        result['Tier difference'] = result['Source tier']-target_tier
        labels = tuple(sorted(set(result['League'].astype(str))|{target_league}))
        strength = strength_fn(labels,strength_path)
        ratings = strength.set_index('League')['rating'] if not strength.empty else pd.Series(dtype=float)
        target_rating = ratings.get(target_league,np.nan)
        result['Source Opta rating'] = result['League'].map(ratings)
        result['Opta difference'] = result['Source Opta rating']-target_rating
        st.caption(f'Target league Opta rating: {target_rating:.2f}' if pd.notna(target_rating) else 'Target league Opta rating unavailable.')
        terms=[]
        if league_on:
            result['League Adjustment'] = league_beta*result['Opta difference']; terms.append('League Adjustment')
        if club_on:
            result['Club Adjustment'] = np.nan
            if season_team in result.columns:
                try:
                    lookup=club_fn(audit_path,audit_path.stat().st_mtime_ns,snapshot_path,snapshot_path.stat().st_mtime_ns)
                    keys=pd.MultiIndex.from_frame(result[['League',season_team]])
                    result['Club Adjustment']=club_beta*lookup.reindex(keys).to_numpy()
                except (OSError,ValueError,KeyError) as e:
                    st.warning(f'Club adjustment data unavailable: {e}')
            else: st.warning('Season-specific team column unavailable; club adjustment cannot be verified.')
            terms.append('Club Adjustment')
        rank='Profile Score'
        if terms:
            rank='Adjusted Score'
            result[rank]=result['Profile Score']+result[terms].sum(axis=1,min_count=len(terms))
        result=result.sort_values(rank,ascending=False,na_position='last',kind='stable')
        a,b,c=st.columns(3); a.metric('Candidates',len(result)); b.metric('Source leagues',result['League'].nunique()); c.metric('Rankable',int(result[rank].notna().sum()))
        if result[rank].isna().any(): st.warning('Blank ranking scores indicate insufficient metric coverage or unavailable required Opta ratings. No missing adjustment is treated as zero.')
        columns=list(dict.fromkeys(['Player',team_col,'League','Main Position','Age','Minutes played','Source tier','Tier difference','Source Opta rating','Opta difference','Profile Score',*terms,rank]))
        columns=[c for c in columns if c in result]
        limit=st.slider('Rows to display',10,500,100,10,key='rm9_limit')
        st.dataframe(result[columns].head(limit).round(3),use_container_width=True,hide_index=True)
        st.download_button('Export recruitment shortlist (CSV)',result[columns].to_csv(index=False).encode('utf-8-sig'),'recruitment_shortlist.csv','text/csv',key='rm9_export')
        st.caption('League adjustment references the recruiting club’s league. Club adjustment references each source club’s own league average, using the existing verified snapshot rules. Both use the existing coefficients; no target-club rating term is added.')
