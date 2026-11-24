from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import hashlib

# Plotly is imported only when the chart is requested.

@st.cache_data(show_spinner=False, max_entries=2)
def load_reference(path):
    return pd.read_csv(path).drop_duplicates('Wyscout league (exact)').set_index('Wyscout league (exact)')

def render_recruitment_market(players, profiles, eligibility, score_fn, benchmark_fn, strength_fn, strength_path, club_fn, audit_path, snapshot_path, season_team, league_beta, club_beta, dataset_key=None):
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
        max_value = None
        if 'Market value (M€)' in players:
            if st.checkbox('Limit market value', key='rm10_value_on'):
                max_value = st.number_input('Maximum market value (€m)', min_value=0.0, value=2.0, step=0.1, key='rm10_value')
                unknown_value = st.checkbox('Include unknown market values', value=True, key='rm10_unknown_value')
        expiry = None
        if 'Contract expires' in players and st.checkbox('Filter contract expiry', key='rm10_expiry_on'):
            expiry = st.date_input('Contract expires on or before', key='rm10_expiry')
            unknown_expiry = st.checkbox('Include unknown expiry dates', value=True, key='rm10_unknown_expiry')
        squad_minutes = st.number_input('Minimum minutes for squad benchmark', min_value=0, value=450, step=50, key='rm10_squad_minutes')
    with left:
        run = st.button('Run Recruitment Search', type='primary', key='rm101_run')
    with right:
        eligible = eligibility.get(role, [])
        if not eligible:
            st.warning('No position eligibility defined for this profile.'); return
        if dataset_key is None:
            dataset_key = hashlib.sha256(pd.util.hash_pandas_object(players, index=True).values.tobytes()).hexdigest()
        def stamp(path):
            try:
                return Path(path).stat().st_mtime_ns
            except OSError:
                return None
        signature = (dataset_key, target_league, target_club, role, depth, tuple(tiers), league_on, club_on, age, minutes, coverage, exclude, max_value, unknown_value if max_value is not None else None, expiry, unknown_expiry if expiry is not None else None, squad_minutes, league_beta, club_beta, stamp(strength_path), stamp(audit_path), stamp(snapshot_path))
        saved = st.session_state.get('rm101_results')
        if run:
            st.session_state.pop('rm101_results', None)
            try:
                mins = pd.to_numeric(players['Minutes played'],errors='coerce')
                ages = pd.to_numeric(players['Age'],errors='coerce')
                # Stable role benchmark: all selected seasons, independent of source tiers and age.
                bench = benchmark_fn(players.loc[mins.ge(minutes)],role,'Role position',[])
                mask = tier.isin(tiers) & players['Main Position'].isin(eligible) & mins.ge(minutes) & ages.between(*age)
                own = players['League'].eq(target_league) & teams.eq(target_club)
                if exclude: mask &= ~own
                if max_value is not None:
                    values = pd.to_numeric(players['Market value (M€)'], errors='coerce')
                    mask &= values.le(max_value) | (values.isna() & unknown_value)
                if expiry is not None:
                    dates = pd.to_datetime(players['Contract expires'], errors='coerce', format='mixed')
                    mask &= dates.le(pd.Timestamp(expiry)) | (dates.isna() & unknown_expiry)
                squad_mask = own & players['Main Position'].isin(eligible) & mins.ge(squad_minutes)
                # Score the full benchmark and squad together, before candidate-only filters.
                population = players.loc[players.index.isin(bench.index) | squad_mask | mask]
                benchmark_caption = f'Eligible positions: {", ".join(eligible)} | Role benchmark: {len(bench):,} rows across selected seasons'
                if population.empty or bench.empty:
                    st.info('No eligible benchmark rows. Broaden the filters.'); return
                if len(bench)<20: st.warning('Small role benchmark; interpret scores cautiously.')
                with st.spinner('Calculating existing profile scores…'):
                    score_key = (dataset_key, role, depth, coverage, minutes, tuple(population.index), tuple(bench.index))
                    cached = st.session_state.get('rm101_score_cache')
                    if cached is not None and cached[0] == score_key:
                        scores = cached[1]
                    else:
                        scored, _ = score_fn(population,bench,role,depth,coverage)
                        scores = scored['_shadow_role_score'].copy()
                        st.session_state['rm101_score_cache'] = (score_key, scores)
                        del scored
                    result = population.assign(_shadow_role_score=scores)
                result = result.rename(columns={'_shadow_role_score':'Profile Score'})
                result['Source tier'] = result['League'].map(ref['Sporting tier'])
                result['Tier difference'] = result['Source tier']-target_tier
                labels = tuple(sorted(set(result['League'].astype(str))|{target_league}))
                strength = strength_fn(labels,strength_path)
                ratings = strength.set_index('League')['rating'] if not strength.empty else pd.Series(dtype=float)
                target_rating = ratings.get(target_league,np.nan)
                result['Source Opta rating'] = result['League'].map(ratings)
                result['Opta difference'] = result['Source Opta rating']-target_rating
                rating_caption = f'Target league Opta rating: {target_rating:.2f}' if pd.notna(target_rating) else 'Target league Opta rating unavailable.'
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
                # Empirical percentile against the same adjusted role benchmark, not the shortlist.
                distribution = np.sort(result.loc[result.index.isin(bench.index),rank].dropna().to_numpy())
                valid = result[rank].notna()
                result['Benchmark Percentile'] = np.nan
                if len(distribution):
                    v = result.loc[valid,rank].to_numpy()
                    result.loc[valid,'Benchmark Percentile'] = 100*(np.searchsorted(distribution,v,'left')+np.searchsorted(distribution,v,'right'))/(2*len(distribution))
                squad = result.loc[result.index.isin(players.index[squad_mask])].sort_values(rank,ascending=False)
                squad_scores = squad[rank].dropna()
                result = result.loc[result.index.isin(players.index[mask])].copy()
                result['Difference vs Club Average'] = result[rank] - squad_scores.mean()
                result['Difference vs Best Club Player'] = result[rank] - squad_scores.max()
                result=result.sort_values(rank,ascending=False,na_position='last',kind='stable')
                st.session_state['rm101_results'] = (signature, result, squad, rank, terms, len(distribution), benchmark_caption, rating_caption)
            except Exception as exc:
                st.error('Recruitment calculation failed. See technical details below.')
                st.exception(exc)
                return
            saved = st.session_state.get('rm101_results')
        if saved is None or saved[0] != signature:
            st.info('Choose your recruitment settings, then click Run Recruitment Search.')
            return
        _, result, squad, rank, terms, distribution_count, benchmark_caption, rating_caption = saved
        squad_scores = squad[rank].dropna()
        st.caption(benchmark_caption)
        st.caption(rating_caption)
        a,b,c=st.columns(3); a.metric('Candidates',len(result)); b.metric('Source leagues',result['League'].nunique()); c.metric('Rankable',int(result[rank].notna().sum()))
        if result[rank].isna().any(): st.warning('Blank scores indicate insufficient coverage or unavailable required ratings.')
        st.subheader('Recruiting club benchmark')
        st.caption('Selected club and league-season only. Independent of candidate age, tiers, value and contract filters. Squad average is unweighted.')
        if squad_scores.empty:
            st.info('No rankable squad players at the selected minutes threshold. Upgrade differences remain blank.')
        else:
            a,b,c=st.columns(3); a.metric('Rankable squad players',len(squad_scores)); b.metric('Club average',f'{squad_scores.mean():.3f}'); c.metric('Best club score',f'{squad_scores.max():.3f}')
        squad_cols=[c for c in ['Player','Age','Minutes played','Profile Score',*terms,rank,'Benchmark Percentile'] if c in squad]
        st.dataframe(squad[list(dict.fromkeys(squad_cols))].round(3),hide_index=True,use_container_width=True)
        st.subheader('Recruitment shortlist')
        st.caption(f'Percentiles use {distribution_count:,} rankable role-benchmark rows with the same adjustment settings. Positive differences indicate higher statistical profile scores, not guaranteed upgrades.')
        defaults=list(dict.fromkeys(['Player',team_col,'League','Main Position','Age','Minutes played',rank,'Benchmark Percentile','Difference vs Club Average','Difference vs Best Club Player','Source tier','Tier difference','Source Opta rating','Opta difference','Profile Score',*terms,'Market value (M€)','Contract expires']))
        defaults=[c for c in defaults if c in result]
        options=defaults+[c for c in result.columns if c not in defaults and not c.startswith('_')]
        columns=st.multiselect('Table columns (in display order)',options,default=defaults,key='rm101_columns')
        columns=[c for c in columns if c in result]
        limit=st.slider('Rows to display',10,500,100,10,key='rm9_limit')
        if columns:
            st.dataframe(result[columns].head(limit).round(3),use_container_width=True,hide_index=True)
            st.download_button('Export recruitment shortlist (CSV)',result[columns].to_csv(index=False).encode('utf-8-sig'),'recruitment_shortlist.csv','text/csv',key='rm9_export')
        if st.checkbox('Show recruitment market overview', key='rm101_market'):
            market=result.groupby(['Source tier','League'],dropna=False)[rank].agg(Candidates='size',Rankable='count',Average='mean',Best='max').reset_index().sort_values('Best',ascending=False)
            st.dataframe(market.round(3),hide_index=True,use_container_width=True)
        if st.checkbox('Show recruitment scatter plot', key='rm101_scatter'):
            import plotly.express as px
            plot=result.dropna(subset=[rank,'Age']).head(2000).copy()
            st.caption('Chart shows up to 2,000 highest-ranked candidates.')
            if not plot.empty:
                plot['Tier']=plot['Source tier'].astype(str)
                fig=px.scatter(plot,x='Age',y=rank,color='Tier',hover_name='Player',hover_data=[team_col,'League','Minutes played'])
                if not squad_scores.empty:
                    fig.add_hline(y=squad_scores.mean(),line_dash='dash',annotation_text='Club average')
                    fig.add_hline(y=squad_scores.max(),line_dash='dot',annotation_text='Best club player')
                st.plotly_chart(fig,use_container_width=True)
        if st.checkbox('Compare shortlisted players', key='rm101_compare_on'):
            choices=result.head(limit).index.tolist()
            selected=st.multiselect('Select up to five player records',choices,format_func=lambda i:f"{result.loc[i,'Player']} — {result.loc[i,team_col]} — {result.loc[i,'League']}",max_selections=5,key='rm101_compare')
            if selected:
                st.dataframe(result.loc[selected,defaults].round(3),hide_index=True,use_container_width=True)
        st.caption('Existing league and club adjustment formulas are unchanged. Market values and contract dates are dataset observations, not verified current availability.')
