import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config for better layout
st.set_page_config(layout="wide", page_title="Advanced Football Scouting App", page_icon="⚽")

st.title("⚽ Advanced Football Player Scouting App")
st.markdown("Upload your football data CSV to analyze player metrics from your extensive database.")

# List of all your provided columns for easy reference and dynamic usage
ALL_COLUMNS = [
    'Column1', 'Player', 'Team', 'Team within selected timeframe', 'Position', 'Age',
    'Market value', 'Contract expires', 'Matches played', 'Minutes played', 'Goals', 'xG',
    'Assists', 'xA', 'Duels per 90', 'Duels won, %', 'Birth country', 'Passport country',
    'Foot', 'Height', 'Weight', 'On loan', 'Successful defensive actions per 90',
    'Defensive duels per 90', 'Defensive duels won, %', 'Aerial duels per 90',
    'Aerial duels won, %', 'Sliding tackles per 90', 'PAdj Sliding tackles',
    'Shots blocked per 90', 'Interceptions per 90', 'PAdj Interceptions', 'Fouls per 90',
    'Yellow cards', 'Yellow cards per 90', 'Red cards', 'Red cards per 90',
    'Successful attacking actions per 90', 'Goals per 90', 'Non-penalty goals',
    'Non-penalty goals per 90', 'xG per 90', 'Head goals', 'Head goals per 90', 'Shots',
    'Shots per 90', 'Shots on target, %', 'Goal conversion, %', 'Assists per 90',
    'Crosses per 90', 'Accurate crosses, %', 'Crosses from left flank per 90',
    'Accurate crosses from left flank, %', 'Crosses from right flank per 90',
    'Accurate crosses from right flank, %', 'Crosses to goalie box per 90',
    'Dribbles per 90', 'Successful dribbles, %', 'Offensive duels per 90',
    'Offensive duels won, %', 'Touches in box per 90', 'Progressive runs per 90',
    'Accelerations per 90', 'Received passes per 90', 'Received long passes per 90',
    'Fouls suffered per 90', 'Passes per 90', 'Accurate passes, %', 'Forward passes per 90',
    'Accurate forward passes, %', 'Back passes per 90', 'Accurate back passes, %',
    'Lateral passes per 90', 'Accurate lateral passes, %', 'Short / medium passes per 90',
    'Accurate short / medium passes, %', 'Long passes per 90', 'Accurate long passes, %',
    'Average pass length, m', 'Average long pass length, m', 'xA per 90',
    'Shot assists per 90', 'Second assists per 90', 'Third assists per 90',
    'Smart passes per 90', 'Accurate smart passes, %', 'Key passes per 90',
    'Passes to final third per 90', 'Accurate passes to final third, %',
    'Passes to penalty area per 90', 'Accurate passes to penalty area, %',
    'Through passes per 90', 'Accurate through passes, %', 'Deep completions per 90',
    'Deep completed crosses per 90', 'Progressive passes per 90',
    'Accurate progressive passes, %', 'Conceded goals', 'Conceded goals per 90',
    'Shots against', 'Shots against per 90', 'Clean sheets', 'Save rate, %',
    'xG against', 'xG against per 90', 'Prevented goals', 'Prevented goals per 90',
    'Back passes received as GK per 90', 'Exits per 90', 'Aerial duels per 90.1',
    'Free kicks per 90', 'Direct free kicks per 90', 'Direct free kicks on target, %',
    'Corners per 90', 'Penalties taken', 'Penalty conversion, %', 'League', 'Main Position'
]

# Identify numerical columns (most of them except the identifying ones)
NUMERIC_COLUMNS = [col for col in ALL_COLUMNS if col not in [
    'Column1', 'Player', 'Team', 'Team within selected timeframe', 'Position', 'Birth country',
    'Passport country', 'Foot', 'On loan', 'Contract expires', 'League', 'Main Position'
]]

# --- File Uploader ---
uploaded_file = st.sidebar.file_uploader("Upload your Football Data CSV", type=["csv"])

df = None # Initialize df outside the if block

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV file loaded successfully!")

        # Basic cleanup: Rename 'Column1' if it's just an index
        if 'Column1' in df.columns and df['Column1'].nunique() == len(df): # Check if it looks like an index
            df = df.drop(columns=['Column1'])
            st.sidebar.info("Dropped 'Column1' as it appears to be an index.")

        # Ensure essential columns exist, or provide a warning
        required_cols = ['Player', 'Team', 'Main Position', 'Age', 'Market value', 'League']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing one or more critical columns: {', '.join(required_cols)}. Please check your CSV.")
            df = None # Invalidate df if critical columns are missing
        else:
            # --- Data Type Conversion ---
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN for numeric columns that might be used in calculations/metrics display
            for col in NUMERIC_COLUMNS:
                if col in df.columns:
                    df[col] = df[col].fillna(0) # Fill numeric NaNs with 0

            # --- Sidebar Filters ---
            st.sidebar.header("Player Filters")

            # Player search
            player_names = sorted(df['Player'].unique().tolist())
            selected_player_name = st.sidebar.selectbox("Select a specific player (optional)", ["All"] + player_names)

            # Conditional filtering based on player selection
            if selected_player_name != "All":
                # If a specific player is selected, filter the entire DataFrame for that player
                filtered_df = df[df['Player'] == selected_player_name]
                st.sidebar.info(f"Showing data for: {selected_player_name}")
            else:
                # If "All" players are selected, show general filters

                # --- NEW: League filter moved up and applied first ---
                leagues = sorted(df['League'].unique().tolist())
                selected_leagues = st.sidebar.multiselect("Select League(s)", leagues, default=leagues)

                # Apply league filter immediately
                initial_filtered_df = df[df['League'].isin(selected_leagues)]

                # Check if any players remain after league filter
                if initial_filtered_df.empty:
                    st.sidebar.warning("No players found for the selected leagues. Please adjust your league selection.")
                    filtered_df = pd.DataFrame() # Set filtered_df to empty to prevent further operations
                else:
                    teams = sorted(initial_filtered_df['Team'].unique().tolist()) # Teams are now based on selected leagues
                    main_positions = sorted(initial_filtered_df['Main Position'].unique().tolist()) # Positions also
                    
                    selected_teams = st.sidebar.multiselect("Select Team(s)", teams, default=teams)
                    selected_main_positions = st.sidebar.multiselect("Select Main Position(s)", main_positions, default=main_positions)

                    min_age, max_age = int(initial_filtered_df['Age'].min()), int(initial_filtered_df['Age'].max())
                    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

                    min_mv, max_mv = initial_filtered_df['Market value'].min(), initial_filtered_df['Market value'].max()
                    market_value_range = st.sidebar.slider("Select Market Value Range", float(min_mv), float(max_mv), (float(min_mv), float(max_mv)))


                    # Apply remaining filters on the initial_filtered_df
                    filtered_df = initial_filtered_df[
                        (initial_filtered_df['Team'].isin(selected_teams)) &
                        (initial_filtered_df['Main Position'].isin(selected_main_positions)) &
                        (initial_filtered_df['Age'] >= age_range[0]) & (initial_filtered_df['Age'] <= age_range[1]) &
                        (initial_filtered_df['Market value'] >= market_value_range[0]) & (initial_filtered_df['Market value'] <= market_value_range[1])
                    ]

                    # Filter by minutes played to exclude players with very low sample size
                    min_minutes_max = int(initial_filtered_df['Minutes played'].max()) if not initial_filtered_df.empty else 0
                    min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, min_minutes_max, 500)
                    filtered_df = filtered_df[filtered_df['Minutes played'] >= min_minutes]

            st.sidebar.markdown(f"**Players matching filters: {len(filtered_df)}**")


            # --- Main Content Area ---
            if not filtered_df.empty:
                st.subheader("Filtered Player Data")

                # Allow user to select which columns to display
                default_display_cols = ['Player', 'Team', 'League', 'Main Position', 'Age', 'Market value',
                                        'Goals', 'Assists', 'xG', 'xA', 'Minutes played']
                # Ensure defaults are actually in ALL_COLUMNS and in the df
                default_display_cols = [col for col in default_display_cols if col in df.columns]

                # Available columns for selection, excluding internal ones and potentially those with mostly 0s or NaNs
                display_options = [col for col in df.columns if col not in ['Column1', 'Team within selected timeframe', 'Position']] # Exclude Column1 and Position (using Main Position)
                selected_display_cols = st.multiselect(
                    "Select columns to display",
                    options=display_options,
                    default=default_display_cols
                )

                if selected_display_cols:
                    st.dataframe(filtered_df[selected_display_cols].sort_values(by="Player").reset_index(drop=True))
                else:
                    st.warning("Please select at least one column to display.")


                # --- Key Metrics Summary (Dynamic) ---
                st.subheader("Key Metrics Averages (Filtered Players)")
                selected_avg_metrics = st.multiselect(
                    "Select metrics for average summary",
                    options=[col for col in NUMERIC_COLUMNS if col in filtered_df.columns and col not in ['Age', 'Market value', 'Minutes played', 'Matches played']], # Exclude Age, MV, Minutes, Matches from averages as they are filtered
                    default=['Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90', 'Accurate passes, %', 'Duels won, %']
                )

                if selected_avg_metrics:
                    num_cols_for_metrics = min(4, len(selected_avg_metrics)) # Display up to 4 metrics per row
                    cols = st.columns(num_cols_for_metrics)
                    for i, metric in enumerate(selected_avg_metrics):
                        if metric in filtered_df.columns:
                            cols[i % num_cols_for_metrics].metric(f"Avg {metric}", f"{filtered_df[metric].mean():.2f}")
                else:
                    st.info("Select metrics above to see their averages.")

                # --- Visualization ---
                st.subheader("Player Performance Visualization")

                # Filter numerical columns to only those actually present in the DataFrame
                available_plot_metrics = [col for col in NUMERIC_COLUMNS if col in filtered_df.columns]

                if available_plot_metrics:
                    col_plot1, col_plot2 = st.columns(2)
                    with col_plot1:
                        x_axis = st.selectbox(
                            "Select X-axis metric",
                            available_plot_metrics,
                            index=available_plot_metrics.index('Goals per 90') if 'Goals per 90' in available_plot_metrics else 0
                        )
                    with col_plot2:
                        y_axis = st.selectbox(
                            "Select Y-axis metric",
                            available_plot_metrics,
                            index=available_plot_metrics.index('Assists per 90') if 'Assists per 90' in available_plot_metrics else min(1, len(available_plot_metrics)-1)
                        )

                    color_by_option = st.selectbox(
                        "Color points by",
                        options=['Main Position', 'Team', 'League', 'Foot', 'None'],
                        index=0
                    )

                    size_by_option = st.selectbox(
                        "Size points by",
                        options=['None', 'Minutes played', 'Market value', 'Age', 'Matches played'],
                        index=1
                    )

                    if x_axis and y_axis:
                        fig = px.scatter(
                            filtered_df,
                            x=x_axis,
                            y=y_axis,
                            hover_name="Player",
                            color=color_by_option if color_by_option != 'None' else None,
                            size=size_by_option if size_by_option != 'None' else None,
                            title=f"{y_axis} vs. {x_axis} by Player",
                            template="plotly_white", # A cleaner plot theme
                            height=600
                        )
                        fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numerical metrics available for plotting in the filtered data.")


                # --- Player Comparison (Enhanced) ---
                st.subheader("Compare Selected Players")
                # Dropdown for selecting players to compare
                compare_players_options = filtered_df['Player'].unique().tolist()
                players_to_compare = st.multiselect(
                    "Select players to compare (max 5 recommended for clarity)",
                    compare_players_options,
                    default=[]
                )

                if players_to_compare:
                    comparison_df = filtered_df[filtered_df['Player'].isin(players_to_compare)].set_index('Player')

                    # Select metrics for comparison table
                    comparison_metrics = st.multiselect(
                        "Select metrics for comparison table",
                        options=[col for col in NUMERIC_COLUMNS if col in comparison_df.columns and not col.endswith(', %')], # Exclude percentages from comparison for now, or handle them specially
                        default=['Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90',
                                 'Accurate passes, %', 'Duels won, %', 'Successful defensive actions per 90',
                                 'Market value', 'Age', 'Minutes played']
                    )

                    if comparison_metrics:
                        # Display comparison table
                        st.dataframe(comparison_df[comparison_metrics].transpose().style.highlight_max(axis=1, color='lightgreen'))

                        # Optional: Radar chart for comparison (requires more setup)
                        st.markdown("##### Radar Chart Comparison (🚧 Coming Soon!)")
                        # Placeholder for a future radar chart implementation
                        # This would involve normalizing data and using go.Figure with go.Scatterpolar
                    else:
                        st.info("Select metrics to compare players in the table.")
                else:
                    st.info("Select players from the list above to compare their stats.")

            else:
                st.warning("No players match the selected filters. Please adjust your criteria.")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
    except pd.errors.ParserError:
        st.error("Could not parse the CSV file. Please ensure it is a valid CSV.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please ensure your CSV is correctly formatted and contains expected columns.")
else:
    st.info("Please upload your football data CSV to begin scouting. It should contain columns like 'Player', 'Team', 'Main Position', 'Age', 'Goals per 90', etc.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Plotly")
