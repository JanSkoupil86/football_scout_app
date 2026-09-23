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
PROFILES: Dict[str, List[str]] = {'Classic Goalkeeper': ['Save rate, %',
                        'Prevented goals per 90',
                        'Conceded goals per 90',
                        'Shots against per 90',
                        'xG against per 90',
                        'Exits per 90',
                        'Aerial duels per 90.1',
                        'Aerial duels won, %',
                        'Passes per 90',
                        'Accurate passes, %',
                        'Long passes per 90',
                        'Accurate long passes, %',
                        'Back passes received as GK per 90',
                        'Forward passes per 90',
                        'Accurate forward passes, %'],
 'Sweeper Keeper': ['Exits per 90',
                    'Aerial duels per 90.1',
                    'Aerial duels won, %',
                    'Back passes received as GK per 90',
                    'Passes per 90',
                    'Accurate passes, %',
                    'Forward passes per 90',
                    'Accurate forward passes, %',
                    'Long passes per 90',
                    'Accurate long passes, %',
                    'Progressive passes per 90',
                    'Save rate, %',
                    'Prevented goals per 90',
                    'Conceded goals per 90',
                    'Shots against per 90'],
 'Build-Up Keeper': ['Back passes received as GK per 90',
                     'Passes per 90',
                     'Accurate passes, %',
                     'Forward passes per 90',
                     'Accurate forward passes, %',
                     'Progressive passes per 90',
                     'Passes to final third per 90',
                     'Accurate progressive passes, %',
                     'Long passes per 90',
                     'Accurate long passes, %',
                     'xG against per 90',
                     'Exits per 90',
                     'Save rate, %',
                     'Prevented goals per 90',
                     'Conceded goals per 90'],
 'Ball-Playing CB': ['Passes per 90',
                     'Accurate passes, %',
                     'Received passes per 90',
                     'Forward passes per 90',
                     'Progressive passes per 90',
                     'Accurate progressive passes, %',
                     'Passes to final third per 90',
                     'Progressive runs per 90',
                     'Long passes per 90',
                     'Accurate long passes, %',
                     'PAdj Interceptions',
                     'Defensive duels per 90',
                     'Defensive duels won, %',
                     'Aerial duels won, %',
                     'Successful defensive actions per 90'],
 'Combative CB / Stopper': ['Successful defensive actions per 90',
                            'Defensive duels per 90',
                            'Defensive duels won, %',
                            'PAdj Sliding tackles',
                            'Accurate forward passes, %',
                            'PAdj Interceptions',
                            'Aerial duels per 90',
                            'Aerial duels won, %',
                            'Shots blocked per 90',
                            'Sliding tackles per 90',
                            'Fouls per 90',
                            'Yellow cards per 90',
                            'Passes per 90',
                            'Accurate passes, %',
                            'Accurate long passes, %'],
 'Libero / Middle Pin CB': ['Received passes per 90',
                            'Passes per 90',
                            'Accurate passes, %',
                            'Forward passes per 90',
                            'Progressive passes per 90',
                            'Accurate progressive passes, %',
                            'Passes to final third per 90',
                            'PAdj Sliding tackles',
                            'Long passes per 90',
                            'Accurate long passes, %',
                            'PAdj Interceptions',
                            'Defensive duels won, %',
                            'Aerial duels won, %',
                            'Successful defensive actions per 90',
                            'Shots blocked per 90'],
 'Wide CB (in 3)': ['Defensive duels per 90',
                    'Defensive duels won, %',
                    'PAdj Interceptions',
                    'Successful defensive actions per 90',
                    'Progressive runs per 90',
                    'Dribbles per 90',
                    'Successful dribbles, %',
                    'Accelerations per 90',
                    'Progressive passes per 90',
                    'Accurate progressive passes, %',
                    'Passes to final third per 90',
                    'Forward passes per 90',
                    'Accurate passes, %',
                    'Long passes per 90',
                    'Accurate long passes, %'],
 'Defensive Midfielder #6': ['PAdj Interceptions',
                             'PAdj Sliding tackles',
                             'Defensive duels per 90',
                             'Defensive duels won, %',
                             'Successful defensive actions per 90',
                             'Received passes per 90',
                             'Passes per 90',
                             'Accurate passes, %',
                             'Forward passes per 90',
                             'Accurate forward passes, %',
                             'Progressive passes per 90',
                             'Accurate progressive passes, %',
                             'Passes to final third per 90',
                             'Progressive runs per 90',
                             'Aerial duels won, %'],
 'Deep-Lying Playmaker': ['Received passes per 90',
                          'Passes per 90',
                          'Accurate passes, %',
                          'Forward passes per 90',
                          'Accurate forward passes, %',
                          'Progressive passes per 90',
                          'Accurate progressive passes, %',
                          'Passes to final third per 90',
                          'Long passes per 90',
                          'Accurate long passes, %',
                          'Smart passes per 90',
                          'Key passes per 90',
                          'xA per 90',
                          'PAdj Interceptions',
                          'Defensive duels won, %'],
 'Box-to-Box Midfielder': ['Successful defensive actions per 90',
                           'Defensive duels per 90',
                           'Defensive duels won, %',
                           'PAdj Interceptions',
                           'Progressive runs per 90',
                           'Progressive passes per 90',
                           'Accelerations per 90',
                           'xA per 90',
                           'Shot assists per 90',
                           'Key passes per 90',
                           'Touches in box per 90',
                           'Shots per 90',
                           'xG per 90',
                           'Non-penalty goals per 90',
                           'Received passes per 90'],
 'Playmaker #10': ['Progressive passes per 90',
                   'Accurate progressive passes, %',
                   'Progressive runs per 90',
                   'xA per 90',
                   'Shot assists per 90',
                   'Key passes per 90',
                   'Smart passes per 90',
                   'Deep completions per 90',
                   'Passes to final third per 90',
                   'Passes to penalty area per 90',
                   'Through passes per 90',
                   'Touches in box per 90',
                   'Shots per 90',
                   'xG per 90',
                   'Dribbles per 90'],
 'Full-Back': ['Successful defensive actions per 90',
               'Defensive duels per 90',
               'Defensive duels won, %',
               'PAdj Interceptions',
               'Aerial duels won, %',
               'Progressive runs per 90',
               'Progressive passes per 90',
               'Forward passes per 90',
               'Accelerations per 90',
               'Crosses per 90',
               'Accurate crosses, %',
               'Crosses to goalie box per 90',
               'Passes to penalty area per 90',
               'Shot assists per 90',
               'xA per 90'],
 'Wing-Back': ['Progressive runs per 90',
               'Accelerations per 90',
               'Dribbles per 90',
               'Successful dribbles, %',
               'Crosses per 90',
               'Accurate crosses, %',
               'Crosses to goalie box per 90',
               'Passes to penalty area per 90',
               'Shot assists per 90',
               'xA per 90',
               'Touches in box per 90',
               'Successful attacking actions per 90',
               'Defensive duels per 90',
               'Defensive duels won, %',
               'PAdj Interceptions'],
 'Inverted Full-Back': ['Received passes per 90',
                        'Passes per 90',
                        'Accurate passes, %',
                        'Forward passes per 90',
                        'Progressive passes per 90',
                        'Accurate progressive passes, %',
                        'Passes to final third per 90',
                        'Progressive runs per 90',
                        'Smart passes per 90',
                        'Accurate forward passes, %',
                        'PAdj Interceptions',
                        'Defensive duels per 90',
                        'Defensive duels won, %',
                        'Successful defensive actions per 90',
                        'PAdj Sliding tackles'],
 'Classic Winger': ['Dribbles per 90',
                    'Successful dribbles, %',
                    'Offensive duels per 90',
                    'Offensive duels won, %',
                    'Progressive runs per 90',
                    'Accelerations per 90',
                    'Crosses per 90',
                    'Accurate crosses, %',
                    'Crosses to goalie box per 90',
                    'Passes to penalty area per 90',
                    'Shot assists per 90',
                    'xA per 90',
                    'Key passes per 90',
                    'Touches in box per 90',
                    'Successful attacking actions per 90'],
 'Inverted Winger': ['Non-penalty goals per 90',
                     'xG per 90',
                     'Shots per 90',
                     'Shots on target, %',
                     'Offensive duels won, %',
                     'Touches in box per 90',
                     'Dribbles per 90',
                     'Successful dribbles, %',
                     'Progressive runs per 90',
                     'Accelerations per 90',
                     'xA per 90',
                     'Shot assists per 90',
                     'Key passes per 90',
                     'Passes to penalty area per 90',
                     'Deep completions per 90'],
 'Target Man #9': ['Aerial duels per 90',
                   'Aerial duels won, %',
                   'Head goals per 90',
                   'Received long passes per 90',
                   'Received passes per 90',
                   'Offensive duels per 90',
                   'Accurate passes, %',
                   'Fouls suffered per 90',
                   'Touches in box per 90',
                   'xG per 90',
                   'Shots per 90',
                   'Non-penalty goals per 90',
                   'Shots on target, %',
                   'Shot assists per 90',
                   'xA per 90'],
 'Poacher': ['Non-penalty goals per 90',
             'xG per 90',
             'Shots per 90',
             'Shots on target, %',
             'Goal conversion, %',
             'Touches in box per 90',
             'Received passes per 90',
             'Head goals per 90',
             'Progressive runs per 90',
             'Accelerations per 90',
             'xA per 90',
             'Shot assists per 90',
             'Received long passes per 90',
             'Successful attacking actions per 90',
             'Offensive duels won, %'],
 'Pressing Forward': ['Successful defensive actions per 90',
                      'Defensive duels per 90',
                      'Defensive duels won, %',
                      'Fouls per 90',
                      'PAdj Interceptions',
                      'Accelerations per 90',
                      'Progressive runs per 90',
                      'Offensive duels per 90',
                      'Touches in box per 90',
                      'Shots per 90',
                      'xG per 90',
                      'Non-penalty goals per 90',
                      'Received passes per 90',
                      'Shot assists per 90',
                      'xA per 90'],
 'Creative Forward / False 9': ['Received passes per 90',
                                'Passes per 90',
                                'Accurate passes, %',
                                'Forward passes per 90',
                                'xA per 90',
                                'Shot assists per 90',
                                'Key passes per 90',
                                'Smart passes per 90',
                                'Deep completions per 90',
                                'Progressive passes per 90',
                                'Progressive runs per 90',
                                'Touches in box per 90',
                                'xG per 90',
                                'Shots per 90',
                                'Non-penalty goals per 90'],
 'Wide Forward / Inside 9': ['Non-penalty goals per 90',
                             'xG per 90',
                             'Shots per 90',
                             'Shots on target, %',
                             'Touches in box per 90',
                             'Progressive runs per 90',
                             'Accelerations per 90',
                             'Dribbles per 90',
                             'Successful dribbles, %',
                             'Goal conversion, %',
                             'xA per 90',
                             'Shot assists per 90',
                             'Key passes per 90',
                             'Passes to penalty area per 90',
                             'Deep completions per 90']}

# =========================
# Single-player percentile profiles
# =========================
# These are deliberately separate from the weighted recruitment-score profiles.
# Each tuple is: (KPI group, requested Wyscout metric).
SINGLE_PLAYER_PROFILES: Dict[str, List[Tuple[str, str]]] = {
    # 🧤 GOALKEEPERS
    "Classic Goalkeeper": [
        ("Shot Stopping", "Save rate, %"),
        ("Shot Stopping", "Prevented goals per 90"),
        ("Shot Stopping", "Conceded goals per 90"),
        ("Shot Stopping", "Shots against per 90"),
        ("Shot Stopping", "xG against per 90"),
        ("Area Control", "Exits per 90"),
        ("Area Control", "Aerial duels per 90.1"),
        ("Area Control", "Aerial duels won, %"),
        ("Distribution", "Passes per 90"),
        ("Distribution", "Accurate passes, %"),
        ("Distribution", "Long passes per 90"),
        ("Distribution", "Accurate long passes, %"),
        ("Build-Up", "Back passes received as GK per 90"),
        ("Build-Up", "Forward passes per 90"),
        ("Build-Up", "Accurate forward passes, %"),
    ],
    "Sweeper Keeper": [
        ("Sweeping", "Exits per 90"),
        ("Sweeping", "Aerial duels per 90.1"),
        ("Sweeping", "Aerial duels won, %"),
        ("Sweeping", "Back passes received as GK per 90"),
        ("Build-Up", "Passes per 90"),
        ("Build-Up", "Accurate passes, %"),
        ("Build-Up", "Forward passes per 90"),
        ("Build-Up", "Accurate forward passes, %"),
        ("Distribution", "Long passes per 90"),
        ("Distribution", "Accurate long passes, %"),
        ("Distribution", "Progressive passes per 90"),
        ("Shot Stopping", "Save rate, %"),
        ("Shot Stopping", "Prevented goals per 90"),
        ("Shot Stopping", "Conceded goals per 90"),
        ("Shot Stopping", "Shots against per 90"),
    ],
    "Build-Up Keeper": [
        ("Build-Up", "Back passes received as GK per 90"),
        ("Build-Up", "Passes per 90"),
        ("Build-Up", "Accurate passes, %"),
        ("Build-Up", "Forward passes per 90"),
        ("Build-Up", "Accurate forward passes, %"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Passes to final third per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Distribution", "Long passes per 90"),
        ("Distribution", "Accurate long passes, %"),
        ("Shot Stopping", "xG against per 90"),
        ("Sweeping", "Exits per 90"),
        ("Shot Stopping", "Save rate, %"),
        ("Shot Stopping", "Prevented goals per 90"),
        ("Shot Stopping", "Conceded goals per 90"),
    ],

    # 🛡️ CENTRE-BACKS
    "Ball-Playing CB": [
        ("Build-Up", "Passes per 90"),
        ("Build-Up", "Accurate passes, %"),
        ("Build-Up", "Received passes per 90"),
        ("Build-Up", "Forward passes per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Progression", "Progressive runs per 90"),
        ("Range", "Long passes per 90"),
        ("Range", "Accurate long passes, %"),
        ("Defending", "PAdj Interceptions"),
        ("Defending", "Defensive duels per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Defending", "Aerial duels won, %"),
        ("Defending", "Successful defensive actions per 90"),
    ],
    "Combative CB / Stopper": [
        ("Ground Defending", "Successful defensive actions per 90"),
        ("Ground Defending", "Defensive duels per 90"),
        ("Ground Defending", "Defensive duels won, %"),
        ("Ground Defending", "PAdj Sliding tackles"),
        ("Possession Security", "Accurate forward passes, %"),
        ("Ground Defending", "PAdj Interceptions"),
        ("Aerial", "Aerial duels per 90"),
        ("Aerial", "Aerial duels won, %"),
        ("Box Defence", "Shots blocked per 90"),
        ("Box Defence", "Sliding tackles per 90"),
        ("Aggression / Discipline", "Fouls per 90"),
        ("Aggression / Discipline", "Yellow cards per 90"),
        ("Possession Security", "Passes per 90"),
        ("Possession Security", "Accurate passes, %"),
        ("Possession Security", "Accurate long passes, %"),
    ],
    "Libero / Middle Pin CB": [
        ("Build-Up", "Received passes per 90"),
        ("Build-Up", "Passes per 90"),
        ("Build-Up", "Accurate passes, %"),
        ("Build-Up", "Forward passes per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Defensive Control", "PAdj Sliding tackles"),
        ("Distribution", "Long passes per 90"),
        ("Distribution", "Accurate long passes, %"),
        ("Defensive Control", "PAdj Interceptions"),
        ("Defensive Control", "Defensive duels won, %"),
        ("Defensive Control", "Aerial duels won, %"),
        ("Defensive Control", "Successful defensive actions per 90"),
        ("Defensive Control", "Shots blocked per 90"),
    ],
    "Wide CB (in 3)": [
        ("Defending", "Defensive duels per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Defending", "PAdj Interceptions"),
        ("Defending", "Successful defensive actions per 90"),
        ("Carrying", "Progressive runs per 90"),
        ("Carrying", "Dribbles per 90"),
        ("Carrying", "Successful dribbles, %"),
        ("Carrying", "Accelerations per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Progression", "Forward passes per 90"),
        ("Distribution / Security", "Accurate passes, %"),
        ("Distribution / Security", "Long passes per 90"),
        ("Distribution / Security", "Accurate long passes, %"),
    ],

    # ⚙️ MIDFIELDERS
    "Defensive Midfielder #6": [
        ("Ball Winning", "PAdj Interceptions"),
        ("Ball Winning", "PAdj Sliding tackles"),
        ("Ball Winning", "Defensive duels per 90"),
        ("Ball Winning", "Defensive duels won, %"),
        ("Ball Winning", "Successful defensive actions per 90"),
        ("Availability", "Received passes per 90"),
        ("Availability", "Passes per 90"),
        ("Availability", "Accurate passes, %"),
        ("Progression", "Forward passes per 90"),
        ("Progression", "Accurate forward passes, %"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Mobility / Security", "Progressive runs per 90"),
        ("Mobility / Security", "Aerial duels won, %"),
    ],
    "Deep-Lying Playmaker": [
        ("Involvement", "Received passes per 90"),
        ("Involvement", "Passes per 90"),
        ("Involvement", "Accurate passes, %"),
        ("Progression", "Forward passes per 90"),
        ("Progression", "Accurate forward passes, %"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Range", "Long passes per 90"),
        ("Range", "Accurate long passes, %"),
        ("Range", "Smart passes per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "xA per 90"),
        ("Defensive Contribution", "PAdj Interceptions"),
        ("Defensive Contribution", "Defensive duels won, %"),
    ],
    "Box-to-Box Midfielder": [
        ("Defending", "Successful defensive actions per 90"),
        ("Defending", "Defensive duels per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Defending", "PAdj Interceptions"),
        ("Progression", "Progressive runs per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accelerations per 90"),
        ("Creation", "xA per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Key passes per 90"),
        ("Final Third", "Touches in box per 90"),
        ("Final Third", "Shots per 90"),
        ("Final Third", "xG per 90"),
        ("Final Third", "Non-penalty goals per 90"),
        ("Involvement", "Received passes per 90"),
    ],
    "Playmaker #10": [
        ("Ball Progression", "Progressive passes per 90"),
        ("Ball Progression", "Accurate progressive passes, %"),
        ("Ball Progression", "Progressive runs per 90"),
        ("Chance Creation", "xA per 90"),
        ("Chance Creation", "Shot assists per 90"),
        ("Chance Creation", "Key passes per 90"),
        ("Chance Creation", "Smart passes per 90"),
        ("Chance Creation", "Deep completions per 90"),
        ("Final-Third Distribution", "Passes to final third per 90"),
        ("Final-Third Distribution", "Passes to penalty area per 90"),
        ("Final-Third Distribution", "Through passes per 90"),
        ("Attacking Threat", "Touches in box per 90"),
        ("Attacking Threat", "Shots per 90"),
        ("Attacking Threat", "xG per 90"),
        ("Attacking Threat", "Dribbles per 90"),
    ],

    # 🌊 FULL-BACKS / WING-BACKS
    "Full-Back": [
        ("Defending", "Successful defensive actions per 90"),
        ("Defending", "Defensive duels per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Defending", "PAdj Interceptions"),
        ("Defending", "Aerial duels won, %"),
        ("Progression", "Progressive runs per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Forward passes per 90"),
        ("Progression", "Accelerations per 90"),
        ("Delivery", "Crosses per 90"),
        ("Delivery", "Accurate crosses, %"),
        ("Delivery", "Crosses to goalie box per 90"),
        ("Creation", "Passes to penalty area per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "xA per 90"),
    ],
    "Wing-Back": [
        ("Running / Progression", "Progressive runs per 90"),
        ("Running / Progression", "Accelerations per 90"),
        ("Running / Progression", "Dribbles per 90"),
        ("Running / Progression", "Successful dribbles, %"),
        ("Delivery", "Crosses per 90"),
        ("Delivery", "Accurate crosses, %"),
        ("Delivery", "Crosses to goalie box per 90"),
        ("Delivery", "Passes to penalty area per 90"),
        ("Creation / Threat", "Shot assists per 90"),
        ("Creation / Threat", "xA per 90"),
        ("Creation / Threat", "Touches in box per 90"),
        ("Creation / Threat", "Successful attacking actions per 90"),
        ("Defending", "Defensive duels per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Defending", "PAdj Interceptions"),
    ],
    "Inverted Full-Back": [
        ("Build-Up", "Received passes per 90"),
        ("Build-Up", "Passes per 90"),
        ("Build-Up", "Accurate passes, %"),
        ("Build-Up", "Forward passes per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Passes to final third per 90"),
        ("Progression", "Progressive runs per 90"),
        ("Central Distribution", "Smart passes per 90"),
        ("Central Distribution", "Accurate forward passes, %"),
        ("Defensive Control", "PAdj Interceptions"),
        ("Defensive Control", "Defensive duels per 90"),
        ("Defensive Control", "Defensive duels won, %"),
        ("Defensive Control", "Successful defensive actions per 90"),
        ("Defensive Control", "PAdj Sliding tackles"),
    ],

    # ⚡ WINGERS
    "Classic Winger": [
        ("1v1", "Dribbles per 90"),
        ("1v1", "Successful dribbles, %"),
        ("1v1", "Offensive duels per 90"),
        ("1v1", "Offensive duels won, %"),
        ("Running", "Progressive runs per 90"),
        ("Running", "Accelerations per 90"),
        ("Delivery", "Crosses per 90"),
        ("Delivery", "Accurate crosses, %"),
        ("Delivery", "Crosses to goalie box per 90"),
        ("Delivery", "Passes to penalty area per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "xA per 90"),
        ("Creation", "Key passes per 90"),
        ("Threat", "Touches in box per 90"),
        ("Threat", "Successful attacking actions per 90"),
    ],
    "Inverted Winger": [
        ("Scoring", "Non-penalty goals per 90"),
        ("Scoring", "xG per 90"),
        ("Scoring", "Shots per 90"),
        ("Scoring", "Shots on target, %"),
        ("1v1 / Carrying", "Offensive duels won, %"),
        ("Box Threat", "Touches in box per 90"),
        ("1v1 / Carrying", "Dribbles per 90"),
        ("1v1 / Carrying", "Successful dribbles, %"),
        ("1v1 / Carrying", "Progressive runs per 90"),
        ("1v1 / Carrying", "Accelerations per 90"),
        ("Creation", "xA per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "Passes to penalty area per 90"),
        ("Creation", "Deep completions per 90"),
    ],

    # 🎯 FORWARDS
    "Target Man #9": [
        ("Aerial", "Aerial duels per 90"),
        ("Aerial", "Aerial duels won, %"),
        ("Aerial", "Head goals per 90"),
        ("Reference / Link Play", "Received long passes per 90"),
        ("Reference / Link Play", "Received passes per 90"),
        ("Reference / Link Play", "Offensive duels per 90"),
        ("Reference / Link Play", "Accurate passes, %"),
        ("Reference / Link Play", "Fouls suffered per 90"),
        ("Box Threat", "Touches in box per 90"),
        ("Box Threat", "xG per 90"),
        ("Box Threat", "Shots per 90"),
        ("Box Threat", "Non-penalty goals per 90"),
        ("Box Threat", "Shots on target, %"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "xA per 90"),
    ],
    "Poacher": [
        ("Scoring", "Non-penalty goals per 90"),
        ("Scoring", "xG per 90"),
        ("Scoring", "Shots per 90"),
        ("Scoring", "Shots on target, %"),
        ("Scoring", "Goal conversion, %"),
        ("Box Presence", "Touches in box per 90"),
        ("Box Presence", "Received passes per 90"),
        ("Box Presence", "Head goals per 90"),
        ("Movement", "Progressive runs per 90"),
        ("Movement", "Accelerations per 90"),
        ("Secondary Creation", "xA per 90"),
        ("Secondary Creation", "Shot assists per 90"),
        ("Movement", "Received long passes per 90"),
        ("Attack", "Successful attacking actions per 90"),
        ("Attack", "Offensive duels won, %"),
    ],
    "Pressing Forward": [
        ("Defensive Work", "Successful defensive actions per 90"),
        ("Defensive Work", "Defensive duels per 90"),
        ("Defensive Work", "Defensive duels won, %"),
        ("Defensive Work", "Fouls per 90"),
        ("Defensive Work", "PAdj Interceptions"),
        ("Mobility", "Accelerations per 90"),
        ("Mobility", "Progressive runs per 90"),
        ("Mobility", "Offensive duels per 90"),
        ("Box Threat", "Touches in box per 90"),
        ("Box Threat", "Shots per 90"),
        ("Box Threat", "xG per 90"),
        ("Box Threat", "Non-penalty goals per 90"),
        ("Link / Creation", "Received passes per 90"),
        ("Link / Creation", "Shot assists per 90"),
        ("Link / Creation", "xA per 90"),
    ],
    "Creative Forward / False 9": [
        ("Link Play", "Received passes per 90"),
        ("Link Play", "Passes per 90"),
        ("Link Play", "Accurate passes, %"),
        ("Link Play", "Forward passes per 90"),
        ("Creation", "xA per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "Smart passes per 90"),
        ("Creation", "Deep completions per 90"),
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Progressive runs per 90"),
        ("Threat", "Touches in box per 90"),
        ("Threat", "xG per 90"),
        ("Threat", "Shots per 90"),
        ("Threat", "Non-penalty goals per 90"),
    ],
    "Wide Forward / Inside 9": [
        ("Scoring", "Non-penalty goals per 90"),
        ("Scoring", "xG per 90"),
        ("Scoring", "Shots per 90"),
        ("Scoring", "Shots on target, %"),
        ("Box Threat", "Touches in box per 90"),
        ("Carrying", "Progressive runs per 90"),
        ("Carrying", "Accelerations per 90"),
        ("Carrying", "Dribbles per 90"),
        ("Carrying", "Successful dribbles, %"),
        ("Scoring", "Goal conversion, %"),
        ("Creation", "xA per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "Passes to penalty area per 90"),
        ("Creation", "Deep completions per 90"),
    ],
}

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


ROLE_POSITION_HINTS: Dict[str, List[str]] = {
    "Classic Goalkeeper": ["GK"],
    "Sweeper Keeper": ["GK"],
    "Build-Up Keeper": ["GK"],
    "Ball-Playing CB": ["CB"],
    "Combative CB / Stopper": ["CB"],
    "Libero / Middle Pin CB": ["CB"],
    "Wide CB (in 3)": ["CB"],
    "Defensive Midfielder #6": ["DMF", "CMF"],
    "Deep-Lying Playmaker": ["DMF", "CMF"],
    "Box-to-Box Midfielder": ["CMF", "DMF", "AMF"],
    "Playmaker #10": ["AMF", "CMF"],
    "Full-Back": ["LB", "RB", "LWB", "RWB"],
    "Wing-Back": ["LWB", "RWB", "LB", "RB"],
    "Inverted Full-Back": ["LB", "RB", "LWB", "RWB"],
    "Classic Winger": ["LW", "RW", "LWF", "RWF", "LMF", "RMF"],
    "Inverted Winger": ["LW", "RW", "LWF", "RWF", "LMF", "RMF"],
    "Target Man #9": ["CF", "ST"],
    "Poacher": ["CF", "ST"],
    "Pressing Forward": ["CF", "ST"],
    "Creative Forward / False 9": ["CF", "ST", "AMF"],
    "Wide Forward / Inside 9": ["CF", "ST", "LW", "RW", "LWF", "RWF"],
}

DEFAULT_WEIGHTS: Dict[str, Dict[str, int]] = {'Classic Goalkeeper': {'Save rate, %': 15,
                        'Prevented goals per 90': 15,
                        'Conceded goals per 90': 10,
                        'Shots against per 90': 4,
                        'xG against per 90': 6,
                        'Exits per 90': 10,
                        'Aerial duels per 90.1': 4,
                        'Aerial duels won, %': 6,
                        'Passes per 90': 3,
                        'Accurate passes, %': 5,
                        'Long passes per 90': 4,
                        'Accurate long passes, %': 8,
                        'Back passes received as GK per 90': 3,
                        'Forward passes per 90': 3,
                        'Accurate forward passes, %': 4},
 'Sweeper Keeper': {'Exits per 90': 12,
                    'Aerial duels per 90.1': 5,
                    'Aerial duels won, %': 7,
                    'Back passes received as GK per 90': 6,
                    'Passes per 90': 6,
                    'Accurate passes, %': 8,
                    'Forward passes per 90': 7,
                    'Accurate forward passes, %': 9,
                    'Long passes per 90': 5,
                    'Accurate long passes, %': 8,
                    'Progressive passes per 90': 7,
                    'Save rate, %': 7,
                    'Prevented goals per 90': 7,
                    'Conceded goals per 90': 3,
                    'Shots against per 90': 3},
 'Build-Up Keeper': {'Back passes received as GK per 90': 6,
                     'Passes per 90': 7,
                     'Accurate passes, %': 9,
                     'Forward passes per 90': 6,
                     'Accurate forward passes, %': 7,
                     'Progressive passes per 90': 9,
                     'Passes to final third per 90': 7,
                     'Accurate progressive passes, %': 9,
                     'Long passes per 90': 6,
                     'Accurate long passes, %': 9,
                     'xG against per 90': 3,
                     'Exits per 90': 5,
                     'Save rate, %': 7,
                     'Prevented goals per 90': 7,
                     'Conceded goals per 90': 3},
 'Ball-Playing CB': {'Passes per 90': 4,
                     'Accurate passes, %': 7,
                     'Received passes per 90': 3,
                     'Forward passes per 90': 6,
                     'Progressive passes per 90': 12,
                     'Accurate progressive passes, %': 10,
                     'Passes to final third per 90': 8,
                     'Progressive runs per 90': 5,
                     'Long passes per 90': 4,
                     'Accurate long passes, %': 6,
                     'PAdj Interceptions': 8,
                     'Defensive duels per 90': 5,
                     'Defensive duels won, %': 9,
                     'Aerial duels won, %': 7,
                     'Successful defensive actions per 90': 6},
 'Combative CB / Stopper': {'Successful defensive actions per 90': 10,
                            'Defensive duels per 90': 9,
                            'Defensive duels won, %': 12,
                            'PAdj Sliding tackles': 8,
                            'Accurate forward passes, %': 3,
                            'PAdj Interceptions': 8,
                            'Aerial duels per 90': 7,
                            'Aerial duels won, %': 11,
                            'Shots blocked per 90': 7,
                            'Sliding tackles per 90': 5,
                            'Fouls per 90': 5,
                            'Yellow cards per 90': 3,
                            'Passes per 90': 3,
                            'Accurate passes, %': 5,
                            'Accurate long passes, %': 4},
 'Libero / Middle Pin CB': {'Received passes per 90': 5,
                            'Passes per 90': 5,
                            'Accurate passes, %': 8,
                            'Forward passes per 90': 7,
                            'Progressive passes per 90': 10,
                            'Accurate progressive passes, %': 10,
                            'Passes to final third per 90': 7,
                            'PAdj Sliding tackles': 5,
                            'Long passes per 90': 5,
                            'Accurate long passes, %': 8,
                            'PAdj Interceptions': 7,
                            'Defensive duels won, %': 7,
                            'Aerial duels won, %': 5,
                            'Successful defensive actions per 90': 6,
                            'Shots blocked per 90': 5},
 'Wide CB (in 3)': {'Defensive duels per 90': 7,
                    'Defensive duels won, %': 9,
                    'PAdj Interceptions': 8,
                    'Successful defensive actions per 90': 6,
                    'Progressive runs per 90': 8,
                    'Dribbles per 90': 5,
                    'Successful dribbles, %': 7,
                    'Accelerations per 90': 5,
                    'Progressive passes per 90': 9,
                    'Accurate progressive passes, %': 8,
                    'Passes to final third per 90': 7,
                    'Forward passes per 90': 6,
                    'Accurate passes, %': 6,
                    'Long passes per 90': 4,
                    'Accurate long passes, %': 5},
 'Defensive Midfielder #6': {'PAdj Interceptions': 10,
                             'PAdj Sliding tackles': 7,
                             'Defensive duels per 90': 7,
                             'Defensive duels won, %': 10,
                             'Successful defensive actions per 90': 6,
                             'Received passes per 90': 7,
                             'Passes per 90': 5,
                             'Accurate passes, %': 8,
                             'Forward passes per 90': 5,
                             'Accurate forward passes, %': 5,
                             'Progressive passes per 90': 8,
                             'Accurate progressive passes, %': 7,
                             'Passes to final third per 90': 5,
                             'Progressive runs per 90': 4,
                             'Aerial duels won, %': 6},
 'Deep-Lying Playmaker': {'Received passes per 90': 7,
                          'Passes per 90': 7,
                          'Accurate passes, %': 8,
                          'Forward passes per 90': 7,
                          'Accurate forward passes, %': 6,
                          'Progressive passes per 90': 11,
                          'Accurate progressive passes, %': 10,
                          'Passes to final third per 90': 9,
                          'Long passes per 90': 6,
                          'Accurate long passes, %': 8,
                          'Smart passes per 90': 6,
                          'Key passes per 90': 4,
                          'xA per 90': 4,
                          'PAdj Interceptions': 4,
                          'Defensive duels won, %': 3},
 'Box-to-Box Midfielder': {'Successful defensive actions per 90': 7,
                           'Defensive duels per 90': 5,
                           'Defensive duels won, %': 7,
                           'PAdj Interceptions': 7,
                           'Progressive runs per 90': 10,
                           'Progressive passes per 90': 8,
                           'Accelerations per 90': 7,
                           'xA per 90': 6,
                           'Shot assists per 90': 5,
                           'Key passes per 90': 5,
                           'Touches in box per 90': 7,
                           'Shots per 90': 5,
                           'xG per 90': 7,
                           'Non-penalty goals per 90': 7,
                           'Received passes per 90': 7},
 'Playmaker #10': {'Progressive passes per 90': 9,
                   'Accurate progressive passes, %': 7,
                   'Progressive runs per 90': 6,
                   'xA per 90': 11,
                   'Shot assists per 90': 10,
                   'Key passes per 90': 10,
                   'Smart passes per 90': 8,
                   'Deep completions per 90': 6,
                   'Passes to final third per 90': 6,
                   'Passes to penalty area per 90': 7,
                   'Through passes per 90': 6,
                   'Touches in box per 90': 4,
                   'Shots per 90': 3,
                   'xG per 90': 3,
                   'Dribbles per 90': 4},
 'Full-Back': {'Successful defensive actions per 90': 7,
               'Defensive duels per 90': 6,
               'Defensive duels won, %': 9,
               'PAdj Interceptions': 8,
               'Aerial duels won, %': 5,
               'Progressive runs per 90': 9,
               'Progressive passes per 90': 8,
               'Forward passes per 90': 6,
               'Accelerations per 90': 7,
               'Crosses per 90': 6,
               'Accurate crosses, %': 8,
               'Crosses to goalie box per 90': 6,
               'Passes to penalty area per 90': 5,
               'Shot assists per 90': 5,
               'xA per 90': 5},
 'Wing-Back': {'Progressive runs per 90': 11,
               'Accelerations per 90': 8,
               'Dribbles per 90': 8,
               'Successful dribbles, %': 8,
               'Crosses per 90': 8,
               'Accurate crosses, %': 8,
               'Crosses to goalie box per 90': 6,
               'Passes to penalty area per 90': 7,
               'Shot assists per 90': 7,
               'xA per 90': 8,
               'Touches in box per 90': 6,
               'Successful attacking actions per 90': 5,
               'Defensive duels per 90': 3,
               'Defensive duels won, %': 4,
               'PAdj Interceptions': 3},
 'Inverted Full-Back': {'Received passes per 90': 8,
                        'Passes per 90': 6,
                        'Accurate passes, %': 8,
                        'Forward passes per 90': 7,
                        'Progressive passes per 90': 10,
                        'Accurate progressive passes, %': 9,
                        'Passes to final third per 90': 7,
                        'Progressive runs per 90': 6,
                        'Smart passes per 90': 5,
                        'Accurate forward passes, %': 6,
                        'PAdj Interceptions': 7,
                        'Defensive duels per 90': 4,
                        'Defensive duels won, %': 6,
                        'Successful defensive actions per 90': 6,
                        'PAdj Sliding tackles': 5},
 'Classic Winger': {'Dribbles per 90': 10,
                    'Successful dribbles, %': 10,
                    'Offensive duels per 90': 6,
                    'Offensive duels won, %': 7,
                    'Progressive runs per 90': 10,
                    'Accelerations per 90': 7,
                    'Crosses per 90': 8,
                    'Accurate crosses, %': 9,
                    'Crosses to goalie box per 90': 5,
                    'Passes to penalty area per 90': 5,
                    'Shot assists per 90': 6,
                    'xA per 90': 7,
                    'Key passes per 90': 4,
                    'Touches in box per 90': 3,
                    'Successful attacking actions per 90': 3},
 'Inverted Winger': {'Non-penalty goals per 90': 11,
                     'xG per 90': 11,
                     'Shots per 90': 8,
                     'Shots on target, %': 6,
                     'Offensive duels won, %': 5,
                     'Touches in box per 90': 8,
                     'Dribbles per 90': 9,
                     'Successful dribbles, %': 8,
                     'Progressive runs per 90': 9,
                     'Accelerations per 90': 6,
                     'xA per 90': 6,
                     'Shot assists per 90': 5,
                     'Key passes per 90': 3,
                     'Passes to penalty area per 90': 3,
                     'Deep completions per 90': 2},
 'Target Man #9': {'Aerial duels per 90': 11,
                   'Aerial duels won, %': 13,
                   'Head goals per 90': 6,
                   'Received long passes per 90': 8,
                   'Received passes per 90': 5,
                   'Offensive duels per 90': 8,
                   'Accurate passes, %': 5,
                   'Fouls suffered per 90': 8,
                   'Touches in box per 90': 8,
                   'xG per 90': 8,
                   'Shots per 90': 5,
                   'Non-penalty goals per 90': 7,
                   'Shots on target, %': 3,
                   'Shot assists per 90': 2,
                   'xA per 90': 3},
 'Poacher': {'Non-penalty goals per 90': 15,
             'xG per 90': 15,
             'Shots per 90': 9,
             'Shots on target, %': 8,
             'Goal conversion, %': 5,
             'Touches in box per 90': 12,
             'Received passes per 90': 5,
             'Head goals per 90': 4,
             'Progressive runs per 90': 6,
             'Accelerations per 90': 4,
             'xA per 90': 3,
             'Shot assists per 90': 3,
             'Received long passes per 90': 3,
             'Successful attacking actions per 90': 5,
             'Offensive duels won, %': 3},
 'Pressing Forward': {'Successful defensive actions per 90': 9,
                      'Defensive duels per 90': 7,
                      'Defensive duels won, %': 7,
                      'Fouls per 90': 3,
                      'PAdj Interceptions': 6,
                      'Accelerations per 90': 9,
                      'Progressive runs per 90': 7,
                      'Offensive duels per 90': 6,
                      'Touches in box per 90': 8,
                      'Shots per 90': 6,
                      'xG per 90': 8,
                      'Non-penalty goals per 90': 8,
                      'Received passes per 90': 5,
                      'Shot assists per 90': 5,
                      'xA per 90': 6},
 'Creative Forward / False 9': {'Received passes per 90': 8,
                                'Passes per 90': 6,
                                'Accurate passes, %': 6,
                                'Forward passes per 90': 4,
                                'xA per 90': 10,
                                'Shot assists per 90': 9,
                                'Key passes per 90': 9,
                                'Smart passes per 90': 8,
                                'Deep completions per 90': 6,
                                'Progressive passes per 90': 7,
                                'Progressive runs per 90': 5,
                                'Touches in box per 90': 6,
                                'xG per 90': 6,
                                'Shots per 90': 4,
                                'Non-penalty goals per 90': 6},
 'Wide Forward / Inside 9': {'Non-penalty goals per 90': 11,
                             'xG per 90': 11,
                             'Shots per 90': 7,
                             'Shots on target, %': 5,
                             'Touches in box per 90': 10,
                             'Progressive runs per 90': 9,
                             'Accelerations per 90': 5,
                             'Dribbles per 90': 8,
                             'Successful dribbles, %': 6,
                             'Goal conversion, %': 3,
                             'xA per 90': 6,
                             'Shot assists per 90': 5,
                             'Key passes per 90': 4,
                             'Passes to penalty area per 90': 5,
                             'Deep completions per 90': 5}}

def validate_builtin_profile_framework() -> List[str]:
    """Validate that built-in scoring and radar profiles stay perfectly aligned."""
    issues: List[str] = []
    radar_roles = set(SINGLE_PLAYER_PROFILES)
    score_roles = set(PROFILES)
    weight_roles = set(DEFAULT_WEIGHTS)

    if radar_roles != score_roles:
        issues.append(f"Role mismatch: radar={sorted(radar_roles)} score={sorted(score_roles)}")
    if radar_roles != weight_roles:
        issues.append(f"Role mismatch: radar={sorted(radar_roles)} weights={sorted(weight_roles)}")

    for role in sorted(radar_roles & score_roles & weight_roles):
        radar_metrics = [metric for _, metric in SINGLE_PLAYER_PROFILES[role]]
        score_metrics = PROFILES[role]
        weight_map = DEFAULT_WEIGHTS[role]

        if len(radar_metrics) != 15:
            issues.append(f"{role}: radar has {len(radar_metrics)} metrics, expected 15.")
        if len(score_metrics) != 15:
            issues.append(f"{role}: score has {len(score_metrics)} metrics, expected 15.")
        if radar_metrics != score_metrics:
            issues.append(f"{role}: radar and score metric lists differ.")
        if set(weight_map) != set(score_metrics):
            issues.append(f"{role}: default-weight metrics differ from score metrics.")
        if sum(int(v) for v in weight_map.values()) != 100:
            issues.append(f"{role}: default weights total {sum(weight_map.values())}%, expected 100%.")

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
    band_inner = 78.0
    band_outer = 86.0
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

    # Same z-score performance annulus as the single-player wheel.
    perf_inner = 20.0
    perf_outer = 74.0
    perf_span = perf_outer - perf_inner
    ring_theta = np.linspace(0, 360, 361)

    for tick in [-2, -1, 0, 1, 2]:
        rv = perf_inner + ((tick + 2.0) / 4.0) * perf_span
        fig.add_trace(
            go.Scatterpolar(
                r=[rv] * len(ring_theta),
                theta=ring_theta,
                mode="lines",
                line=dict(
                    color="rgba(45,55,65,0.62)" if tick == 0 else "rgba(120,130,140,0.16)",
                    width=2.8 if tick == 0 else 1,
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
        raw = pd.to_numeric(raw_table.loc[player, metrics], errors="coerce")
        clipped = z.clip(lower=-2.0, upper=2.0).to_numpy(dtype=float)
        marker_r = perf_inner + ((clipped + 2.0) / 4.0) * perf_span

        custom = np.column_stack([
            raw.to_numpy(dtype=float),
            z.to_numpy(dtype=float),
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
                    "%{customdata[3]}<br>"
                    "KPI: %{customdata[2]}<br>"
                    "Raw value: %{customdata[0]:.2f}<br>"
                    "Z-score: %{customdata[1]:+.2f}<extra></extra>"
                ),
            )
        )

    # Centre context mirrors the single-player chart.
    centre_text = f"<b>{profile_name}</b><br><span style='font-size:11px'>Multi-Player Comparison</span>"
    fig.add_annotation(
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        text=centre_text,
        showarrow=False,
        align="center",
        font=dict(size=15, color="#17202A"),
        bgcolor="rgba(255,255,255,0.94)",
        bordercolor="rgba(120,130,140,0.22)",
        borderwidth=1,
        borderpad=10,
    )

    fig.add_annotation(
        x=0.01, y=0.01,
        xref="paper", yref="paper",
        text="Z-score: −2 to +2 · 0 = benchmark mean",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=10, color="#6B7785"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{profile_name}</b><br><sup>Multi-Player Role Comparison</sup>",
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
            radialaxis=dict(range=[0, 96], visible=False),
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

    # KPI legend, matching the single-player visual language, is kept separate
    # from the player legend so player colours remain unambiguous.
    kpi_text = "  ·  ".join(
        f"<span style='color:{group_colors[g]}'><b>■ {g}</b></span>" for g in groups
    )
    fig.add_annotation(
        x=0.5, y=1.055,
        xref="paper", yref="paper",
        text=kpi_text,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=11),
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
# Profile builder (FORM)
# =========================
st.sidebar.header("Player profiles (z-score)")
if BUILTIN_PROFILE_FRAMEWORK_ISSUES:
    st.sidebar.error("Built-in profile configuration error: " + " | ".join(BUILTIN_PROFILE_FRAMEWORK_ISSUES))


# Clear active profile button (outside the form)
if st.sidebar.button("Clear active profile"):
    st.session_state["active_profile"] = None
    st.rerun()

with st.sidebar.form("profile_form"):
    st.caption("Built-in scores use the same 15 role metrics as the radar. Weights total 100% and scores are direction-aware z-score composites across the currently filtered players.")
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

        # Benchmark against the same CURRENT FILTERED population used by the
        # weighted profile score, not against only the selected comparison players.
        baseX = filtered[comp_metrics].apply(pd.to_numeric, errors="coerce")
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

        # Multi-player role wheel: same architecture as the single-player profile.
        if radar_metric_mode == "Profile metrics" and selected_radar_profile:
            fig_radar = multi_player_profile_wheel(
                raw_table=show_table,
                z_table=z_table,
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
                players=available_players,
                profile_name="Custom Metrics",
                metrics=comp_metrics,
                kpi_lookup=custom_kpi_lookup,
            )

        st.plotly_chart(fig_radar, use_container_width=True)

        st.caption(
            "Multi-player comparison now uses the same fixed-radius scouting-wheel architecture as the single-player profile: "
            "the same 15 metrics, KPI bands, metric order, KPI gaps and −2 to +2 direction-aware z-score scale. "
            "Player traces use lines and markers without fills so comparisons remain readable. Profile Score weighting remains separate from wheel geometry."
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

    # Use the current season/league/minutes universe as the benchmark source,
    # but do NOT apply the sidebar team/age/position filters to the benchmark.
    benchmark_source = df_league.copy()
    if "Minutes played" in benchmark_source.columns:
        benchmark_source = benchmark_source.loc[
            pd.to_numeric(benchmark_source["Minutes played"], errors="coerce") >= min_minutes
        ].copy()

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

        benchmark_mode = st.radio(
            "Benchmark population",
            ["Position-relevant — selected leagues", "Same Main Position — selected leagues", "All filtered players"],
            horizontal=True,
            key="single_benchmark_mode",
        )

        if benchmark_mode == "Position-relevant — selected leagues":
            if "Main Position" in benchmark_source.columns:
                pos_mask = position_family_mask(benchmark_source["Main Position"], single_role)
                benchmark_df = benchmark_source.loc[pos_mask].copy()
            else:
                benchmark_df = benchmark_source.copy()
            benchmark_desc = f"Position benchmark · selected leagues · {min_minutes}+ min"

        elif benchmark_mode == "Same Main Position — selected leagues":
            if "Main Position" in benchmark_source.columns:
                player_pos = player_row.get("Main Position")
                benchmark_df = benchmark_source.loc[benchmark_source["Main Position"] == player_pos].copy()
                benchmark_desc = f"{player_pos} benchmark · selected leagues · {min_minutes}+ min"
            else:
                benchmark_df = benchmark_source.copy()
                benchmark_desc = f"Selected leagues · {min_minutes}+ min"

        else:
            benchmark_df = filtered.copy()
            benchmark_desc = f"Current filtered population · {min_minutes}+ min"

        # If position parsing is too restrictive for a particular export, fall back safely.
        if len(benchmark_df) < 5:
            st.warning(
                "The selected position benchmark contains fewer than 5 players. "
                "Using the current filtered population instead."
            )
            benchmark_df = filtered.copy()
            benchmark_desc = f"Current filtered population · {min_minutes}+ min"

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
st.markdown("Developed with ❤️ using Streamlit & Plotly | Season-aware ✨")
