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
        ("Distribution", "Average pass length, m"),
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
        ("Defending", "Interceptions per 90"),
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
        ("Ground Defending", "Interceptions per 90"),
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
        ("Progression", "Deep completions per 90"),
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
        ("Wide Contribution", "Crosses per 90"),
        ("Wide Contribution", "Passes to penalty area per 90"),
        ("Wide Contribution", "Accurate long passes, %"),
    ],

    # ⚙️ MIDFIELDERS
    "Defensive Midfielder #6": [
        ("Ball Winning", "PAdj Interceptions"),
        ("Ball Winning", "Interceptions per 90"),
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
    "Attacking Midfielder #8": [
        ("Progression", "Progressive passes per 90"),
        ("Progression", "Accurate progressive passes, %"),
        ("Progression", "Progressive runs per 90"),
        ("Progression", "Passes to final third per 90"),
        ("Creation", "xA per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Smart passes per 90"),
        ("Final Third", "Shots per 90"),
        ("Final Third", "Touches in box per 90"),
        ("Final Third", "Passes to penalty area per 90"),
        ("Final Third", "xG per 90"),
        ("Defending", "Interceptions per 90"),
        ("Defending", "Defensive duels won, %"),
        ("Attack", "Successful attacking actions per 90"),
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
        ("Final Third", "Successful attacking actions per 90"),
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
        ("Attacking Threat", "Successful attacking actions per 90"),
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
        ("Central Creation", "Smart passes per 90"),
        ("Central Creation", "xA per 90"),
        ("Defensive Control", "PAdj Interceptions"),
        ("Defensive Control", "Defensive duels per 90"),
        ("Defensive Control", "Defensive duels won, %"),
        ("Defensive Control", "Successful defensive actions per 90"),
        ("Defensive Control", "Interceptions per 90"),
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
        ("Scoring", "Goal conversion, %"),
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
        ("Reference / Link Play", "Passes per 90"),
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
        ("Secondary Creation", "Key passes per 90"),
        ("Attack", "Successful attacking actions per 90"),
        ("Attack", "Offensive duels won, %"),
    ],
    "Pressing Forward": [
        ("Defensive Work", "Successful defensive actions per 90"),
        ("Defensive Work", "Defensive duels per 90"),
        ("Defensive Work", "Defensive duels won, %"),
        ("Defensive Work", "Interceptions per 90"),
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
        ("Carrying", "Offensive duels won, %"),
        ("Creation", "xA per 90"),
        ("Creation", "Shot assists per 90"),
        ("Creation", "Key passes per 90"),
        ("Creation", "Passes to penalty area per 90"),
        ("Creation", "Deep completions per 90"),
    ],
}

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
    "Attacking Midfielder #8": ["CMF", "AMF"],
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
    Circular role profile.
    Percentile: bars grow 0 -> 100 from the centre.
    Z-score: a true zero ring sits at radius 3; positive values grow outward,
    negative values grow inward toward the centre. Display is clipped to ±3.
    """
    if profile_df.empty:
        return go.Figure()

    profile_df = profile_df.reset_index(drop=True).copy()
    groups = profile_df["KPI Group"].drop_duplicates().tolist()

    # Muted professional palette with stable KPI-group colours.
    palette = [
        "#4E79A7", "#59A14F", "#F28E2B", "#E15759",
        "#76B7B2", "#B07AA1", "#EDC948", "#9C755F",
    ]
    group_colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

    labels = profile_df["Metric"].tolist()
    group_sequence = profile_df["KPI Group"].tolist()
    n = len(labels)

    # Wider gaps between KPI families than between individual metrics.
    gap_units = 0.55
    boundaries = sum(
        1 for i in range(1, n)
        if group_sequence[i] != group_sequence[i - 1]
    )
    unit_width = 360.0 / (n + boundaries * gap_units)

    theta_list = []
    cursor = 0.0
    for i in range(n):
        if i > 0 and group_sequence[i] != group_sequence[i - 1]:
            cursor += gap_units * unit_width
        theta_list.append(cursor)
        cursor += unit_width
    theta = np.array(theta_list, dtype=float)

    # Slightly narrower bars create clean white separators.
    bar_width = unit_width * 0.88

    fig = go.Figure()

    if scale_mode == "Percentile":
        for group in groups:
            sub = profile_df[profile_df["KPI Group"] == group]
            idx = sub.index.to_list()
            positions = [float(theta[i]) for i in idx]

            custom = np.column_stack([
                sub["Raw Value"].to_numpy(dtype=float),
                sub["Percentile"].to_numpy(dtype=float),
                sub["Z-score"].to_numpy(dtype=float),
                sub["Metric"].astype(str).to_numpy(),
                sub["KPI Group"].astype(str).to_numpy(),
            ])

            fig.add_trace(
                go.Barpolar(
                    r=sub["Percentile"].to_numpy(dtype=float),
                    theta=positions,
                    width=[bar_width] * len(sub),
                    base=0,
                    name=group,
                    marker_color=group_colors[group],
                    marker_line_color="white",
                    marker_line_width=1.4,
                    opacity=0.90,
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "KPI: %{customdata[4]}<br>"
                        "Raw value: %{customdata[0]:.2f}<br>"
                        "Percentile: %{customdata[1]:.0f}<br>"
                        "Z-score: %{customdata[2]:+.2f}<extra></extra>"
                    ),
                )
            )

        value_r = np.clip(
            profile_df["Percentile"].to_numpy(dtype=float) - 5.0,
            7.0,
            96.0,
        )
        value_text = [
            f"<b>{int(round(v))}</b>"
            for v in profile_df["Percentile"].to_numpy(dtype=float)
        ]
        radial_range = [0, 100]
        radial_tickvals = [20, 40, 60, 80, 100]
        radial_ticktext = ["20", "40", "60", "80", "100"]

    else:
        # True z-score geometry:
        # radius 3 = z 0
        # z +3 ends at radius 6
        # z -3 ends at radius 0
        # Positive bars start at zero and extend outward.
        # Negative bars start at their negative endpoint and extend outward to zero,
        # so the wedge occupies the correct side of the zero baseline.
        z_clipped = np.clip(
            profile_df["Z-score"].to_numpy(dtype=float),
            -3.0,
            3.0,
        )

        for group in groups:
            sub = profile_df[profile_df["KPI Group"] == group]
            idx = sub.index.to_list()
            positions = [float(theta[i]) for i in idx]
            zvals = np.clip(sub["Z-score"].to_numpy(dtype=float), -3.0, 3.0)

            bases = np.where(zvals >= 0.0, 3.0, 3.0 + zvals)
            lengths = np.abs(zvals)

            custom = np.column_stack([
                sub["Raw Value"].to_numpy(dtype=float),
                sub["Percentile"].to_numpy(dtype=float),
                sub["Z-score"].to_numpy(dtype=float),
                sub["Metric"].astype(str).to_numpy(),
                sub["KPI Group"].astype(str).to_numpy(),
            ])

            fig.add_trace(
                go.Barpolar(
                    r=lengths,
                    base=bases,
                    theta=positions,
                    width=[bar_width] * len(sub),
                    name=group,
                    marker_color=group_colors[group],
                    marker_line_color="white",
                    marker_line_width=1.4,
                    opacity=0.90,
                    customdata=custom,
                    hovertemplate=(
                        "<b>%{customdata[3]}</b><br>"
                        "KPI: %{customdata[4]}<br>"
                        "Raw value: %{customdata[0]:.2f}<br>"
                        "Percentile: %{customdata[1]:.0f}<br>"
                        "Z-score: %{customdata[2]:+.2f}<extra></extra>"
                    ),
                )
            )

        # Put the value label close to the actual endpoint of each wedge.
        endpoints = 3.0 + z_clipped
        value_r = np.where(
            z_clipped >= 0,
            np.minimum(endpoints - 0.18, 5.82),
            np.maximum(endpoints + 0.18, 0.18),
        )
        value_text = [
            f"<b>{v:+.2f}</b>"
            for v in profile_df["Z-score"].to_numpy(dtype=float)
        ]
        radial_range = [0, 6]
        radial_tickvals = [0, 1, 2, 3, 4, 5, 6]
        radial_ticktext = ["-3", "-2", "-1", "0", "+1", "+2", "+3"]

        # Strong zero baseline.
        zero_theta = np.linspace(0, 360, 361)
        fig.add_trace(
            go.Scatterpolar(
                r=[3.0] * len(zero_theta),
                theta=zero_theta,
                mode="lines",
                line=dict(color="rgba(45,45,45,0.72)", width=2.4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Values on wedges.
    fig.add_trace(
        go.Scatterpolar(
            r=value_r,
            theta=theta,
            mode="text",
            text=value_text,
            textfont=dict(size=12, color="#17202A"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # KPI group labels positioned around the outer perimeter.
    group_label_theta = []
    group_label_text = []
    for group in groups:
        positions = [
            theta[i]
            for i, g in enumerate(group_sequence)
            if g == group
        ]
        if positions:
            group_label_theta.append(float(np.mean(positions)))
            group_label_text.append(f"<b>{group}</b>")

    group_label_radius = 109 if scale_mode == "Percentile" else 6.55

    fig.add_trace(
        go.Scatterpolar(
            r=[group_label_radius] * len(group_label_theta),
            theta=group_label_theta,
            mode="text",
            text=group_label_text,
            textfont=dict(size=11, color="#34495E"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Shorter title; benchmark context moves into subtitle.
    fig.update_layout(
        title=dict(
            text=f"<b>{player_name}</b><br><sup>{subtitle}</sup>",
            x=0.5,
            xanchor="center",
            y=0.985,
            font=dict(size=21, color="#17202A"),
        ),
        template="plotly_white",
        height=900,
        margin=dict(l=150, r=150, t=150, b=115),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.055,
            xanchor="center",
            x=0.5,
            title_text="",
            font=dict(size=11),
        ),
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                range=radial_range,
                tickvals=radial_tickvals,
                ticktext=radial_ticktext,
                tickfont=dict(size=10, color="#7B8794"),
                gridcolor="rgba(110,120,130,0.18)",
                gridwidth=1,
                showline=False,
                angle=90,
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=theta,
                ticktext=labels,
                direction="clockwise",
                rotation=90,
                gridcolor="rgba(255,255,255,0)",
                tickfont=dict(size=10, color="#5F6B7A"),
                showline=False,
            ),
        ),
        barmode="overlay",
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
    # Existing comparison logic preserved.
    player_options = sorted(filtered["Player"].dropna().unique().tolist()) if "Player" in filtered.columns else []
    compare_players = st.multiselect(
        "Players to compare (max 5 recommended)",
        options=player_options,
        default=[],
        key="multi_compare_players",
    )

    if compare_players and "Player" in filtered.columns:
        comp_rows = filtered.loc[filtered["Player"].isin(compare_players)].copy()
        # If a player has more than one row after filtering, keep the row with most minutes.
        if "Minutes played" in comp_rows.columns:
            comp_rows["_cmp_minutes"] = pd.to_numeric(comp_rows["Minutes played"], errors="coerce").fillna(0)
            comp_rows = comp_rows.sort_values("_cmp_minutes", ascending=False).drop_duplicates("Player")
            comp_rows = comp_rows.drop(columns="_cmp_minutes")
        else:
            comp_rows = comp_rows.drop_duplicates("Player")
        comp_df = comp_rows.set_index("Player")

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

        comp_metrics = st.multiselect(
            "Metrics for comparison table & radar",
            options=comp_metric_choices,
            default=default_comp,
            key="multi_compare_metrics",
        )

        if comp_metrics:
            show_table = comp_df[comp_metrics].copy()
            for c in show_table.columns:
                show_table[c] = pd.to_numeric(show_table[c], errors="coerce").round(2)

            st.dataframe(show_table.T, use_container_width=True)

            baseX = filtered[comp_metrics].apply(pd.to_numeric, errors="coerce")
            means = baseX.mean(axis=0)
            stds = baseX.std(axis=0, ddof=0).replace(0, np.nan)

            theta = comp_metrics
            fig_radar = go.Figure()
            for player in compare_players:
                if player not in show_table.index:
                    continue
                row = show_table.loc[player, comp_metrics]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                row = row.apply(pd.to_numeric, errors="coerce")
                z = ((row - means) / stds).fillna(0.0)

                for m in theta:
                    if m in LOWER_IS_BETTER:
                        z[m] = -z[m]

                r = z.to_list()
                if not r:
                    continue
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

else:
    st.markdown("#### Single-Player Role Profile")
    st.caption(
        "15 role-specific metrics grouped by KPI family. Percentiles and z-scores use the same benchmark and are direction-aware. "
        "The weighted recruitment profile score is not changed by this visualization."
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
            help="Percentile shows 0–100 rank. Z-score shows standard deviations from the benchmark mean and is direction-aware.",
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
        st.markdown(f"### {single_player}" + (f" — {' | '.join(header_bits)}" if header_bits else ""))

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
