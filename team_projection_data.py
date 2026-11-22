"""Verified Wyscout club identities and dated Opta strength context."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


SEASON_TEAM = "Team within selected timeframe"
AUDIT_PATH = Path(__file__).with_name("wyscout_opta_club_audit.csv")
SNAPSHOT_PATH = Path(__file__).with_name("opta_mens_club_rankings_2026-11-20.csv")
MIN_CLUB_MEDIAN_PLAYERS = 3

AUDIT_COLUMNS = [
    "wyscout_league", "wyscout_team", "match_status", "opta_team_id",
    "opta_competition_at_snapshot", "opta_competition_country",
    "opta_snapshot_date", "opta_rating_at_snapshot", "competition_relationship",
]
SNAPSHOT_COLUMNS = [
    "snapshot_date", "opta_team_id", "competition_country", "competition", "opta_rating",
]


@st.cache_data(show_spinner=False, max_entries=2)
def read_club_audit(path: Path, modified_ns: int) -> pd.DataFrame:
    audit = pd.read_csv(path, usecols=AUDIT_COLUMNS, dtype={"opta_team_id": "string"})
    return audit.set_index(["wyscout_league", "wyscout_team"], verify_integrity=True)


@st.cache_data(show_spinner=False, max_entries=2)
def read_competition_means(path: Path, modified_ns: int) -> pd.Series:
    snapshot = pd.read_csv(path, usecols=SNAPSHOT_COLUMNS)
    snapshot = snapshot.drop_duplicates(["snapshot_date", "opta_team_id"])
    snapshot["opta_rating"] = pd.to_numeric(snapshot["opta_rating"], errors="coerce")
    return snapshot.groupby(["snapshot_date", "competition_country", "competition"])["opta_rating"].mean()


def club_audit_record(audit: pd.DataFrame, league: str, team: str) -> pd.Series | None:
    key = (league, team)
    if key in audit.index:
        return audit.loc[key]
    return None


def club_options(target_pool: pd.DataFrame, league: str, audit: pd.DataFrame) -> tuple[str, dict]:
    """Group audited aliases by ID within a league-season, keeping unrated names selectable."""
    team_column = SEASON_TEAM
    if team_column not in target_pool or target_pool[team_column].dropna().empty:
        team_column = "Team"
    options = {}
    for name in sorted(target_pool[team_column].dropna().unique(), key=str.casefold):
        if not name.strip():
            continue
        row = club_audit_record(audit, league, name) if team_column == SEASON_TEAM else None
        if row is not None and row["match_status"] == "matched" and pd.notna(row["opta_team_id"]):
            key = ("id", row["opta_team_id"])
        else:
            key = ("name", name)
        options.setdefault(key, []).append(name)
    return team_column, options


def club_strength_row(
    side: str,
    club_name: str,
    audit_row: pd.Series | None,
    averages: pd.Series,
    has_season_team: bool,
) -> dict:
    row = {
        "Side": side, "Club": club_name, "Opta club rating": np.nan,
        "Snapshot competition": "", "Competition mean": np.nan,
        "vs. competition mean": np.nan, "Snapshot date": "", "Rating status": "",
    }
    if not has_season_team:
        row["Rating status"] = "Season team unavailable"
    elif audit_row is None:
        row["Rating status"] = "Outside verified audit"
    else:
        status = audit_row["match_status"]
        row["Snapshot date"] = audit_row["opta_snapshot_date"]
        row["Rating status"] = {
            "matched": "Verified",
            "youth_team_unrated_in_snapshot": "Youth team unrated in snapshot",
            "club_unavailable_in_snapshot": "Club absent from snapshot",
            "ambiguous_opta_id": "Opta ID unresolved",
        }.get(status, "Rating unavailable")
        if status == "matched":
            rating = pd.to_numeric(audit_row["opta_rating_at_snapshot"], errors="coerce")
            competition = audit_row["opta_competition_at_snapshot"]
            snapshot_key = (row["Snapshot date"], audit_row["opta_competition_country"], competition)
            mean = averages.get(snapshot_key, np.nan)
            row.update({
                "Opta club rating": rating,
                "Snapshot competition": competition,
                "Competition mean": mean,
                "vs. competition mean": rating - mean,
            })
            relation = audit_row["competition_relationship"]
            if relation == "different_competition_at_snapshot":
                row["Rating status"] = "Verified in a different competition at snapshot date"
            elif relation == "japan_transitional_competition":
                row["Rating status"] = "Verified in regular J.League"
    return row
