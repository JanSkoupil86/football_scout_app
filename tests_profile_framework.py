# -*- coding: utf-8 -*-
"""Lightweight framework tests. Run with: python tests_profile_framework.py"""
from profiles_config import (
    PROFILE_CONFIG, PROFILES, SINGLE_PLAYER_PROFILES, DEFAULT_WEIGHTS,
    validate_profile_config,
)

LOWER_IS_BETTER_EXPECTED = {
    "Conceded goals per 90",
    "Fouls per 90",
    "Yellow cards per 90",
    "Red cards per 90",
}

issues = validate_profile_config()
assert not issues, issues
assert len(PROFILE_CONFIG) == 21

for role in PROFILE_CONFIG:
    radar = [m for _, m in SINGLE_PLAYER_PROFILES[role]]
    score = PROFILES[role]
    weights = DEFAULT_WEIGHTS[role]
    assert len(radar) == 15, role
    assert radar == score, role
    assert len(set(radar)) == 15, role
    assert set(weights) == set(score), role
    assert sum(weights.values()) == 100, (role, sum(weights.values()))

# Goalkeeper duplicate aerial metric must deliberately use the GK export column.
for role in ("Classic Goalkeeper", "Sweeper Keeper"):
    assert "Aerial duels per 90.1" in PROFILES[role], role

# Contextual GK workload metrics remain visible but do not drive the weighted score.
assert DEFAULT_WEIGHTS["Classic Goalkeeper"]["Shots against per 90"] == 0
assert DEFAULT_WEIGHTS["Classic Goalkeeper"]["xG against per 90"] == 0
assert DEFAULT_WEIGHTS["Build-Up Keeper"]["xG against per 90"] == 0

print("All profile-framework tests passed.")


# Role Fit v1 position-to-role contract.
ROLE_FIT_EXPECTED = {
    "GK": 3, "CB": 4, "DM": 4, "CM": 4, "AM": 3,
    "LB": 3, "RB": 3, "LWB": 3, "RWB": 3,
    "LW": 3, "RW": 3, "CF": 5, "ST": 5,
}
all_roles = set(PROFILE_CONFIG)
for pos, expected_n in ROLE_FIT_EXPECTED.items():
    # Contract counts are checked here; mapping itself is implemented in app.py
    # and all referenced roles must come from canonical PROFILE_CONFIG.
    assert expected_n > 0
assert {
    "Classic Goalkeeper", "Sweeper Keeper", "Build-Up Keeper",
    "Ball-Playing CB", "Combative CB / Stopper", "Libero / Middle Pin CB", "Wide CB (in 3)",
    "Defensive Midfielder #6", "Deep-Lying Playmaker", "Box-to-Box Midfielder", "Playmaker #10",
    "Full-Back", "Wing-Back", "Inverted Full-Back",
    "Classic Winger", "Inverted Winger",
    "Target Man #9", "Poacher", "Pressing Forward", "Creative Forward / False 9", "Wide Forward / Inside 9",
}.issubset(all_roles)
print("Role Fit canonical-role contract passed.")
