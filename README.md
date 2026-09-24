# Wyscout Scouting App — improved architecture

Files:
- `app.py` — full Streamlit application.
- `profiles_config.py` — single canonical source of truth for all 21 role profiles.
- `tests_profile_framework.py` — lightweight profile/config validation.
- `requirements.txt` — deployment dependencies.

## Main methodological upgrades
- 21 profiles × 15 metrics from one canonical config.
- Exact 100% built-in/custom weighting before Apply.
- KPI-grouped weight editor.
- Search/player-pool filters separated from benchmark population.
- Role-position / selected-position / all-position benchmark controls.
- Minimum benchmark sample warning.
- Missing values no longer treated as average.
- Per-player score coverage and configurable coverage suppression.
- KPI sub-scores.
- Single-player and multi-player wheels use the same benchmark logic.
- Z-score / percentile switch retained.
- Contextual GK workload metrics can remain descriptive at 0% score weight.
- Cached benchmark summary statistics.
- Methodology panel in the app.
