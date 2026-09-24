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

## Role Fit v1
- Player selector using the currently filtered player pool.
- Automatic Main Position → compatible canonical roles.
- Each compatible role uses its own role-position benchmark.
- Weighted Role Score, within-role Role Percentile, Coverage %, Benchmark N.
- Horizontal Role Percentile comparison.
- Selectable role detail.
- KPI sub-score decomposition.
- Percentile / Z-score detail wheel.
- Raw role-metric table with KPI, raw value, z-score, percentile and weight.
- No qualitative fit labels or separate scoring model.

## Navigation update
The app now exposes five sidebar workspaces:
- Players
- Compare
- Single Player
- Multi Player
- Role Fit

Role Fit is a real workspace rather than a section at the bottom of the app.
The long profile-weight editor is collapsed under `Advanced profile settings`.
