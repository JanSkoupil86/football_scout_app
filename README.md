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

## Fixed workspace navigation
Only one main workspace renders at a time. Players no longer renders Player Comparison below the scatter.
Role Fit is directly selectable in the sidebar. The long profile-weight editor is collapsed under
Advanced profile settings.

## Radar feedback update
- Percentile is now the default presentation scale across Single Player, Multi Player and Role Fit.
- Z-score remains available as an advanced view; Profile Score methodology is unchanged.
- KPI colours use a neutral categorical palette rather than traffic-light red/green/orange semantics.
- Performance spokes/lines and endpoint markers are substantially thicker/larger.
- Metric labels are larger.
- Exact outer-tile scores are larger and easier to scan.
- Percentile reference rings emphasize 25 / 50 / 75, with 50 as the benchmark median.
- Role Fit detail wheel now follows the same visual language.

## Radar polish v3
- Percentile remains the default; Z-score remains optional.
- Thicker performance geometry and larger endpoint markers.
- Larger metric labels and outer percentile values.
- Quieter secondary reference rings.
- Explicit `Higher = better` percentile interpretation.
- Lower-is-better raw metrics use a ↓ display cue where the alias map is available.
- KPI colours remain categorical rather than performance traffic lights.

## Scouting Position overrides v1
- Original `Main Position` from Wyscout is preserved.
- New `Scouting Position` is the effective analytical position.
- Sidebar `Position overrides` editor corrects a player-season-team record.
- Overrides are session-state based and do not mutate the uploaded CSV.
- Position filters, role-position benchmarks, Single Player and Role Fit use `Scouting Position`.
- The default Players table shows both source and scouting positions.
