Wyscout Scouting app — Team Projection

Put the files from this ZIP in the app directory together. The Streamlit
entry point is football_scout_app.py. If your existing deployment uses app.py,
copy the updated football_scout_app.py over app.py and keep that entry point.
For a local run, install requirements.txt and run:

    streamlit run football_scout_app.py

Upload your usual Wyscout CSV, choose Team Projection in the workspace menu,
and select a player, role, target league and club. The sidebar filters define
the player search. The target club list uses the selected season, and the
comparison uses the selected role's compatible Main Positions and minutes.

The projected role score uses the existing league coefficient. The radar
compares the projected player with the target club median for metrics that
have at least three scored club players with data. The squad table shows
individual options when the club sample is small.

The bundled audit and club ratings are dated 20 November 2026. The strength
table uses the club's competition at the snapshot date to calculate its
relative strength. Those figures are context for recruitment and do not enter
the projected score. Club ratings remain blank when a season-team identity
cannot be verified.
