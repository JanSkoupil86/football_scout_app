import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import plotly.io as pio

# ensure kaleido is available for PNG export
pio.kaleido.scope.default_format = "png"

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    layout="wide",
    page_title="Advanced Football Scouting App",
    page_icon="⚽",
)

st.title("⚽ Advanced Football Player Scouting App — Improved")
st.markdown("Upload your football data CSV to analyze player metrics. Caching, robust parsing, ALL filters, drag-and-drop ordering, rounding to 2 decimals, downloads, and a radar chart included.")

# ... (rest of your code unchanged until the Radar chart section)

        theta = ordered_comp_metrics
        
        fig_radar = go.Figure()
        for player in compare_players:
            row = comp_df.loc[player, ordered_comp_metrics]
            r = [scale(float(row[m]), *mm[m]) if pd.notna(row[m]) else 0.0 for m in ordered_comp_metrics]
            fig_radar.add_trace(go.Scatterpolar(
                r=r + [r[0]],
                theta=theta + [theta[0]],
                fill='toself',
                name=player,
                text=[f"{player}: {val:.2f}" for val in r] + [f"{player}: {r[0]:.2f}"],
                hoverinfo="text"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template='plotly_white',
            height=640
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Download radar as PNG
        try:
            png_bytes = fig_radar.to_image(format="png", scale=2)
            st.download_button(
                "🖼️ Download radar as PNG",
                data=png_bytes,
                file_name="player_radar.png",
                mime="image/png"
            )
        except Exception:
            st.info("To enable **Download radar as PNG**, install the `kaleido` package: `pip install -U kaleido`")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit & Plotly | Enhanced edition ✨")
