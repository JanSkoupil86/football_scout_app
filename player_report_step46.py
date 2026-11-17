"""Step 46: editable single-page landscape recruitment poster workspace.
Requires Streamlit, pandas, numpy, plotly, reportlab, requests, beautifulsoup4.
No database: report drafts are portable JSON files.
"""
from __future__ import annotations
import base64
import io
import json
import math
import re
from datetime import date
from html import unescape
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from poster_step46 import poster_pdf, poster_png

CATEGORIES = ["Physical", "Pace & Speed", "Technical", "Tactical", "Work Rate", "Attitude"]
VERDICTS = ["Sign Now", "Strong Follow-up", "Monitor", "Decline", "Not Assessed"]
DETAIL_FIELDS = {
    "name": "Player name", "club": "Current club", "season_team": "Season club", "age": "Age",
    "position": "Report position", "secondary_positions": "Secondary positions", "nationality": "Nationality / passport",
    "birth_country": "Birth country", "date_of_birth": "Date of birth", "foot": "Preferred foot",
    "height": "Height (cm)", "weight": "Weight (kg)", "contract": "Contract expiry",
    "market_value": "Market value", "agent": "Agent", "loan": "Loan status",
    "league": "League", "season": "Season", "scout": "Scout", "organization": "Organization",
    "report_date": "Report date", "transfermarkt_url": "Transfermarkt URL", "transfermarkt_id": "Transfermarkt ID",
}
TEXT_FIELDS = {
    "summary": "Overall scouting summary", "strengths": "Key strengths (one per line)",
    "weaknesses": "Points to work on (one per line)", "risks": "Recruitment risks (one per line)",
    "attacking": "Attacking assessment", "defensive": "Defensive assessment",
    "projection": "Projection / development pathway", "fit": "Fit with recruiting club",
    "adaptation": "Adaptation considerations", "next_action": "Next recruitment action",
    "team_context": "Team / league context", "data_interpretation": "Data interpretation",
    "custom_notes": "Additional notes",
}

def clean(v):
    if v is None: return ""
    try:
        if pd.isna(v): return ""
    except (TypeError, ValueError): pass
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "nat", "0.0") else s

def pick(row, *names):
    for name in names:
        if name in row and clean(row[name]): return clean(row[name])
    return ""

def defaults(row, role, depth):
    return {
        "name": pick(row,"Player"), "club": pick(row,"Team"),
        "season_team": pick(row,"Team within selected timeframe", "Team"),
        "age": pick(row,"Age"), "position": pick(row,"Main Position", "Position"),
        "secondary_positions": pick(row,"Position"), "nationality": pick(row,"Passport country"),
        "birth_country": pick(row,"Birth country"), "date_of_birth": "",
        "foot": pick(row,"Foot"), "height": pick(row,"Height"), "weight": pick(row,"Weight"),
        "contract": pick(row,"Contract expires"), "market_value": pick(row,"Market value"),
        "agent": "", "loan": pick(row,"On loan"), "league": pick(row,"League"),
        "season": pick(row,"Season label", "Season group"), "scout": "", "organization": "",
        "report_date": date.today().isoformat(), "transfermarkt_url": "", "transfermarkt_id": "",
        "role": role or "", "depth": depth or "Complete 15", "verdict": "Not Assessed",
        "current_ability": "", "potential": "", "projected_level": "", "score_override": "",
        "percentile_override": "", "show_chart": True, "show_scatters": True,
        "show_ratings": True, "show_history": True, "show_risks": True,
        "theme": "Dark", "metrics": [], "matches": [], "source_notes": "",
        "photo_b64": "", "badge_b64": "", "photo_name": "", "badge_name": "",
        **{k:"" for k in TEXT_FIELDS},
        **{f"rating_{k}": 0 for k in CATEGORIES},
        **{f"strength_{k}": "" for k in CATEGORIES},
        **{f"development_{k}": "" for k in CATEGORIES},
    }

def tm_id(url):
    m = re.search(r"/spieler/(\d+)(?:[/?#]|$)", url)
    return m.group(1) if m else ""

def tm_fetch(url):
    """Best-effort public page fetch, without authentication, proxy or anti-bot bypass."""
    p = urlparse(url)
    if p.scheme != "https" or p.hostname not in ("www.transfermarkt.com", "transfermarkt.com", "www.transfermarkt.co.uk", "transfermarkt.co.uk", "www.transfermarkt.us", "transfermarkt.us", "www.transfermarkt.de", "transfermarkt.de"):
        raise ValueError("Use a valid HTTPS Transfermarkt player-profile URL.")
    if not tm_id(url): raise ValueError("The URL must contain /spieler/<numeric ID>.")
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0 (compatible; FootballReport/1.0)"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    if "captcha" in soup.get_text(" ", strip=True).lower()[:1500]:
        raise RuntimeError("Transfermarkt requested a verification challenge.")
    fields = {"transfermarkt_id":tm_id(url)}
    title = soup.find("h1")
    if title: fields["name"] = title.get_text(" ", strip=True)
    # Only parse recognizable structured markup; never infer missing fields.
    for item in soup.select(".info-table__content--bold"):
        label_el = item.find_previous_sibling(class_="info-table__content--regular")
        if not label_el: continue
        label = label_el.get_text(" ",strip=True).lower().rstrip(":")
        value = item.get_text(" ",strip=True)
        mapping = {"height":"height", "foot":"foot", "current club":"club", "contract expires":"contract", "player agent":"agent", "citizenship":"nationality", "position":"position", "date of birth/age":"date_of_birth"}
        if label in mapping and value: fields[mapping[label]] = value
    return fields

def percentiles_for_row(row, pool, metrics, lower_is_better):
    vals=[]
    for m in metrics:
        if m not in pool or m not in row: continue
        x=pd.to_numeric(pd.Series([row[m]]),errors="coerce").iloc[0]
        s=pd.to_numeric(pool[m],errors="coerce").dropna()
        if not len(s) or not np.isfinite(x): continue
        pct=100*((s<x).sum()+0.5*(s==x).sum())/len(s)
        if m in lower_is_better: pct=100-pct
        vals.append((m,round(float(pct),1),round(float(x),2)))
    return vals

def score_rank(row, scored, score_col):
    if not score_col or score_col not in scored: return "", "", 0
    value=pd.to_numeric(pd.Series([row.get(score_col)]),errors="coerce").iloc[0]
    pool=pd.to_numeric(scored[score_col],errors="coerce").dropna()
    if not len(pool) or not np.isfinite(value): return "", "",len(pool)
    pct=100*((pool<value).sum()+0.5*(pool==value).sum())/len(pool)
    rank=1+int((pool>value).sum())
    return f"{value:+.2f}", f"{pct:.0f}", rank

def _wrap(text, font, size, width, max_lines=None):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words=str(text or "").replace("\r","").split()
    lines=[]; line=""
    for word in words:
        trial=(line+" "+word).strip()
        if stringWidth(trial,font,size)<=width: line=trial
        else:
            if line: lines.append(line)
            line=word
    if line: lines.append(line)
    if max_lines and len(lines)>max_lines:
        lines=lines[:max_lines]
        last=lines[-1]
        while last and stringWidth(last+"…",font,size)>width: last=last[:-1]
        lines[-1]=last+"…"
    return lines

def render_workspace(filtered, benchmark, active, score_col, lower_is_better, profiles, profile_definition, role_kpi_groups=None):
    st.subheader("Player Report · One-page Scouting Poster")
    st.caption("Dark 16:9 recruitment poster · editable Wyscout/Transfermarkt fields · video scouting notes · PDF and PNG export.")
    if filtered.empty or "Player" not in filtered:
        st.warning("No players in the current filters.");return
    choices=filtered.reset_index(drop=True)
    names=[f"{r.get('Player','')}  ·  {r.get('Team within selected timeframe',r.get('Team',''))}  ·  {r.get('League','')}  [{i}]" for i,r in choices.iterrows()]
    idx=st.selectbox("Select player",range(len(names)),format_func=lambda i:names[i],key="report_player_select")
    row=choices.iloc[idx]
    role=(active or {}).get("profile_name") or st.session_state.get("sidebar_selected_profile") or next(iter(profiles))
    depth=(active or {}).get("depth") or st.session_state.get("profile_depth_global","Complete 15")
    uid=f"{row.get('Player','')}::{row.get('League','')}::{row.get('Team','')}::{role}"
    state_key="player_report_data"
    if st.session_state.get("player_report_uid")!=uid:
        st.session_state[state_key]=defaults(row,role,depth)
        st.session_state["player_report_uid"]=uid
    d=st.session_state[state_key]
    d["role"]=role;d["depth"]=depth
    with st.expander("Draft management",expanded=True):
        up=st.file_uploader("Load saved report draft (JSON)",type=["json"],key="report_draft_upload")
        if up and st.button("Apply uploaded draft"):
            try:
                imported=json.load(up)
                if not isinstance(imported,dict) or imported.get("schema")!="wyscout_player_report_v1":raise ValueError("Not a Player Report draft")
                d.update({k:v for k,v in imported.get("report",{}).items() if k in d})
                st.success("Draft loaded. Review the player identity before export.");st.rerun()
            except Exception as e:st.error(f"Unable to load draft: {e}")
        if st.button("Reset report to Wyscout defaults"):
            st.session_state[state_key]=defaults(row,role,depth);st.rerun()
    with st.expander("1 · Player identity & recruitment details",expanded=True):
        cols=st.columns(3)
        for i,(key,label) in enumerate(DETAIL_FIELDS.items()):
            with cols[i%3]:
                d[key]=st.text_input(label,value=str(d.get(key,"")),key=f"report_{key}_{uid}")
        photo=st.file_uploader("Player photo (optional)",type=["png","jpg","jpeg"],key=f"photo_{uid}")
        badge=st.file_uploader("Club badge (optional)",type=["png","jpg","jpeg"],key=f"badge_{uid}")
        if photo and photo.size<3_000_000:
            d["photo_b64"]=base64.b64encode(photo.getvalue()).decode();d["photo_name"]=photo.name
        if badge and badge.size<3_000_000:
            d["badge_b64"]=base64.b64encode(badge.getvalue()).decode();d["badge_name"]=badge.name
        if photo and photo.size>=3_000_000:st.warning("Photo exceeds 3 MB; not imported.")
        if badge and badge.size>=3_000_000:st.warning("Badge exceeds 3 MB; not imported.")
        st.markdown("**Transfermarkt enrichment (optional)**")
        if d.get("transfermarkt_url"):
            d["transfermarkt_id"]=d.get("transfermarkt_id") or tm_id(d["transfermarkt_url"])
        if st.button("Fetch Transfermarkt profile",key="tm_fetch_btn"):
            try:
                fetched=tm_fetch(d.get("transfermarkt_url",""))
                st.session_state["report_tm_candidate"]=fetched
                st.success(f"Found {len(fetched)} candidate fields. Review before applying.")
            except Exception as e:st.warning(f"Automatic retrieval unavailable: {e}. You can enter all fields manually.")
        candidate=st.session_state.get("report_tm_candidate")
        if candidate:
            st.caption("Select fields to import; no automatic overwrites.")
            selections=[]
            for k,v in candidate.items():
                if k not in DETAIL_FIELDS:continue
                if st.checkbox(f"{DETAIL_FIELDS[k]}: {d.get(k,'—')}  →  {v}",key=f"tm_apply_{k}"):
                    selections.append(k)
            if st.button("Apply selected Transfermarkt fields"):
                for k in selections:d[k]=candidate[k]
                st.session_state.pop("report_tm_candidate",None)
                # Clear widget keys so new values appear in editable fields on next rerun.
                for k in selections:st.session_state.pop(f"report_{k}_{uid}",None)
                st.rerun()
    with st.expander("2 · Recruitment decision & data settings",expanded=True):
        a,b,c=st.columns(3)
        d["verdict"]=a.selectbox("Recommendation",VERDICTS,index=VERDICTS.index(d.get("verdict","Not Assessed")) if d.get("verdict") in VERDICTS else 4)
        d["current_ability"]=b.text_input("Current ability (0–20)",str(d.get("current_ability","")))
        d["potential"]=c.text_input("Potential (0–20)",str(d.get("potential","")))
        d["projected_level"]=st.text_input("Projected competition / level",str(d.get("projected_level","")))
        st.caption(f"Active role: {role} · Depth: {depth} (synchronized with sidebar)")
        d["score_override"]=st.text_input("Display Profile Score override (optional; clearly labeled)",str(d.get("score_override","")))
        d["percentile_override"]=st.text_input("Display percentile override (optional; clearly labeled)",str(d.get("percentile_override","")))
        d["theme"]="Dark"
        st.caption("Poster theme: Dark (fixed for Step 46).")
        a,b,c,dcol=st.columns(4)
        d["show_chart"]=a.checkbox("KPI wheel",value=bool(d.get("show_chart",True)))
        d["show_scatters"]=b.checkbox("Retain scatter settings in draft (not on poster)",value=bool(d.get("show_scatters",True)))
        d["show_ratings"]=c.checkbox("Video ratings",value=bool(d.get("show_ratings",True)))
        d["show_history"]=dcol.checkbox("Observation log",value=bool(d.get("show_history",True)))
        d["show_risks"]=st.checkbox("Recruitment risks",value=bool(d.get("show_risks",True)))
    try:
        requested,_=profile_definition(role,depth)
    except Exception:requested=[]
    resolved=[m for m in requested if m in benchmark.columns and m in row.index]
    # Resolve common aliases through app-level canonical alias engine when supplied externally.
    numeric_choices=[m for m in benchmark.select_dtypes(include="number").columns if not m.startswith("Score:")]
    metrics=st.multiselect("Statistical chart metrics (editable)",numeric_choices,default=[m for m in resolved if m in numeric_choices],max_selections=15,key=f"report_metrics_{uid}")
    metric_rows=percentiles_for_row(row,benchmark,metrics,lower_is_better)
    if metric_rows:
        st.caption(f"{len(metric_rows)} metrics · n={len(benchmark):,} benchmark rows · direction-aware percentiles")
        with st.expander("Metric preview"):
            st.dataframe(pd.DataFrame(metric_rows,columns=["Metric","Percentile","Raw value"]),hide_index=True)
    auto_score,auto_pct,rank=score_rank(row,filtered,score_col)
    d["display_score"]=d.get("score_override") or auto_score
    d["display_percentile"]=d.get("percentile_override") or auto_pct
    d["benchmark_n"]=len(benchmark)
    d["score_is_manual"]=bool(d.get("score_override"))
    d["percentile_is_manual"]=bool(d.get("percentile_override"))
    if d["score_is_manual"] or d["percentile_is_manual"]:st.warning("One or more displayed analytical scores are manually overridden, not calculated.")
    with st.expander("3 · Written scouting report",expanded=True):
        for key,label in TEXT_FIELDS.items():
            d[key]=st.text_area(label,value=str(d.get(key,"")),height=95 if key in ("summary","attacking","defensive") else 68,key=f"report_text_{key}_{uid}")
    with st.expander("4 · Video scouting ratings & observations"):
        for cat in CATEGORIES:
            a,b=st.columns([1,3]);v=int(d.get(f"rating_{cat}",0) or 0)
            d[f"rating_{cat}"]=a.selectbox(f"{cat} (1–7)",list(range(8)),index=v if 0<=v<=7 else 0,format_func=lambda x:"Not assessed" if x==0 else str(x),key=f"rate_{cat}_{uid}")
            with b:
                d[f"strength_{cat}"]=st.text_input(f"{cat} · Strengths",value=d.get(f"strength_{cat}",""),key=f"strength_{cat}_{uid}")
                d[f"development_{cat}"]=st.text_input(f"{cat} · Points to work on",value=d.get(f"development_{cat}",""),key=f"development_{cat}_{uid}")
        st.markdown("**Matches watched**")
        match_count=st.number_input("Number of matches",min_value=0,max_value=12,value=len(d.get("matches",[])),step=1)
        matches=[]
        for i in range(match_count):
            old=d.get("matches",[])[i] if i<len(d.get("matches",[])) else {}
            a,b,c,e=st.columns(4)
            matches.append({"date":a.text_input(f"Date #{i+1}",old.get("date",""),key=f"match_date_{uid}_{i}"),
                            "opponent":b.text_input(f"Opponent #{i+1}",old.get("opponent",""),key=f"match_opp_{uid}_{i}"),
                            "mode":c.selectbox(f"Mode #{i+1}",["Video","Live"],index=1 if old.get("mode")=="Live" else 0,key=f"match_mode_{uid}_{i}"),
                            "minutes":e.text_input(f"Minutes #{i+1}",str(old.get("minutes","")),key=f"match_min_{uid}_{i}")})
        d["matches"]=matches
        if len(matches)>=4:st.success("Observation sample: 4+ matches (verify opponent variety and match context).")
        elif len(matches)>=2:st.info("Observation sample: 2–3 matches; further viewing recommended.")
        else:st.warning("Observation sample: limited; no video conclusions should be implied from data alone.")
    scatter_points=[]
    if len(metric_rows)>=2:
        m1,m2=metric_rows[0][0],metric_rows[1][0]
        d["scatter_x"]=m1
        frame=benchmark[[m1,m2]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(frame):
            scatter_points=[(float(x),float(y),False) for x,y in frame.to_numpy()[:2500]]
            xv=pd.to_numeric(pd.Series([row[m1]]),errors="coerce").iloc[0]
            yv=pd.to_numeric(pd.Series([row[m2]]),errors="coerce").iloc[0]
            if np.isfinite(xv) and np.isfinite(yv):scatter_points.append((float(xv),float(yv),True))
    st.markdown("### Report preview & exports")
    st.caption("Single-page 16:9 dark poster. The editor retains full notes; the poster shows the concise recruitment summary.")
    a,b,c=st.columns(3)
    a.metric("Recommendation",d.get("verdict","Not Assessed"))
    b.metric("Profile percentile",d.get("display_percentile") or "N/A")
    c.metric("Video matches",len(d.get("matches",[])))
    if d.get("score_is_manual") or d.get("percentile_is_manual"):
        d["source_notes"]="MANUAL OVERRIDE: score and/or percentile displayed may not match Wyscout calculations."
    draft=json.dumps({"schema":"wyscout_player_report_v1","report":d},ensure_ascii=False,indent=2,default=str).encode("utf-8")
    st.download_button("⬇ Save editable draft (JSON)",draft,file_name="player_report_draft.json",mime="application/json")
    try:
        pdf=poster_pdf(d,metric_rows,(role_kpi_groups or {}).get(role,{}))
        try:
            png=poster_png(pdf,scale=2)
            st.image(png,caption="Live poster preview · 1920 × 1080",use_container_width=True)
            st.download_button("⬇ Download poster PNG (1920 × 1080)",png,file_name="player_scouting_poster.png",mime="image/png")
        except ImportError:
            st.info("Install pymupdf to enable the PNG preview and PNG export. PDF export is available below.")
        st.download_button("⬇ Download one-page poster PDF",pdf,file_name="player_scouting_poster.pdf",mime="application/pdf",type="primary")
        st.caption("The PDF uses editable scouting text and direction-aware Wyscout percentiles. Video conclusions are never generated from data alone.")
    except Exception as e:st.error(f"PDF generation failed: {e}")
