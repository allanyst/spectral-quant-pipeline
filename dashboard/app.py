from __future__ import annotations
import streamlit as st
import psycopg2, psycopg2.extras
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import os

DB_URL  = os.environ.get("XRF_DB_URL", "postgresql://xrf:xrf_pass@localhost:5434/spectra_db")
API_URL = os.environ.get("API_URL", "http://api:8000")

st.set_page_config(page_title="XRF Spectral Quantification", page_icon="⚗️", layout="wide")


def db_query(sql: str, params: list | None = None) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or [])
            return pd.DataFrame(cur.fetchall())
    finally:
        conn.close()


@st.cache_data(ttl=120)
def load_summary() -> pd.DataFrame:
    return db_query("""
        SELECT sample_id, mix_type,
               conc_na, conc_mg, conc_al, conc_fe, conc_si, conc_p,
               net_na, net_mg, net_al, net_fe, net_si, net_p,
               total_counts
        FROM gold.fact_sample_summary
        ORDER BY sample_id
    """)


@st.cache_data(ttl=120)
def load_spectrum(sample_id: str) -> dict:
    conn = psycopg2.connect(DB_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT two_theta, intensities FROM bronze.xrf_spectra WHERE sample_id = %s",
                [sample_id]
            )
            row = cur.fetchone()
    finally:
        conn.close()
    return {"two_theta": row[0], "intensities": row[1]} if row else {}


# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚗️ Spectral Quantification Pipeline")
st.caption("WDXRF multi-element calibration | PLS2 | Bronze → Silver (UNNEST) → Gold")

# ── Model metrics card ────────────────────────────────────────────────────────
try:
    resp = requests.get(f"{API_URL}/model/metrics", timeout=3)
    if resp.ok:
        info = resp.json()
        m = info["metrics"]
        st.subheader("PLS2 Model Performance")
        cols = st.columns(6)
        for i, el in enumerate(["na", "mg", "al", "fe", "si", "p"]):
            with cols[i]:
                st.metric(
                    label=el.upper(),
                    value=f"R²={m[el]['r2']:.4f}",
                    delta=f"RMSE {m[el]['rmse']:.4f}",
                )
        st.caption(f"Components: {info['n_components']} | Samples: {info['n_samples']} | Channels: {info['n_channels']}")
except Exception:
    st.warning("API not reachable — model metrics unavailable")

st.divider()

# ── Spectrum viewer ───────────────────────────────────────────────────────────
summary_df = load_summary()

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Spectrum Viewer")
    selected = st.selectbox("Select sample", summary_df["sample_id"].tolist())
with col2:
    st.subheader("Mix type filter")
    mix_filter = st.multiselect("Mix types", ["unary", "binary", "ternary"],
                                default=["unary", "binary", "ternary"])

spec = load_spectrum(selected)
if spec:
    two_theta   = np.array(spec["two_theta"])
    intensities = np.array(spec["intensities"])

    row = summary_df[summary_df["sample_id"] == selected].iloc[0]

    # Peak annotation positions
    PEAKS = {"Na": 55.1, "Mg": 45.2, "Al": 57.5, "Fe": 17.5, "Si": 75.0, "P": 89.3}
    CONC_COLS = {"Na": "conc_na", "Mg": "conc_mg", "Al": "conc_al",
                 "Fe": "conc_fe", "Si": "conc_si", "P":  "conc_p"}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=two_theta, y=intensities,
        mode="lines", name="Intensity",
        line=dict(color="steelblue", width=1),
    ))
    for el, pos in PEAKS.items():
        conc = row.get(CONC_COLS[el], 0) or 0
        if conc > 0.01:
            fig.add_vline(x=pos, line=dict(color="red", dash="dash", width=1))
            fig.add_annotation(x=pos, y=max(intensities) * 0.95,
                               text=f"{el}<br>{conc:.2f}", showarrow=False,
                               font=dict(size=10, color="red"))
    fig.update_layout(
        title=f"{selected} — {row['mix_type']}",
        xaxis_title="2θ (degrees)", yaxis_title="Intensity (counts)",
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Calibration scatter: predicted vs true ────────────────────────────────────
st.subheader("Calibration — Net Peak Intensity vs True Concentration")

filtered = summary_df[summary_df["mix_type"].isin(mix_filter)]
element = st.selectbox("Element", ["Na", "Mg", "Al", "Fe", "Si", "P"])
el_l = element.lower()

fig2 = px.scatter(
    filtered,
    x=f"conc_{el_l}", y=f"net_{el_l}",
    color="mix_type",
    trendline="ols",
    labels={f"conc_{el_l}": f"True [%] {element}", f"net_{el_l}": "Net peak intensity"},
    title=f"{element} — concentration vs. net peak intensity",
    height=380,
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Dataset overview ──────────────────────────────────────────────────────────
st.subheader("Dataset Overview")
col1, col2 = st.columns(2)

with col1:
    counts = summary_df["mix_type"].value_counts().reset_index()
    counts.columns = ["mix_type", "count"]
    fig3 = px.pie(counts, names="mix_type", values="count",
                  title="Samples by mix type", height=320)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    melt = summary_df[["conc_na","conc_mg","conc_al","conc_fe","conc_si","conc_p"]].copy()
    melt.columns = ["Na","Mg","Al","Fe","Si","P"]
    melt = melt.melt(var_name="Element", value_name="Concentration")
    fig4 = px.box(melt[melt["Concentration"] > 0],
                  x="Element", y="Concentration",
                  title="Concentration distribution (non-zero only)",
                  height=320)
    st.plotly_chart(fig4, use_container_width=True)
