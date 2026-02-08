from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.services.api_client import get_eda, run_model, upload

st.set_page_config(page_title="Lift Impact Platform", layout="wide")
st.title("Lift Impact Platform")

if "file_id" not in st.session_state:
    st.session_state.file_id = None
    st.session_state.schema = None
    st.session_state.summary = None

with st.sidebar:
    st.header("Upload")
    file = st.file_uploader("Excel file", type=["xlsx", "xls"])
    if file and st.button("Upload & Detect Schema"):
        payload = upload(file.read(), file.name)
        st.session_state.file_id = payload["file_id"]
        st.session_state.schema = payload["schema"]

if st.session_state.file_id:
    st.success(f"Loaded file_id: {st.session_state.file_id}")
    st.json(st.session_state.schema)

    eda = get_eda(st.session_state.file_id)
    st.subheader("EDA Dashboard")
    st.metric("Action uptake rate", f"{eda['action_uptake_rate']:.2%}")
    st.metric("Suggestion→Action conversion", f"{eda['suggestion_to_action_conversion']:.2%}")

    trends = pd.DataFrame(eda["volume_trends"])
    if not trends.empty:
        x_col = trends.columns[0]
        fig = px.line(trends, x=x_col, y=trends.columns[1:])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Execution")
    multiplier = st.slider("Scenario multiplier", 0.0, 3.0, 1.0, 0.1)
    channel = st.selectbox("Isolate channel (optional)", [None] + st.session_state.schema["action_cols"])
    if st.button("Run Path A + Path B"):
        st.session_state.summary = run_model(st.session_state.file_id, multiplier, channel)["summary"]

if st.session_state.summary:
    summary = st.session_state.summary
    st.subheader("Path A Dashboard")
    st.json(summary["path_a"])
    st.subheader("Path B Dashboard")
    st.json(summary["path_b"])
    st.subheader("Final Lift Summary")
    st.write(
        {
            "total_incremental_trx": summary["path_b"]["aggregated_incremental_trx"],
            "total_incremental_nbrx": summary["path_b"]["aggregated_incremental_nbrx"],
        }
    )
