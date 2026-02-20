from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.services.api_client import get_checklist, get_eda, run_model, upload

st.set_page_config(page_title="Pharma Causality Lab", layout="wide")
st.title("Pharma Causality Lab")
st.caption("Causal analytics workspace for suggestions, actions, and prescription outcomes.")

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
        st.session_state.summary = None

if not st.session_state.file_id:
    st.info("Upload a source dataset to begin.")
    st.stop()

schema = st.session_state.schema
st.success(f"Active file ID: {st.session_state.file_id}")

tab_upload, tab_schema, tab_eda, tab_model = st.tabs(["Upload", "Schema & Variables", "EDA", "Modeling"])

with tab_upload:
    st.subheader("Upload Summary")
    st.json(schema)

with tab_schema:
    st.subheader("Variable Checklist")
    checklist = get_checklist(st.session_state.file_id)
    for item in checklist["items"]:
        status_icon = "✅" if item["status"] == "pass" else "⚠️"
        st.write(f"{status_icon} **{item['label']}** — {item['detail']}")

    st.divider()
    st.subheader("Detected Variable Classes")
    cols = st.columns(3)
    cols[0].write({"Suggestions": schema["suggestion_cols"]})
    cols[1].write({"Actions": schema["action_cols"]})
    cols[2].write({"Outcomes": schema["outcome_cols"]})

with tab_eda:
    st.subheader("Monthly Exploratory Analysis")
    metric_group = st.selectbox("Metric Group", ["Suggestions", "Actions", "Outcomes", "Other Variables"])
    seed_eda = get_eda(st.session_state.file_id, metric_group=metric_group)
    group_vars = seed_eda["metric_groups"].get(metric_group, [])
    variable = st.selectbox("Variable", options=group_vars if group_vars else ["N/A"])
    include_zscore = st.checkbox("Include z-score outlier count", value=False)

    eda = get_eda(
        st.session_state.file_id,
        metric_group=metric_group,
        variable=None if variable == "N/A" else variable,
        include_zscore=include_zscore,
    )

    c1, c2 = st.columns(2)
    c1.metric("Action uptake rate", f"{eda['action_uptake_rate']:.2%}")
    c2.metric("Suggestion→Action conversion", f"{eda['suggestion_to_action_conversion']:.2%}")

    trend_df = pd.DataFrame(eda["monthly_variable_trend"])
    if not trend_df.empty and variable != "N/A":
        fig = px.line(trend_df, x="month", y=variable, title=f"Monthly trend: {variable}")
        outliers = trend_df[trend_df["is_outlier"]]
        if not outliers.empty:
            fig.add_scatter(x=outliers["month"], y=outliers[variable], mode="markers", marker=dict(color="red", size=10), name="Outlier")
        st.plotly_chart(fig, use_container_width=True)

    st.write("Variable Profile", eda["variable_profile"])
    st.write("Outlier Summary", eda["outliers"])

with tab_model:
    st.subheader("Path A Setup & Execution")
    st.info("Path A first estimates action propensity, then treatment effect for incremental actions.")
    multiplier = st.slider("Scenario multiplier", 0.0, 3.0, 1.0, 0.1)
    channel = st.selectbox("Isolate channel (optional)", [None] + schema["action_cols"])

    if st.button("Run Path A + Path B"):
        st.session_state.summary = run_model(st.session_state.file_id, multiplier, channel)["summary"]

    if st.session_state.summary:
        summary = st.session_state.summary
        st.write("Path A Guidance", summary["path_a"]["setup"])
        st.write("Path A Diagnostics", summary["path_a"]["diagnostics"])
        st.dataframe(pd.DataFrame(summary["path_a"]["monthly_rollup"]))
        st.dataframe(pd.DataFrame(summary["path_a"]["channel_monthly_rollup"]))

        st.subheader("Path B Classification & Results")
        st.write(summary["path_b"]["classification"])
        st.write("Model Diagnostics", summary["path_b"]["diagnostics"])
        st.dataframe(pd.DataFrame(summary["path_b"]["monthly_rollup"]))

        st.subheader("Final Monthly Rollups & Totals")
        st.write(
            {
                "total_incremental_actions": summary["path_a"]["aggregated_incremental_actions"],
                "total_incremental_trx": summary["path_b"]["aggregated_incremental_trx"],
                "total_incremental_nbrx": summary["path_b"]["aggregated_incremental_nbrx"],
            }
        )
