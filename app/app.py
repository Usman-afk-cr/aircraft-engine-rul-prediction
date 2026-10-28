# ============================================================
# CAUC-AeroRULNet - Professional Streamlit Application
# ============================================================

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    FINAL_RESULT_DIR,
    DECISION_RESULT_DIR,
    SELECTED_SENSORS,
    VALIDATION_RESULTS,
    DECISION_RESULTS,
    SEQUENCE_LENGTH,
    NUM_SENSORS,
    MODEL_PARAMETERS,
    FINAL_LEARNING_RATE,
    FINAL_DROPOUT,
    MODEL_COLOR_MAP,
    RISK_COLOR_MAP,
)
from inference import predict_fleet, get_loaded_model_paths

st.set_page_config(
    page_title="CAUC-AeroRULNet",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    :root { --accent:#F97316; --ink:#0F172A; --muted:#64748B; }
    .block-container {max-width:1500px; padding-top:1.15rem; padding-bottom:3rem;}
    h1,h2,h3 {letter-spacing:-0.025em;}
    h1 {font-weight:850!important;}
    h2 {font-weight:800!important; margin-top:1.2rem!important;}
    div[data-testid="stMetric"] {
        border:1px solid rgba(148,163,184,.26); border-radius:16px;
        padding:1rem 1.05rem; background:rgba(255,255,255,.02);
        box-shadow:0 4px 18px rgba(15,23,42,.04);
    }
    div[data-testid="stMetricValue"] {font-weight:800;}
    .hero {
        padding:1.35rem 1.5rem; border-radius:20px;
        border:1px solid rgba(249,115,22,.22);
        background:linear-gradient(135deg,rgba(249,115,22,.10),rgba(14,165,233,.06));
        margin-bottom:1rem;
    }
    .hero h1 {margin:0 0 .25rem 0; font-size:2.55rem;}
    .hero p {margin:.15rem 0; color:#64748B; font-size:1.02rem;}
    .section-note {color:#64748B; font-size:.92rem;}
    div.stButton > button {border-radius:11px; font-weight:750; min-height:46px;}
    div[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "prediction_dataset" not in st.session_state:
    st.session_state.prediction_dataset = None

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
with st.sidebar:
    st.title("✈️ CAUC-AeroRULNet")
    st.caption("Aircraft Predictive Maintenance")
    page = st.radio(
        "Navigation",
        [
            "Overview",
            "RUL Prediction",
            "Fleet Dashboard",
            "Model Performance",
            "Research Analysis",
            "Methodology",
            "System Check",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Final model configuration")
    st.write(f"**Sequence:** {SEQUENCE_LENGTH} cycles")
    st.write(f"**Sensors:** {NUM_SENSORS}")
    st.write(f"**Parameters:** {MODEL_PARAMETERS:,}")

st.markdown(
    """
    <div class="hero">
      <h1>CAUC-AeroRULNet</h1>
      <p>Aircraft Engine Remaining Useful Life Prediction & Maintenance Decision Support</p>
      <p>NASA C-MAPSS · FD002 + FD003 · Condition-aware preprocessing · Deep temporal modeling</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def validation_dataframe():
    rows = []
    for dataset, models in VALIDATION_RESULTS.items():
        for model_name, metrics in models.items():
            rows.append({
                "Dataset": dataset,
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R²": metrics["R2"],
            })
    return pd.DataFrame(rows)


def rmse_improvement(dataset):
    b = VALIDATION_RESULTS[dataset]["BiLSTM"]["RMSE"]
    c = VALIDATION_RESULTS[dataset]["CAUC-AeroRULNet"]["RMSE"]
    return (b - c) / b * 100


def risk_summary(successful):
    return {
        risk: int((successful["Risk_Level"] == risk).sum())
        for risk in ["Critical", "High Risk", "Moderate", "Healthy"]
    }

# ------------------------------------------------------------
# Overview
# ------------------------------------------------------------
if page == "Overview":
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Research Data", "NASA C-MAPSS")
    c2.metric("Datasets", "FD002 + FD003")
    c3.metric("Selected Sensors", NUM_SENSORS)
    c4.metric("Sequence Length", f"{SEQUENCE_LENGTH} cycles")
    c5.metric("Model Parameters", f"{MODEL_PARAMETERS:,}")

    st.subheader("End-to-End Predictive Maintenance Pipeline")
    pipeline = pd.DataFrame({
        "Stage": [
            "Engine Sensor Data", "Input Validation", "Operating-Condition Analysis",
            "Condition-Aware Normalization", "50×16 Sequence", "CAUC-AeroRULNet",
            "Continuous RUL", "Maintenance Decision"
        ],
        "Output": [
            "FD002 / FD003 history", "Validated numerical records", "FD002 regime assignment",
            "Training-fitted sensor scaling", "Temporal model input", "Degradation representation",
            "Remaining cycles", "Risk / Priority / Action"
        ]
    })
    st.dataframe(pipeline, hide_index=True, use_container_width=True)

    st.subheader("Maintenance Policy")
    policy = pd.DataFrame({
        "Predicted RUL": ["0–30", "31–60", "61–90", ">90"],
        "Risk": ["Critical", "High Risk", "Moderate", "Healthy"],
        "Priority": ["P1", "P2", "P3", "P4"],
        "Recommended Action": [
            "Immediate maintenance required", "Schedule maintenance soon",
            "Increase monitoring frequency", "Continue routine monitoring"
        ]
    })
    st.dataframe(policy, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
elif page == "RUL Prediction":
    st.header("RUL Prediction & Maintenance Assessment")
    left, right = st.columns([1, 2])
    with left:
        dataset_type = st.selectbox("Dataset Profile", ["FD002", "FD003"])
    with right:
        if dataset_type == "FD002":
            st.info("FD002 requires cycle, 16 selected sensors and 3 operational settings.")
        else:
            st.info("FD003 requires cycle and the final 16 selected sensor measurements.")

    with st.expander("Required CSV Columns"):
        cols = ["cycle", *SELECTED_SENSORS]
        if dataset_type == "FD002":
            cols += [
                "operational_setting_1",
                "operational_setting_2",
                "operational_setting_3",
            ]
        st.dataframe(pd.DataFrame({"Required Column": cols}), hide_index=True)
        st.caption("Recommended history: at least 50 cycles per engine. Add engine_id for fleet prediction.")

    uploaded = st.file_uploader("Upload Engine Sensor CSV", type=["csv"])
    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            if input_df.empty:
                st.error("Uploaded CSV is empty.")
            else:
                ecount = input_df["engine_id"].nunique() if "engine_id" in input_df.columns else 1
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", f"{len(input_df):,}")
                m2.metric("Columns", input_df.shape[1])
                m3.metric("Engines", ecount)
                m4.metric("Max Cycle", input_df["cycle"].max() if "cycle" in input_df.columns else "N/A")

                with st.expander("Preview Uploaded Data"):
                    st.dataframe(input_df.head(30), use_container_width=True)

                if st.button("Run CAUC-AeroRULNet Prediction", type="primary", use_container_width=True):
                    with st.spinner("Running train-fitted preprocessing and model inference..."):
                        results = predict_fleet(input_df, dataset_type)
                    st.session_state.prediction_results = results
                    st.session_state.prediction_dataset = dataset_type
                    st.success("Prediction completed successfully.")
        except Exception as exc:
            st.error(f"Unable to process CSV: {exc}")

    results = st.session_state.prediction_results
    if results is not None:
        successful = results[results["Status"] == "Success"].copy()
        failed = results[results["Status"] == "Failed"].copy()

        if not successful.empty:
            st.subheader("Prediction Results")
            show_cols = [
                "Engine_ID", "Observed_Cycles", "Predicted_RUL", "Risk_Level",
                "Maintenance_Priority", "History_Quality", "Recommendation"
            ]
            st.dataframe(successful[show_cols], hide_index=True, use_container_width=True)

            # Important sanity information for deployment debugging
            with st.expander("Inference Sanity Check"):
                sanity_cols = [
                    "Engine_ID", "Predicted_RUL_Raw", "Input_Mean", "Input_Std",
                    "Input_Min", "Input_Max", "Padding_Used"
                ]
                st.dataframe(successful[sanity_cols], hide_index=True, use_container_width=True)
                st.caption(
                    "If all engines have almost identical normalized input statistics and RUL values, "
                    "verify the uploaded data and training-time preprocessing artifacts."
                )

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Prediction Report",
                data=csv_bytes,
                file_name="CAUC_AeroRULNet_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if not failed.empty:
            st.error(f"{len(failed)} engine(s) failed during inference.")
            st.dataframe(failed, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# Fleet dashboard
# ------------------------------------------------------------
elif page == "Fleet Dashboard":
    st.header("Fleet Maintenance Dashboard")
    results = st.session_state.prediction_results
    if results is None:
        st.info("Run predictions from the RUL Prediction page first.")
    else:
        successful = results[results["Status"] == "Success"].copy()
        if successful.empty:
            st.warning("No successful predictions are available.")
        else:
            counts = risk_summary(successful)
            a,b,c,d,e,f = st.columns(6)
            a.metric("Fleet Size", len(successful))
            b.metric("Critical", counts["Critical"])
            c.metric("High Risk", counts["High Risk"])
            d.metric("Moderate", counts["Moderate"])
            e.metric("Healthy", counts["Healthy"])
            f.metric("Average RUL", f"{successful['Predicted_RUL'].mean():.1f}")

            left, right = st.columns([1.05, 1])
            with left:
                risk_counts = pd.DataFrame({
                    "Risk Level": ["Critical", "High Risk", "Moderate", "Healthy"],
                    "Engines": [counts["Critical"], counts["High Risk"], counts["Moderate"], counts["Healthy"]]
                })
                fig = px.bar(
                    risk_counts, x="Risk Level", y="Engines", color="Risk Level",
                    text="Engines", color_discrete_map=RISK_COLOR_MAP,
                    title="Fleet Risk Distribution"
                )
                fig.update_layout(showlegend=False, height=430, margin=dict(l=20,r=20,t=60,b=20))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                fig2 = px.histogram(
                    successful, x="Predicted_RUL", nbins=20,
                    title="Predicted RUL Distribution"
                )
                fig2.update_traces(marker_color="#F97316")
                fig2.update_layout(height=430, xaxis_title="Predicted RUL (cycles)", yaxis_title="Engines")
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Maintenance Priority Queue")
            queue_cols = [
                "Engine_ID", "Predicted_RUL", "Risk_Level", "Maintenance_Priority",
                "Observed_Cycles", "History_Quality", "Recommendation"
            ]
            st.dataframe(successful[queue_cols], hide_index=True, use_container_width=True)

            critical = successful[successful["Risk_Level"] == "Critical"]
            if not critical.empty:
                st.error(f"{len(critical)} engine(s) require immediate maintenance attention.")
            else:
                st.success("No engine is currently classified as Critical.")

# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------
elif page == "Model Performance":
    st.header("Model Performance & Evaluation")
    df = validation_dataframe()

    fd2_imp = rmse_improvement("FD002")
    fd3_imp = rmse_improvement("FD003")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("FD002 CAUC RMSE", f"{VALIDATION_RESULTS['FD002']['CAUC-AeroRULNet']['RMSE']:.4f}")
    m2.metric("FD002 RMSE Improvement", f"{fd2_imp:.2f}%")
    m3.metric("FD003 CAUC RMSE", f"{VALIDATION_RESULTS['FD003']['CAUC-AeroRULNet']['RMSE']:.4f}")
    m4.metric("FD003 RMSE Improvement", f"{fd3_imp:.2f}%")

    st.caption("FD002 now uses the improved final CAUC-AeroRULNet result; the old 4.99% value is no longer used.")

    st.subheader("Final Validation Results")
    st.dataframe(df.round(4), hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_rmse = px.bar(
            df, x="Dataset", y="RMSE", color="Model", barmode="group",
            text_auto=".2f", color_discrete_map=MODEL_COLOR_MAP,
            title="RMSE Comparison"
        )
        fig_rmse.update_layout(height=460, legend_title="Model")
        st.plotly_chart(fig_rmse, use_container_width=True)
    with c2:
        fig_r2 = px.bar(
            df, x="Dataset", y="R²", color="Model", barmode="group",
            text_auto=".3f", color_discrete_map=MODEL_COLOR_MAP,
            title="R² Comparison"
        )
        fig_r2.update_layout(height=460, legend_title="Model")
        st.plotly_chart(fig_r2, use_container_width=True)

    st.info(
        "CAUC-AeroRULNet achieved the lowest RMSE and highest R² on both validation subsets, "
        "while BiLSTM retained a slightly lower MAE."
    )

    st.subheader("Maintenance Decision-Support Validation")
    decision_rows = []
    for dataset, metrics in DECISION_RESULTS.items():
        decision_rows.append({
            "Dataset": dataset,
            "Accuracy": metrics["Accuracy"],
            "Macro Precision": metrics["Precision"],
            "Macro Recall": metrics["Recall"],
            "Macro F1": metrics["F1"],
            "Critical Recall": metrics["Critical_Recall"],
            "Critical F1": metrics["Critical_F1"],
        })
    st.dataframe(pd.DataFrame(decision_rows).round(4), hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# Research analysis
# ------------------------------------------------------------
elif page == "Research Analysis":
    st.header("Research Analysis & Diagnostics")
    st.subheader("Final Selected Sensors")
    st.dataframe(
        pd.DataFrame({"No.": range(1, len(SELECTED_SENSORS)+1), "Sensor": SELECTED_SENSORS}),
        hide_index=True, use_container_width=True
    )

    range_path = os.path.join(FINAL_RESULT_DIR, "fd002_rul_range_comparison.csv")
    if os.path.exists(range_path):
        range_df = pd.read_csv(range_path)
        st.subheader("FD002 Performance by RUL Range")
        st.dataframe(range_df.round(4), hide_index=True, use_container_width=True)
    else:
        st.warning("FD002 RUL-range comparison CSV was not found.")

    cm1 = os.path.join(DECISION_RESULT_DIR, "fd002_confusion_matrix.png")
    cm2 = os.path.join(DECISION_RESULT_DIR, "fd003_confusion_matrix.png")
    if os.path.exists(cm1) and os.path.exists(cm2):
        st.subheader("Maintenance Risk Confusion Matrices")
        x,y = st.columns(2)
        with x: st.image(cm1, caption="FD002 Maintenance Risk Classification", use_container_width=True)
        with y: st.image(cm2, caption="FD003 Maintenance Risk Classification", use_container_width=True)

# ------------------------------------------------------------
# Methodology
# ------------------------------------------------------------
elif page == "Methodology":
    st.header("Research Methodology & Architecture")
    st.subheader("CAUC-AeroRULNet Architecture")
    architecture = pd.DataFrame({
        "Component": [
            "Input", "Conv1D k=3", "Conv1D k=5", "Conv1D k=7", "Feature Fusion",
            "Batch Normalization + Dropout", "Bidirectional GRU", "Temporal Attention",
            "Feature Attention", "Dense 64 → Dense 32", "RUL Output"
        ],
        "Configuration / Purpose": [
            "50 cycles × 16 sensors", "32 filters", "32 filters", "32 filters",
            "96 fused features", "Dropout 0.20", "64 units per direction → 128 features",
            "Weights informative time steps", "Weights informative latent features",
            "Regression representation", "Continuous remaining cycles"
        ]
    })
    st.dataframe(architecture, hide_index=True, use_container_width=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sequence Length", SEQUENCE_LENGTH)
    c2.metric("Sensors", NUM_SENSORS)
    c3.metric("Learning Rate", FINAL_LEARNING_RATE)
    c4.metric("Dropout", FINAL_DROPOUT)

# ------------------------------------------------------------
# System check
# ------------------------------------------------------------
elif page == "System Check":
    st.header("Deployment System Check")
    st.caption("Use this page to verify that Streamlit is loading the intended trained artifacts.")

    try:
        paths = get_loaded_model_paths()
        st.success("Models loaded successfully.")
        st.code(f"FD002 model:\n{paths['FD002']}\n\nFD003 model:\n{paths['FD003']}")
        if "improved" in os.path.basename(paths["FD002"]).lower():
            st.success("FD002 improved final model is active.")
        else:
            st.warning("FD002 loaded filename does not contain 'improved'. Verify that this is the intended model file.")
    except Exception as exc:
        st.error(str(exc))

    st.subheader("Expected Final Validation Values")
    check = pd.DataFrame([
        {"Dataset":"FD002", **VALIDATION_RESULTS["FD002"]["CAUC-AeroRULNet"]},
        {"Dataset":"FD003", **VALIDATION_RESULTS["FD003"]["CAUC-AeroRULNet"]},
    ])
    st.dataframe(check.round(4), hide_index=True, use_container_width=True)
    st.metric("Expected FD002 RMSE Improvement", f"{rmse_improvement('FD002'):.2f}%")

st.divider()
st.caption("CAUC-AeroRULNet · NASA C-MAPSS · Aircraft Engine RUL Prediction · Predictive Maintenance Decision Support")
