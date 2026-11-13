
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.feature_engineering import FEATURE_COLUMNS
from src.risk_interpreter import (
    format_contributors,
    generate_insights,
    top_contributors,
)
from src.shap_explainer import (
    compute_shap_values,
    load_model,
    prepare_shap_inputs,
)


# --------------------------------------------------
# RISK CLASSIFICATION
# --------------------------------------------------

def classify_risk(score: float) -> str:
    if score < 35:
        return "Low Risk"
    elif score < 70:
        return "Medium Risk"
    elif score <= 100:
        return "High Risk"
    return "Unknown"


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Health Risk Scoring Dashboard",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #070b18, #101b35);
    color: #e7f1ff;
}
h1, h2, h3 {
    color: #e7f1ff;
}
[data-testid="stSidebar"] {
    background-color: #080e1e;
}
.metric-card {
    background: #172344;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #3df0ff;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_pipeline(model_path):
    return load_model(model_path)


@st.cache_data
def load_feature_data(dataset_path):
    df = pd.read_csv(dataset_path)
    return df[FEATURE_COLUMNS].copy()


# --------------------------------------------------
# STATIC PATIENT INPUTS
# --------------------------------------------------

def build_static_form():

    with st.sidebar.form("patient_form"):

        st.markdown("## Patient Information")

        st.markdown("### Demographics")

        age = st.slider("Age", 18, 80, 40)

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        st.markdown("### Lifestyle")

        smoking = st.radio(
            "Smoking",
            ["Yes", "No"]
        )

        alcohol_consumption = st.radio(
            "Alcohol Consumption",
            ["Yes", "No"]
        )

        physical_activity = st.selectbox(
            "Physical Activity",
            ["Low", "Moderate", "High"],
            index=1
        )

        family_history = st.radio(
            "Family History",
            ["Yes", "No"]
        )

        st.markdown("### Clinical Indicators")

        bmi = st.number_input(
            "BMI",
            15.0, 45.0, 25.0, 0.1
        )

        systolic_bp = st.slider(
            "Systolic BP",
            90, 200, 125
        )

        diastolic_bp = st.slider(
            "Diastolic BP",
            60, 130, 80
        )

        cholesterol_mg_dl = st.slider(
            "Cholesterol (mg/dL)",
            120, 320, 190
        )

        submitted = st.form_submit_button(
            "Predict Health Risk",
            use_container_width=True
        )

    patient_values = {
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol_mg_dl": cholesterol_mg_dl,
        "smoking": smoking,
        "alcohol_consumption": alcohol_consumption,
        "physical_activity": physical_activity,
        "family_history": family_history,
    }

    return patient_values, submitted


# --------------------------------------------------
# SMARTWATCH CSV PROCESSING
# --------------------------------------------------

def parse_smartwatch_export(uploaded_file):

    try:
        raw = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"Unable to read CSV: {exc}")

    if raw.empty:
        raise ValueError("The uploaded file is empty.")

    # Normalize column names
    normalized = {
        str(col).strip().lower().replace(" ", "_").replace("-", "_"): col
        for col in raw.columns
    }

    aliases = {
        "heart_rate_bpm": [
            "heart_rate_bpm", "heart_rate", "heartrate", "hr", "bpm"
        ],
        "sdnn_hrv": [
            "sdnn_hrv", "sdnn", "hrv_sdnn"
        ],
        "rmssd_hrv": [
            "rmssd_hrv", "rmssd", "hrv_rmssd"
        ],
        "spo2": [
            "spo2", "spo2_percent",
            "oxygen_saturation", "blood_oxygen"
        ],
    }

    rename = {}
    missing = []

    for target, candidates in aliases.items():

        source = next(
            (normalized[name] for name in candidates
             if name in normalized),
            None
        )

        if source is None:
            missing.append(target)
        else:
            rename[source] = target

    if missing:
        raise ValueError(
            "Missing smartwatch columns: " + ", ".join(missing)
        )

    data = raw.rename(columns=rename).copy()

    required = list(aliases.keys())

    for col in required:
        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        )

    data = data.dropna(subset=required)

    if data.empty:
        raise ValueError("No valid wearable readings found.")

    # Aggregate readings across the uploaded time window
    summary = data[required].mean()

    # Validate against the model's supported input ranges
    bounds = {
        "heart_rate_bpm": (50, 130),
        "sdnn_hrv": (10, 140),
        "rmssd_hrv": (10, 140),
        "spo2": (90, 100),
    }

    for col, (low, high) in bounds.items():

        if not low <= float(summary[col]) <= high:
            raise ValueError(
                f"Average {col} is outside the supported range "
                f"({low}-{high})."
            )

    wearable_values = {
        "heart_rate_bpm": int(round(summary["heart_rate_bpm"])),
        "sdnn_hrv": round(float(summary["sdnn_hrv"]), 2),
        "rmssd_hrv": round(float(summary["rmssd_hrv"]), 2),
        "spo2": round(float(summary["spo2"]), 2),
    }

    return wearable_values, data


# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

def generate_recommendations(values, risk_level):

    recommendations = []

    if values["smoking"] == "Yes":
        recommendations.append(
            "Consider reducing or stopping smoking."
        )

    if values["bmi"] >= 27:
        recommendations.append(
            "Aim for gradual weight management."
        )

    if values["cholesterol_mg_dl"] >= 200:
        recommendations.append(
            "Monitor cholesterol and maintain a balanced diet."
        )

    if values["systolic_bp"] >= 130 or values["diastolic_bp"] >= 85:
        recommendations.append(
            "Monitor blood pressure regularly."
        )

    if values["physical_activity"] == "Low":
        recommendations.append(
            "Consider increasing regular physical activity."
        )

    if values["spo2"] < 95:
        recommendations.append(
            "Monitor oxygen saturation and seek medical advice if persistently low."
        )

    if not recommendations:
        recommendations.append(
            "Maintain healthy lifestyle habits."
        )

    if risk_level == "High Risk":
        recommendations.append(
            "Consult a healthcare professional for further evaluation."
        )

    return recommendations


# --------------------------------------------------
# SHAP VISUALIZATIONS
# --------------------------------------------------

def render_summary_plot(shap_values, X_processed, feature_names):

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        show=False
    )

    st.pyplot(plt.gcf(), clear_figure=True)


def render_bar_plot(shap_values, X_processed, feature_names):

    shap.summary_plot(
        shap_values,
        X_processed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )

    st.pyplot(plt.gcf(), clear_figure=True)


def render_waterfall_plot(
    shap_row,
    base_value,
    data_row,
    feature_names
):

    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=data_row,
        feature_names=feature_names
    )

    shap.plots.waterfall(
        explanation,
        show=False
    )

    st.pyplot(plt.gcf(), clear_figure=True)


# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def run_streamlit():

    project_root = Path(__file__).resolve().parent

    dataset_path = (
        project_root / "dataset" /
        "indian_health_risk_dataset.csv"
    )

    model_path = (
        project_root / "models" /
        "random_forest_model.pkl"
    )

    # HEADER

    st.title("🩺 AI Health Risk Scoring System")

    st.markdown(
        "### Sensor-Augmented Personalized Health Risk Assessment"
    )

    st.write(
        "Combining static clinical information and wearable "
        "sensor measurements for personalized health risk prediction."
    )

    st.divider()

    # --------------------------------------------------
    # SMARTWATCH UPLOAD
    # --------------------------------------------------

    st.header("⌚ Smartwatch Data Import")

    st.write(
        "Upload the exported smartwatch CSV file. "
        "The application processes the readings and extracts "
        "physiological features for prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload Smartwatch Export",
        type=["csv"],
        key="smartwatch_upload"
    )

    wearable_values = None
    wearable_data = None

    if uploaded_file is not None:

        try:

            wearable_values, wearable_data = (
                parse_smartwatch_export(uploaded_file)
            )

            st.success(
                f"{len(wearable_data)} valid smartwatch readings processed."
            )

            st.subheader("Extracted Wearable Measurements")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Heart Rate",
                f"{wearable_values['heart_rate_bpm']} BPM"
            )

            col2.metric(
                "SDNN",
                f"{wearable_values['sdnn_hrv']} ms"
            )

            col3.metric(
                "RMSSD",
                f"{wearable_values['rmssd_hrv']} ms"
            )

            col4.metric(
                "SpO2",
                f"{wearable_values['spo2']}%"
            )

            with st.expander("View Uploaded Smartwatch Readings"):
                st.dataframe(
                    wearable_data,
                    use_container_width=True
                )

            st.info(
                "Both data sources will be used: static clinical "
                "and lifestyle information from the patient form, "
                "combined with the aggregated smartwatch measurements."
            )

        except ValueError as exc:

            st.error(str(exc))

    else:

        st.warning(
            "Upload smartwatch data before generating a combined prediction."
        )

    st.divider()

    # --------------------------------------------------
    # STATIC PATIENT INFORMATION
    # --------------------------------------------------

    st.header("Patient Clinical Information")

    patient_values, submitted = build_static_form()

    # --------------------------------------------------
    # COMBINE BOTH DATA SOURCES
    # --------------------------------------------------

    if wearable_values is not None:

        combined_values = {
            **patient_values,
            **wearable_values
        }

        st.success("Static and wearable data successfully combined.")

        with st.expander("View Combined Prediction Input"):

            combined_display = pd.DataFrame(
                [{
                    "Age": combined_values["age"],
                    "BMI": combined_values["bmi"],
                    "Systolic BP": combined_values["systolic_bp"],
                    "Diastolic BP": combined_values["diastolic_bp"],
                    "Cholesterol": combined_values["cholesterol_mg_dl"],
                    "Smoking": combined_values["smoking"],
                    "Alcohol": combined_values["alcohol_consumption"],
                    "Activity": combined_values["physical_activity"],
                    "Family History": combined_values["family_history"],
                    "Heart Rate": combined_values["heart_rate_bpm"],
                    "SDNN": combined_values["sdnn_hrv"],
                    "RMSSD": combined_values["rmssd_hrv"],
                    "SpO2": combined_values["spo2"],
                }]
            )

            st.dataframe(
                combined_display,
                use_container_width=True
            )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if submitted:

        if wearable_values is None:

            st.error(
                "Please upload a valid smartwatch CSV before predicting."
            )

        else:

            try:

                pipeline = get_pipeline(str(model_path))

                # Construct ONE combined feature row
                input_df = pd.DataFrame(
                    [combined_values],
                    columns=FEATURE_COLUMNS
                )

                st.write(
                    "Prediction is based on combined static "
                    "clinical and wearable features."
                )

                # Random Forest prediction
                prediction = float(
                    pipeline.predict(input_df)[0]
                )

                risk_level = classify_risk(prediction)

                # SHAP explanation for this patient
                X_processed, feature_names = prepare_shap_inputs(
                    pipeline,
                    input_df
                )

                shap_values, base_value = compute_shap_values(
                    pipeline,
                    X_processed
                )

                shap_row = shap_values[0]

                positive, negative = top_contributors(
                    shap_row,
                    feature_names,
                    top_n=6
                )

                insights = generate_insights(
                    positive,
                    negative,
                    combined_values
                )

                recommendations = generate_recommendations(
                    combined_values,
                    risk_level
                )

                # --------------------------------------------------
                # DISPLAY RESULTS
                # --------------------------------------------------

                st.divider()

                st.header("Personalized Health Risk Prediction")

                score_col, risk_col = st.columns(2)

                with score_col:

                    st.metric(
                        "Predicted Risk Score",
                        f"{prediction:.2f} / 100"
                    )

                with risk_col:

                    st.metric(
                        "Risk Category",
                        risk_level
                    )

                st.progress(
                    max(0.0, min(1.0, prediction / 100))
                )

                # --------------------------------------------------
                # SHAP CONTRIBUTORS
                # --------------------------------------------------

                st.header("AI Explanation")

                st.subheader("Top Contributing Features")

                contributor_lines = (
                    format_contributors(positive, "+")
                    + format_contributors(negative, "-")
                )

                for line in contributor_lines:
                    st.write("•", line)

                st.subheader("Personalized Insights")

                for insight in insights:
                    st.write("•", insight)

                # --------------------------------------------------
                # RECOMMENDATIONS
                # --------------------------------------------------

                st.header("Personalized Recommendations")

                for recommendation in recommendations:
                    st.write("•", recommendation)

                # --------------------------------------------------
                # SHAP PLOTS
                # --------------------------------------------------

                st.divider()

                st.header("Explainable AI – SHAP Visualizations")

                X_reference = load_feature_data(
                    str(dataset_path)
                )

                sample = X_reference.sample(
                    min(200, len(X_reference)),
                    random_state=42
                )

                X_ref_processed, ref_feature_names = (
                    prepare_shap_inputs(
                        pipeline,
                        sample
                    )
                )

                ref_shap_values, _ = compute_shap_values(
                    pipeline,
                    X_ref_processed
                )

                plot_col1, plot_col2 = st.columns(2)

                with plot_col1:

                    st.subheader("SHAP Summary Plot")

                    render_summary_plot(
                        ref_shap_values,
                        X_ref_processed,
                        ref_feature_names
                    )

                with plot_col2:

                    st.subheader("SHAP Feature Importance")

                    render_bar_plot(
                        ref_shap_values,
                        X_ref_processed,
                        ref_feature_names
                    )

                st.subheader("Individual Patient Explanation")

                render_waterfall_plot(
                    shap_row,
                    base_value,
                    X_processed[0],
                    feature_names
                )

                st.success(
                    "Combined static and wearable data prediction completed."
                )

            except Exception as exc:

                st.error(
                    f"Unable to generate prediction: {exc}"
                )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; padding:20px;">
            <h4>AI Healthcare System</h4>
            Sensor-Augmented Personalized Health Risk Scoring
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    run_streamlit()
