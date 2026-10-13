
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wearable Stress Monitor",
    page_icon="💗",
    layout="wide"
)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

MODEL = ART / "best_model.joblib"
FEATURES = ART / "feature_columns.joblib"
METRICS = ART / "evaluation_metrics.csv"

# -------------------- PAGE HEADER --------------------

st.title("💗 AI-Based Wearable Stress Monitoring")
st.caption("WESAD research prototype • ML stress classification")

st.info(
    "Research prototype only—not a medical or diagnostic tool. "
    "Upload features generated using the same preprocessing pipeline as training."
)

# -------------------- LOAD MODEL --------------------

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL)
    feature_cols = list(joblib.load(FEATURES))
    return model, feature_cols


if not MODEL.exists() or not FEATURES.exists():
    st.error(
        "Model files not found. Ensure best_model.joblib and "
        "feature_columns.joblib are inside the artifacts/ folder."
    )
    st.code(
        "artifacts/best_model.joblib\n"
        "artifacts/feature_columns.joblib"
    )
    st.stop()

try:
    model, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

st.success(f"Model loaded — {len(feature_cols)} required features")

# -------------------- APPLICATION TABS --------------------

csv_tab, manual_tab, metrics_tab = st.tabs(
    ["Upload feature CSV", "Test one window", "Evaluation"]
)

# =========================================================
# TAB 1: UPLOAD FEATURE CSV
# =========================================================

with csv_tab:
    st.subheader("Predict stress from feature data")

    st.write(
        "Upload a CSV created by the feature-extraction step "
        "in your Google Colab notebook."
    )

    file = st.file_uploader(
        "Feature CSV",
        type=["csv"],
        key="feature_csv"
    )

    if file is not None:
        try:
            df = pd.read_csv(file)

            missing = [
                column for column in feature_cols
                if column not in df.columns
            ]

            if missing:
                st.error(
                    f"Missing {len(missing)} required feature columns."
                )
                st.write("First missing columns:", missing[:8])

            else:
                X = df[feature_cols].copy()
                X = X.replace([np.inf, -np.inf], np.nan)

                predictions = model.predict(X)

                output = df.copy()
                output["prediction_code"] = predictions
                output["prediction"] = [
                    "Stress" if int(value) == 1 else "Non-Stress"
                    for value in predictions
                ]

                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(X)
                        classes = list(model.classes_)

                        if 1 in classes:
                            output["stress_probability_percent"] = np.round(
                                probabilities[:, classes.index(1)] * 100,
                                2
                            )
                    except Exception:
                        st.warning(
                            "Probability estimates are unavailable for this model."
                        )

                stress_count = int(np.sum(predictions == 1))
                non_stress_count = int(np.sum(predictions == 0))

                col1, col2 = st.columns(2)

                col1.metric("Stress windows", stress_count)
                col2.metric("Non-stress windows", non_stress_count)

                st.subheader("Prediction results")
                st.dataframe(
                    output.head(100),
                    use_container_width=True
                )

                csv_bytes = output.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download predictions CSV",
                    data=csv_bytes,
                    file_name="stress_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================================================
# TAB 2: TEST ONE WINDOW
# =========================================================

with manual_tab:
    st.subheader("Test one feature window")

    st.warning(
        "For a meaningful prediction, enter feature values from an "
        "actual extracted window. All-zero inputs are only a UI test "
        "and may produce an unreliable result."
    )

    with st.form("manual_prediction_form"):
        values = {}
        columns = st.columns(3)

        for index, feature_name in enumerate(feature_cols):
            with columns[index % 3]:
                values[feature_name] = st.number_input(
                    feature_name,
                    value=0.0,
                    format="%.6f",
                    key=f"feature_{index}"
                )

        submitted = st.form_submit_button(
            "Predict",
            type="primary"
        )

    if submitted:
        try:
            row = pd.DataFrame(
                [[values[column] for column in feature_cols]],
                columns=feature_cols
            )

            prediction = int(model.predict(row)[0])

            # Display only the prediction message.
            if prediction == 1:
                st.error("Model output: STRESS (class 1)")
            elif prediction == 0:
                st.success("Model output: NON-STRESS (class 0)")
            else:
                st.warning(f"Model returned an unexpected class: {prediction}")

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(row)[0]
                    classes = list(model.classes_)

                    if 1 in classes:
                        stress_probability = (
                            probabilities[classes.index(1)] * 100
                        )

                        st.metric(
                            "Estimated stress probability",
                            f"{stress_probability:.1f}%"
                        )

                except Exception as e:
                    st.caption(
                        f"Probability estimate unavailable: {e}"
                    )

            st.caption(
                "This is a model estimate, not a clinical assessment."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================================================
# TAB 3: EVALUATION
# =========================================================

with metrics_tab:
    st.subheader("Model evaluation")

    if METRICS.exists():
        try:
            metrics = pd.read_csv(METRICS, index_col=0)

            st.dataframe(
                metrics,
                use_container_width=True
            )

            numeric_metrics = metrics.select_dtypes(include="number")

            if not numeric_metrics.empty:
                st.subheader("Numeric metric comparison")
                st.bar_chart(numeric_metrics)

        except Exception as e:
            st.error(f"Could not load evaluation metrics: {e}")

    else:
        st.warning(
            "evaluation_metrics.csv was not found in artifacts/. "
            "Add this file to display the model comparison."
        )

# -------------------- EXTRA INFORMATION --------------------

with st.expander("Required feature columns"):
    st.write(feature_cols)

st.divider()

st.caption(
    "Future scope: validated wearable sensor integration and "
    "retraining with consented, verified labels."
)

