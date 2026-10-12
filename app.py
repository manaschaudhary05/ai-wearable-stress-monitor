import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Wearable Stress Monitor", page_icon="💗", layout="wide")
ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
MODEL, FEATURES, METRICS = ART / "best_model.joblib", ART / "feature_columns.joblib", ART / "evaluation_metrics.csv"

st.title("💗 AI-Based Wearable Stress Monitoring")
st.caption("WESAD research prototype • ML stress classification")
st.info("Research prototype only—not a medical/diagnostic tool. Upload features generated using the same preprocessing pipeline as training.")

@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL), list(joblib.load(FEATURES))

if not MODEL.exists() or not FEATURES.exists():
    st.error("Model files not found. Put best_model.joblib and feature_columns.joblib in the artifacts/ folder.")
    st.code("artifacts/best_model.joblib\nartifacts/feature_columns.joblib")
    st.stop()

try:
    model, feature_cols = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

st.success(f"Model loaded — {len(feature_cols)} required features")
csv_tab, manual_tab, metrics_tab = st.tabs(["Upload feature CSV", "Test one window", "Evaluation"])

with csv_tab:
    st.write("Upload a CSV created by the feature-extraction step in your Colab notebook.")
    file = st.file_uploader("Feature CSV", type="csv")
    if file:
        df = pd.read_csv(file)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing {len(missing)} required feature columns. First missing: {missing[:8]}")
        else:
            X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            try:
                pred = model.predict(X)
                out = df.copy()
                out["prediction_code"] = pred
                out["prediction"] = ["Stress" if int(x) == 1 else "Non-Stress" for x in pred]
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(X)
                        classes = list(model.classes_)
                        if 1 in classes:
                            out["stress_probability_percent"] = np.round(probs[:, classes.index(1)] * 100, 2)
                    except Exception:
                        pass
                a, b = st.columns(2)
                a.metric("Stress windows", int(np.sum(pred == 1)))
                b.metric("Non-stress windows", int(np.sum(pred == 0)))
                st.dataframe(out.head(100), use_container_width=True)
                st.download_button("Download predictions CSV", out.to_csv(index=False).encode(), "stress_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with manual_tab:
    st.write("For a quick UI test, enter feature values from one extracted window.")
    with st.form("manual"):
        cols = st.columns(3)
        values = {}
        for i, name in enumerate(feature_cols):
            with cols[i % 3]:
                values[name] = st.number_input(name, value=0.0, format="%.6f", key=f"v{i}")
        submit = st.form_submit_button("Predict", type="primary")
    if submit:
        row = pd.DataFrame([[values[c] for c in feature_cols]], columns=feature_cols)
        try:
            p = int(model.predict(row)[0])
            st.error("Model output: STRESS (class 1)") if p == 1 else st.success("Model output: NON-STRESS (class 0)")
            if hasattr(model, "predict_proba"):
                try:
                    probs, classes = model.predict_proba(row)[0], list(model.classes_)
                    if 1 in classes:
                        st.metric("Estimated stress probability", f"{probs[classes.index(1)] * 100:.1f}%")
                except Exception:
                    pass
            st.caption("Model estimate only; not a clinical assessment.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with metrics_tab:
    if METRICS.exists():
        metrics = pd.read_csv(METRICS, index_col=0)
        st.dataframe(metrics, use_container_width=True)
        numeric = metrics.select_dtypes(include="number")
        if not numeric.empty:
            st.bar_chart(numeric)
    else:
        st.warning("Optional evaluation_metrics.csv not found in artifacts/. Add it to display the model comparison.")

with st.expander("Required feature columns"):
    st.write(feature_cols)

st.divider()
st.caption("Future scope: validated wearable sensor integration and retraining with consented, verified labels.")
