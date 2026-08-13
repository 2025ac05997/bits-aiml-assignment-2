import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Import preprocessing and model modules
from model.preprocessing import load_and_preprocess
from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest
)

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Absenteeism at Work - Classification",
    layout="wide"
)

st.title("Absenteeism at Work – Classification Models")
st.markdown(
    """
    This application demonstrates multiple classification models on the
    **UCI Absenteeism at Work** dataset.
    """
)

# ---------------- INPUT SECTION (COMPACT) ----------------
input_col1, input_col2 = st.columns([1, 1])

with input_col1:
    uploaded_file = st.file_uploader(
        "Upload Absenteeism CSV",
        type="csv"
    )

# Dataset download option
    dataset_path = Path("data/Absenteeism_at_work.csv")
    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            st.download_button(
                label="Download Dataset (CSV)",
                data=f,
                file_name="Absenteeism_at_work.csv",
                mime="text/csv"
            )
    else:
        st.warning("Dataset file not found in data folder.")



with input_col2:
    model_map = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "KNN": knn,
        "Naive Bayes": naive_bayes,
        "Random Forest": random_forest
    }

    selected_model_name = st.selectbox(
        "Select Model",
        list(model_map.keys())
    )

st.divider()

# ---------------------- MAIN LOGIC ----------------------
if uploaded_file:

    try:
        # Load and preprocess data
        X, X_scaled, y = load_and_preprocess(uploaded_file)

        # Train selected model
        model_module = model_map[selected_model_name]
        model = model_module.train_model(X, X_scaled, y)

        # Predictions
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        # ---------------- OUTPUT SECTION (50% / 50%) ----------------
        left_col, right_col = st.columns(2)

        # -------- LEFT COLUMN --------
        with left_col:
            st.markdown("### Evaluation Metrics")

            metrics = {
                "Accuracy": accuracy_score(y, y_pred),
                "AUC": roc_auc_score(y, y_prob),
                "Precision": precision_score(y, y_pred),
                "Recall": recall_score(y, y_pred),
                "F1 Score": f1_score(y, y_pred),
                "MCC": matthews_corrcoef(y, y_pred)
            }

            m1, m2, m3 = st.columns(3)
            metric_items = list(metrics.items())

            for i, (name, value) in enumerate(metric_items):
                if i % 3 == 0:
                    m1.metric(name, round(value, 4))
                elif i % 3 == 1:
                    m2.metric(name, round(value, 4))
                else:
                    m3.metric(name, round(value, 4))

            st.markdown("### Classification Report")

            report_dict = classification_report(
                y,
                y_pred,
                target_names=[
                    "Absenteeism < 8 Hours",
                    "Absenteeism ≥ 8 Hours"
                ],
                output_dict=True
            )

            report_df = pd.DataFrame(report_dict).transpose().round(4)
            st.dataframe(report_df, use_container_width=True)

        # -------- RIGHT COLUMN --------
        with right_col:
            st.markdown("### Confusion Matrix")

            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                xticklabels=["< 8 Hours", "≥ 8 Hours"],
                yticklabels=["< 8 Hours", "≥ 8 Hours"],
                ax=ax
            )

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    except Exception as e:
        st.error("An error occurred while processing the dataset.")
        st.exception(e)

    except Exception as e:
        st.error("An error occurred while processing the file.")
        st.exception(e)