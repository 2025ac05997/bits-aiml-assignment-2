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

from sklearn.model_selection import train_test_split

from model.preprocessing import load_and_preprocess
from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Absenteeism at Work - Classification",
    layout="wide"
)

st.title("Absenteeism at Work - Classification Models")

st.markdown("""
This application demonstrates multiple machine learning classification
models on the UCI Absenteeism at Work dataset.
""")

# ---------------- INPUT SECTION ---------------- #

input_col1, input_col2 = st.columns(2)

with input_col1:

    uploaded_file = st.file_uploader(
        "Upload Dataset CSV",
        type=["csv"]
    )

    dataset_path = Path("data/Absenteeism_at_work.csv")

    if dataset_path.exists():

        with open(dataset_path, "rb") as f:

            st.download_button(
                label="Download Sample Dataset",
                data=f,
                file_name="Absenteeism_at_work.csv",
                mime="text/csv"
            )

with input_col2:

    selected_model_name = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest"
        ]
    )

st.divider()

# ---------------- MAIN LOGIC ---------------- #

if uploaded_file:

    try:

        # Load data
        X, X_scaled, y = load_and_preprocess(uploaded_file)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        model_map = {
            "Logistic Regression": logistic_regression,
            "Decision Tree": decision_tree,
            "KNN": knn,
            "Naive Bayes": naive_bayes,
            "Random Forest": random_forest
        }

        # ==================================================
        # COMPARISON TABLE FOR ALL MODELS
        # ==================================================

        comparison_results = []

        for model_name, model_module in model_map.items():

            model = model_module.train_model(
                X_train,
                X_train,
                y_train
            )

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred

            comparison_results.append({
                "ML Model": model_name,
                "Accuracy": round(
                    accuracy_score(y_test, y_pred),
                    4
                ),
                "AUC": round(
                    roc_auc_score(y_test, y_prob),
                    4
                ),
                "Precision": round(
                    precision_score(y_test, y_pred),
                    4
                ),
                "Recall": round(
                    recall_score(y_test, y_pred),
                    4
                ),
                "F1": round(
                    f1_score(y_test, y_pred),
                    4
                ),
                "MCC": round(
                    matthews_corrcoef(y_test, y_pred),
                    4
                )
            })

        comparison_df = pd.DataFrame(comparison_results)

        st.subheader("Model Comparison")

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        best_model_name = comparison_df.loc[
            comparison_df["F1"].idxmax(),
            "ML Model"
        ]

        st.success(
            f"Best Performing Model (Based on F1 Score): "
            f"{best_model_name}"
        )

        st.divider()

        # ==================================================
        # INDIVIDUAL MODEL ANALYSIS
        # ==================================================

        model_module = model_map[selected_model_name]

        model = model_module.train_model(
            X_train,
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        left_col, right_col = st.columns(2)

        # ---------------- LEFT ---------------- #

        with left_col:

            st.subheader(
                f"{selected_model_name} Metrics"
            )

            metrics = {
                "Accuracy":
                    accuracy_score(y_test, y_pred),
                "AUC":
                    roc_auc_score(y_test, y_prob),
                "Precision":
                    precision_score(y_test, y_pred),
                "Recall":
                    recall_score(y_test, y_pred),
                "F1 Score":
                    f1_score(y_test, y_pred),
                "MCC":
                    matthews_corrcoef(y_test, y_pred)
            }

            c1, c2, c3 = st.columns(3)

            metric_list = list(metrics.items())

            for i, (name, value) in enumerate(metric_list):

                if i % 3 == 0:
                    c1.metric(name, round(value, 4))

                elif i % 3 == 1:
                    c2.metric(name, round(value, 4))

                else:
                    c3.metric(name, round(value, 4))

            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

        # ---------------- RIGHT ---------------- #

        with right_col:

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            fig, ax = plt.subplots(
                figsize=(5, 4)
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Greens",
                ax=ax
            )

            ax.set_xlabel(
                "Predicted"
            )

            ax.set_ylabel(
                "Actual"
            )

            st.pyplot(fig)

    except Exception as e:

        st.error(
            "An error occurred while processing the file."
        )

        st.exception(e)