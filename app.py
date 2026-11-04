
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification System")

st.markdown("""
This application compares five machine learning classification
algorithms using the Breast Cancer Wisconsin Diagnostic dataset.

You can either use the default test dataset available in this
repository or upload your own CSV file.
""")

# =====================================================
# 2. PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FOLDER = BASE_DIR / "models"

DEFAULT_DATASET = BASE_DIR / "test_data.csv"

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "k_nearest_neighbors.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# =====================================================
# 3. DATASET SELECTION
# =====================================================

st.header("1. Dataset Selection")

dataset_option = st.selectbox(
    "Choose your dataset",
    [
        "Default Repository Dataset",
        "Upload Your Own CSV"
    ],
    index=0
)

data = None

if dataset_option == "Default Repository Dataset":

    if DEFAULT_DATASET.exists():

        data = pd.read_csv(DEFAULT_DATASET)

        st.success(
            "Default test dataset loaded successfully from the repository."
        )

        st.caption(
            "Using the test_data.csv file included in this GitHub repository."
        )

    else:
        st.error(
            "Default dataset not found. Please check test_data.csv "
            "in the project repository."
        )

else:

    st.info(
        "Upload your own CSV containing the same 30 numerical "
        "features used during model training."
    )

    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        key="custom_dataset"
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.success("Your CSV file was uploaded successfully.")

# =====================================================
# 4. MODEL SELECTION
# =====================================================

st.header("2. Classification Model Selection")

selected_model = st.selectbox(
    "Choose a classification algorithm",
    list(model_files.keys()),
    index=4
)

# =====================================================
# 5. DATASET PREVIEW
# =====================================================

if data is not None:

    st.header("3. Dataset Preview")

    st.write("Dataset dimensions:", data.shape)

    st.dataframe(
        data.head(),
        use_container_width=True
    )

    # =================================================
    # 6. LOAD MODEL
    # =================================================

    model_path = MODEL_FOLDER / model_files[selected_model]

    if not model_path.exists():

        st.error("Selected trained model was not found.")

    else:

        model = joblib.load(model_path)

        # =============================================
        # 7. PREPARE INPUT DATA
        # =============================================

        if "diagnosis" in data.columns:

            X = data.drop(columns=["diagnosis"]).copy()

            y = data["diagnosis"].replace({
                "B": 0,
                "M": 1,
                "b": 0,
                "m": 1
            })

            if y.isnull().any() or not set(y.unique()).issubset({0, 1}):

                st.error(
                    "Diagnosis must contain only B/M or 0/1 values."
                )
                st.stop()

            y = y.astype(int)

            has_actual_labels = True

        else:

            X = data.copy()

            y = None

            has_actual_labels = False

        # =============================================
        # 8. VALIDATE FEATURES
        # =============================================

        if hasattr(model, "feature_names_in_"):

            expected_features = list(model.feature_names_in_)

            if set(X.columns) != set(expected_features):

                st.error(
                    "Uploaded dataset does not contain the expected "
                    "feature names. Please use the original 30 feature "
                    "columns."
                )
                st.stop()

            X = X[expected_features]

        elif X.shape[1] != 30:

            st.error(
                "The dataset must contain exactly 30 input features."
            )
            st.stop()

        # Check numerical values

        try:
            X = X.apply(pd.to_numeric, errors="raise")

            if X.isnull().any().any():
                st.error("The dataset contains missing feature values.")
                st.stop()

        except (ValueError, TypeError):
            st.error("All 30 input features must be numerical.")
            st.stop()

        # =============================================
        # 9. GENERATE PREDICTIONS
        # =============================================

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        st.header("4. Prediction Results")

        st.write("Selected Model:", selected_model)

        # =============================================
        # 10. DISPLAY PREDICTIONS
        # =============================================

        output = data.copy()

        output["Predicted_Diagnosis"] = pd.Series(
            predictions,
            index=output.index
        ).map({
            0: "Benign",
            1: "Malignant"
        })

        output["Malignant_Probability"] = probabilities.round(4)

        st.dataframe(
            output,
            use_container_width=True
        )

        csv_output = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Results",
            data=csv_output,
            file_name="classification_predictions.csv",
            mime="text/csv"
        )

        # =============================================
        # 11. EVALUATION METRICS
        # =============================================

        if has_actual_labels:

            st.header("5. Model Performance Evaluation")

            accuracy = accuracy_score(y, predictions)

            auc = roc_auc_score(y, probabilities)

            precision = precision_score(
                y, predictions, zero_division=0
            )

            recall = recall_score(
                y, predictions, zero_division=0
            )

            f1 = f1_score(
                y, predictions, zero_division=0
            )

            mcc = matthews_corrcoef(y, predictions)

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC", f"{mcc:.4f}")

            # =========================================
            # 12. CONFUSION MATRIX
            # =========================================

            st.header("6. Confusion Matrix")

            cm = confusion_matrix(
                y,
                predictions,
                labels=[0, 1]
            )

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                ax=ax
            )

            ax.set_xlabel("Predicted Diagnosis")
            ax.set_ylabel("Actual Diagnosis")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)
            plt.close(fig)

            # =========================================
            # 13. CLASSIFICATION REPORT
            # =========================================

            st.header("7. Classification Report")

            report = classification_report(
                y,
                predictions,
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )

            st.success(
                "Classification and model evaluation completed successfully!"
            )

        else:

            st.header("5. Prediction Summary")

            st.info(
                "No diagnosis column was provided. Predictions have "
                "been generated, but evaluation metrics, confusion "
                "matrix and classification report cannot be calculated "
                "without actual diagnosis labels."
            )

else:

    st.info(
        "The application will automatically load the default "
        "repository dataset. Select 'Upload Your Own CSV' above "
        "if you want to use a different dataset."
    )
