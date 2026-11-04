
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

# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification System")

st.write(
    "This application evaluates five machine learning "
    "classification models using the Breast Cancer "
    "Wisconsin Diagnostic dataset."
)

# --------------------------------------------------
# 2. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_FOLDER = Path("models")

model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "k_nearest_neighbors.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# --------------------------------------------------
# 3. CSV FILE UPLOAD
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

# --------------------------------------------------
# 4. MODEL SELECTION
# --------------------------------------------------

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a machine learning algorithm",
    list(model_files.keys())
)

# --------------------------------------------------
# 5. EVALUATION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")
        st.dataframe(data.head())

        if "diagnosis" not in data.columns:
            st.error(
                "The uploaded CSV must contain a diagnosis column."
            )

        else:

            # Separate input features and actual target

            X = data.drop(columns=["diagnosis"])
            y = data["diagnosis"]

            # Convert target labels if necessary

            y = y.replace({
                "B": 0,
                "M": 1,
                "b": 0,
                "m": 1
            })

            if y.isnull().any() or not set(y.unique()).issubset({0, 1}):
                st.error(
                    "Diagnosis must contain only B/M or 0/1 values."
                )

            else:

                model_path = MODEL_FOLDER / model_files[selected_model]

                if not model_path.exists():
                    st.error("Selected model file was not found.")

                else:

                    # Load selected trained model

                    model = joblib.load(model_path)

                    # Check feature compatibility

                    if hasattr(model, "feature_names_in_"):
                        expected_features = list(model.feature_names_in_)

                        if set(X.columns) != set(expected_features):
                            st.error(
                                "Uploaded CSV features do not match "
                                "the selected model."
                            )
                            st.stop()

                        X = X[expected_features]

                    elif X.shape[1] != 30:
                        st.error(
                            "The uploaded dataset must contain "
                            "exactly 30 input features."
                        )
                        st.stop()

                    # Generate predictions

                    predictions = model.predict(X)

                    probabilities = model.predict_proba(X)[:, 1]

                    # Calculate evaluation metrics

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

                    # Display evaluation results

                    st.header("3. Model Performance")

                    st.subheader(selected_model)

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("AUC", f"{auc:.4f}")
                    col3.metric("Precision", f"{precision:.4f}")

                    col4, col5, col6 = st.columns(3)

                    col4.metric("Recall", f"{recall:.4f}")
                    col5.metric("F1 Score", f"{f1:.4f}")
                    col6.metric("MCC", f"{mcc:.4f}")

                    # Display confusion matrix

                    st.header("4. Confusion Matrix")

                    cm = confusion_matrix(
                        y,
                        predictions,
                        labels=[0, 1]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

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

                    # Display classification report

                    st.header("5. Classification Report")

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

                    # Display predictions

                    st.header("6. Prediction Results")

                    output = data.copy()

                    output["Predicted_Diagnosis"] = predictions

                    output["Predicted_Diagnosis"] = (
                        output["Predicted_Diagnosis"]
                        .map({0: "Benign", 1: "Malignant"})
                    )

                    st.dataframe(output, use_container_width=True)

                    st.success(
                        "Model evaluation completed successfully!"
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info(
        "Please upload test_data.csv to evaluate the selected model."
    )
