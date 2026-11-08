from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FOLDER = BASE_DIR / "model"
DEFAULT_DATASET = BASE_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "k_nearest_neighbors.joblib",
    "Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

LABEL_NAMES = {0: "Benign", 1: "Malignant"}
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def calculate_metrics(y_true, y_pred, y_probability):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_probability),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def prepare_dataset(data: pd.DataFrame):
    """Validate the labelled test CSV and return ordered numeric features and labels."""
    if "diagnosis" not in data.columns:
        st.error(
            "The test CSV must contain a 'diagnosis' target column (B/M or 0/1) "
            "as well as all 30 input features. Please upload a labelled test dataset."
        )
        st.stop()

    y = data["diagnosis"].replace(
        {"B": 0, "M": 1, "b": 0, "m": 1, "0": 0, "1": 1}
    )
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        st.error("The diagnosis column must contain only B/M or 0/1 values.")
        st.stop()
    y = y.astype(int)

    X = data.drop(columns=["diagnosis"]).copy()

    if "id" in X.columns:
        st.error(
            "Remove the identifier column 'id'. The trained models require exactly "
            "the 30 diagnostic feature columns."
        )
        st.stop()

    # The trained pipelines retain the feature names used during training.
    reference_model_path = MODEL_FOLDER / MODEL_FILES["Logistic Regression"]
    if not reference_model_path.exists():
        st.error(f"Required model file not found: {reference_model_path.name}")
        st.stop()

    reference_model = load_model(str(reference_model_path))
    expected_features = list(reference_model.feature_names_in_)

    missing = [col for col in expected_features if col not in X.columns]
    extra = [col for col in X.columns if col not in expected_features]
    if missing or extra:
        st.error(
            "The CSV columns do not match the training features. "
            f"Missing columns: {missing or 'None'}. "
            f"Unexpected columns: {extra or 'None'}."
        )
        st.stop()

    X = X[expected_features]
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        st.error("All 30 input features must contain numerical values.")
        st.stop()

    if X.isna().any().any():
        st.error("The input features contain missing values. Please clean the CSV.")
        st.stop()

    return X, y


st.title("Breast Cancer Wisconsin Diagnostic – Model Comparison")
st.write(
    "Compare five trained classification models on the same labelled test dataset. "
    "The application loads saved models; it does not retrain them on uploaded data."
)

st.header("1. Dataset")
dataset_option = st.radio(
    "Choose the test dataset",
    ["Default repository test data", "Upload a labelled test CSV"],
    horizontal=True,
)

data = None
if dataset_option == "Default repository test data":
    if DEFAULT_DATASET.exists():
        data = pd.read_csv(DEFAULT_DATASET)
        st.success("Loaded test_data.csv from the project repository.")
    else:
        st.error("test_data.csv was not found in the repository.")
else:
    uploaded_file = st.file_uploader(
        "Upload a CSV containing the 30 feature columns and diagnosis",
        type=["csv"],
    )
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("CSV uploaded.")
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")

if data is not None:
    st.caption(
        "Required target values: B (Benign) / M (Malignant), or 0 / 1. "
        "Do not include the ID column."
    )
    st.write(f"Test dataset dimensions: **{data.shape[0]} rows × {data.shape[1]} columns**")
    with st.expander("Preview test data"):
        st.dataframe(data.head(), use_container_width=True)

    X, y = prepare_dataset(data)

    # Evaluate every saved model on exactly the same uploaded/default test rows.
    st.header("2. Comparison of all five models")
    all_predictions = {}
    comparison_rows = []

    for model_name, filename in MODEL_FILES.items():
        model_path = MODEL_FOLDER / filename
        if not model_path.exists():
            st.error(f"Saved model is missing: {filename}")
            st.stop()

        current_model = load_model(str(model_path))
        model_predictions = current_model.predict(X)
        model_probabilities = current_model.predict_proba(X)[:, 1]
        all_predictions[model_name] = (model_predictions, model_probabilities)

        row = {"Model": model_name}
        row.update(calculate_metrics(y, model_predictions, model_probabilities))
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    st.dataframe(
        comparison_df[METRIC_COLUMNS].style.format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(
        "All metrics above are calculated at runtime on the currently selected "
        "test dataset, not copied from the training results."
    )

    best_model = comparison_df["Accuracy"].idxmax()
    st.write(f"**Highest accuracy on this test dataset:** {best_model}")

    st.header("3. Select a model to inspect")
    selected_model = st.selectbox(
        "Classification algorithm",
        list(MODEL_FILES.keys()),
        index=list(MODEL_FILES.keys()).index("Support Vector Machine"),
    )

    predictions, probabilities = all_predictions[selected_model]
    selected_metrics = comparison_df.loc[selected_model]

    st.subheader(f"Prediction results – {selected_model}")
    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["Accuracy", "AUC", "Precision"]):
        col.metric(metric, f"{selected_metrics[metric]:.4f}")
    metric_cols = st.columns(3)
    for col, metric in zip(metric_cols, ["Recall", "F1 Score", "MCC"]):
        col.metric(metric, f"{selected_metrics[metric]:.4f}")

    output = data.copy()
    output["Actual_Diagnosis"] = y.map(LABEL_NAMES).values
    output["Predicted_Diagnosis"] = pd.Series(predictions, index=output.index).map(
        LABEL_NAMES
    )
    output["Malignant_Probability"] = probabilities.round(4)

    st.dataframe(output, use_container_width=True)
    st.download_button(
        "Download prediction results",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv",
    )

    st.header("4. Confusion matrix")
    cm = confusion_matrix(y, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax,
    )
    ax.set_xlabel("Predicted diagnosis")
    ax.set_ylabel("Actual diagnosis")
    ax.set_title(f"Confusion Matrix – {selected_model}")
    st.pyplot(fig)
    plt.close(fig)

    st.header("5. Classification report")
    report = classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.success(
        "Evaluation completed. Each of the five saved models was evaluated on "
        "the same test dataset without retraining."
    )
else:
    st.info("Choose the default test dataset or upload a labelled test CSV to begin.")
