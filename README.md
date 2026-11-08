# Breast Cancer Wisconsin (Diagnostic) – Classification Models

**Programme:** M.Tech Artificial Intelligence and Machine Learning, BITS Pilani
**Course:** Machine Learning – Assignment 2
**Student:** Rohal Kurup | **BITS ID:** 2026AJ05995

- GitHub repository: https://github.com/Rohal25/breast-cancer-classification-models
- Live Streamlit app: https://2026aj05995.streamlit.app/

---

## a. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms that predict whether a breast tumour is **benign** or **malignant** from diagnostic measurements.

Because this is a medical classification problem, identifying malignant cases correctly is particularly important. Recall is therefore a key evaluation metric: a false negative means a malignant tumour is classified as benign.

## b. Dataset Description

| Item | Detail |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository |
| Problem type | Binary classification |
| Instances | 569 (357 benign, 212 malignant) |
| Input features | 30 numerical features |
| Target | `diagnosis` – Benign (B) / Malignant (M) |

The 30 features are the mean, standard error and worst value of ten nuclear measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).

The identifier column was excluded because it carries no predictive information. The target was encoded as **0 = Benign, 1 = Malignant**.

Dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
A copy is stored in the repository at `data/wdbc.data`.

## c. GitHub Repository

https://github.com/Rohal25/breast-cancer-classification-models

The repository contains the Jupyter Notebook, the Streamlit application, `requirements.txt`, this README, the test dataset, five saved models and the evaluation results.

### Project structure

```
breast-cancer-classification-models/
├── app.py                      # Streamlit application
├── assignment2.ipynb           # Data analysis, training and evaluation
├── requirements.txt
├── README.md
├── test_data.csv               # Held-out test set (114 rows, 30 features + diagnosis)
├── model_metrics.csv           # Evaluation results for all five models
├── data/
│   ├── wdbc.data
│   └── README.txt
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

### Data preprocessing

1. Loaded the dataset from the UCI repository.
2. Removed the identifier column.
3. Encoded the diagnosis (0 = Benign, 1 = Malignant).
4. Checked for missing values (none) and duplicate records.
5. Separated the input features and the target.
6. Split into training and test sets (80:20, `random_state=42`).
7. Used stratified sampling to preserve the class distribution (455 training / 114 test rows).
8. Applied `StandardScaler` to Logistic Regression, KNN and SVM through scikit-learn pipelines (so the scaler is fitted on training data only).

## d. Models Used

Models: Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes, Support Vector Machine. All were trained on the same split and evaluated on the same 114-row test set.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 96.49% and the highest AUC (0.9960). Precision (0.9750) and recall (0.9286) are well balanced: it missed 3 of 42 malignant cases and raised 1 false alarm. A strong, simple baseline on scaled data. |
| Decision Tree | Weakest model: accuracy 92.11%, recall 83.33% (7 malignant cases missed) and lowest AUC (0.9448). Easy to interpret, but a single depth-limited tree generalised less well than the other models here. |
| KNN | Accuracy 95.61%, F1 0.9383, recall 90.48% (4 malignant cases missed). Performed well after feature scaling, slightly below Logistic Regression and SVM. |
| Naive Bayes | Accuracy 93.86% and a high AUC (0.9934). Precision is perfect (no false positives), but recall is only 83.33% (7 malignant cases missed), so it is the riskier choice when missing a malignant case is costly. |
| Support Vector Machine | Highest accuracy (97.37%), F1 (0.9630) and MCC (0.9442). Precision is perfect (1.0) and recall is 0.9286, matching Logistic Regression (3 malignant cases missed, 0 false alarms). |
| **Overall winner** | **Support Vector Machine.** It has the highest accuracy, F1 and MCC, ties Naive Bayes for the highest precision and ties Logistic Regression for the highest recall. Logistic Regression has a marginally higher AUC (0.9960 vs 0.9947). |

**Note on reliability:** the test set has only 114 samples, so the gap between SVM and Logistic Regression is one patient. Treat the ranking of the top models as indicative rather than conclusive. These results come from a machine learning experiment and are not clinical validation of a diagnostic system.

## Streamlit Application

Live app: https://2026aj05995.streamlit.app/

**Dataset options**

1. **Default repository dataset** – loads `test_data.csv` from the repository automatically.
2. **Upload your own CSV** – the file must contain the 30 numerical input features (column names exactly as in `test_data.csv`, no `id` column) and a target column named `diagnosis` with values **B/M or 0/1** (0 = Benign, 1 = Malignant). The app validates the file and shows a clear error if columns or values are wrong.

**Features**

- CSV test-data upload
- Model selection dropdown (SVM selected by default)
- Predictions
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison table of all five models

The app loads the saved models from `model/` and predicts on the uploaded data **without retraining**.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the default `test_data.csv` or upload another labelled test CSV, choose a model, and inspect its metrics and confusion matrix.

## Conclusion

Five classifiers were compared on the Breast Cancer Wisconsin (Diagnostic) dataset using six metrics on the same test set. Support Vector Machine gave the best overall results, with Logistic Regression close behind. The project also shows how trained models can be saved and served through a Streamlit app for interactive evaluation.
