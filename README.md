# Breast Cancer Wisconsin Diagnostic – Classification Models

## 1. Problem Statement

The objective of this project is to develop and compare five machine learning classification algorithms to predict whether a breast tumour is benign or malignant using diagnostic measurements.

The project uses the Breast Cancer Wisconsin Diagnostic dataset. Since this is a medical classification problem, identifying malignant cases correctly is particularly important. Therefore, recall is an important evaluation metric because a false negative means that a malignant tumour is incorrectly classified as benign.

## 2. Dataset Description

- **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
- **Source:** UCI Machine Learning Repository
- **Problem Type:** Binary Classification
- **Total Instances:** 569
- **Input Features:** 30 numerical features
- **Target Variable:** Diagnosis
- **Target Classes:** Benign (B) and Malignant (M)

The dataset contains measurements such as radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry.

The identifier column was excluded because it does not contribute meaningful predictive information.

The target variable was encoded as:
- 0 – Benign
- 1 – Malignant

Dataset source: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data

## 3. GitHub Repository Link

**Repository:** [INSERT YOUR GITHUB REPOSITORY URL]

The repository contains the Jupyter Notebook, Streamlit application, saved classification models, test dataset and supporting documentation.

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset from the UCI Machine Learning Repository.
2. Removed the identifier column.
3. Encoded the diagnosis variable into numerical values.
4. Checked missing values and duplicate records.
5. Separated input features and the target variable.
6. Split the dataset into training and testing sets using an 80:20 ratio.
7. Used stratified sampling to maintain the class distribution.
8. Applied StandardScaler to Logistic Regression, KNN and SVM through pipelines.

The train-test split used random_state=42 for reproducibility.

## 5. Classification Models and Evaluation

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours
4. Gaussian Naive Bayes
5. Support Vector Machine

The models were evaluated using six performance metrics.

### Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

**Logistic Regression**

Logistic Regression achieved an accuracy of 96.49% and an AUC score of 0.9960. It provided a good balance between precision and recall. Its recall of 92.86% indicates that it identified most malignant cases correctly.

**Decision Tree**

The Decision Tree achieved an accuracy of 92.11%. Its performance was comparatively lower than the other models, with a recall of 83.33%. Although the model is relatively easy to interpret, its overall classification performance was weaker in this experiment.

**K-Nearest Neighbours**

KNN achieved an accuracy of 95.61% and an F1 score of 0.9383. It performed reasonably well after feature scaling. Its recall of 90.48% indicates that it identified most of the malignant cases, although it performed slightly below Logistic Regression and SVM.

**Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 93.86% and an AUC score of 0.9934. It recorded perfect precision, meaning that none of its malignant predictions were false positives in this test set. However, its recall was 83.33%, indicating that some actual malignant cases were missed.

**Support Vector Machine**

SVM achieved the highest accuracy of 97.37%, with an F1 score of 0.9630 and an MCC score of 0.9442. It also achieved perfect precision and a recall of 92.86%. Overall, it provided the strongest classification performance among the five models in this experiment.

### Overall Winner

The Support Vector Machine was the best-performing model based on the overall comparison.

It achieved the highest accuracy, precision, F1 score and MCC among the five classifiers. Logistic Regression recorded a slightly higher AUC score, but SVM performed better across most of the other evaluation metrics.

Based on these results, SVM is selected as the overall winner for this dataset.

These results represent a machine learning experiment and should not be interpreted as clinical validation of a diagnostic system.

## 6. Streamlit Application

**Live Application:** [INSERT YOUR STREAMLIT APP URL]

The Streamlit application provides the following features:

- CSV test dataset upload.
- Selection of any of the five trained classification models.
- Display of accuracy, AUC, precision, recall, F1 score and MCC.
- Confusion matrix visualisation.
- Classification report.
- Prediction results.

The application loads the previously trained models and evaluates the uploaded test dataset without retraining.

## 7. Project Structure

```text
aiml_assignment2/
│
├── app.py
├── assignment2.ipynb
├── requirements.txt
├── README.md
├── test_data.csv
├── model_metrics.csv
│
├── data/
│   ├── wdbc.data
│   └── README.txt
│
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── k_nearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## 8. How to Run the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

Upload `test_data.csv`, select a classification model and view its evaluation results.

## 9. Conclusion

This project implemented and compared five machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic dataset. All models were evaluated using six performance metrics on the same test dataset.

The experiment showed that Support Vector Machine achieved the best overall results, while Logistic Regression also demonstrated strong classification performance.

The project also demonstrates how trained machine learning models can be integrated into a Streamlit application to provide an interactive evaluation interface.