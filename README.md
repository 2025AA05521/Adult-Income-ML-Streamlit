# Adult Income Classification using Machine Learning and Streamlit

---

## a. Problem Statement

The objective of this project is to develop and compare multiple Machine Learning classification models to predict whether an individual earns more than 50K annually based on demographic and employment-related attributes.

The project demonstrates a complete Machine Learning pipeline including data preprocessing, model training, evaluation, visualization, and deployment using an interactive Streamlit web application.

---

## b. Dataset Description

The dataset used in this project is the **Adult Income Dataset**, obtained from public repositories such as Kaggle and UCI Machine Learning Repository.

### Dataset Characteristics

- Problem Type: Binary Classification
- Number of Instances: 48,842
- Number of Features: 14
- Target Variable: Income Category

### Target Classes:
- <=50K
- >50K

### Features Include:
- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

The dataset contains both categorical and numerical features. Label encoding and feature scaling were applied during preprocessing.

---

## c. Models Used

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble Model)
6. XGBoost (Ensemble Model)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----------|-----------|--------|----------|-----|
| Logistic Regression | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |
| Decision Tree | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |
| KNN | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |
| Naive Bayes | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |
| Random Forest (Ensemble) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |
| XGBoost (Ensemble) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) | (Add Value) |

*Values were obtained after evaluating trained models on the test dataset.*

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Provides stable baseline performance and works well when relationship between features and target is linear. |
| Decision Tree | Highly interpretable model but prone to overfitting if tree depth is large. |
| KNN | Performance depends heavily on feature scaling and can be computationally expensive for large datasets. |
| Naive Bayes | Fast and efficient but assumes independence among features which may reduce performance in complex datasets. |
| Random Forest (Ensemble) | Provides high accuracy and reduces overfitting by combining multiple decision trees. |
| XGBoost (Ensemble) | Achieves strong performance using gradient boosting and regularization, generally providing the best overall results. |

---

## Streamlit Application Features

The project includes an interactive Streamlit dashboard with the following functionalities:

### Sidebar Features:
- Dataset download option
- CSV dataset upload functionality
- Machine Learning model selection dropdown

### Visualization and Output Features:
- Dataset preview display
- Performance summary metrics (Accuracy, Precision, Recall, F1 Score)
- Detailed classification report displayed in table format
- Confusion matrix displayed using heatmap visualization
- Prediction results download option

---

## Project Folder Structure
## Project Folder Structure

```
adult-income-ml-streamlit/
│
├── app.py
├── adult.csv
├── train_models.py
├── preprocess.py
├── requirements.txt
├── README.md
│
└── models/
    ├── logistic.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── encoders.pkl
```

