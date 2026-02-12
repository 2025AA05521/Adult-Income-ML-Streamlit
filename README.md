# 📊 Adult Income Classification Dashboard (Streamlit + Machine Learning)

## 🔹 Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether an individual earns more than $50K annually based on demographic and employment-related attributes.

This project demonstrates an end-to-end Machine Learning workflow including data preprocessing, model training, evaluation, visualization, and deployment using Streamlit.

---

## 🔹 Dataset Description

The dataset used in this project is the **Adult Income Dataset** obtained from public repositories (UCI ML Repository / Kaggle).

### Dataset Details:

- Total Instances: 48,842  
- Number of Features: 14  
- Problem Type: Binary Classification  
- Target Variable: Income  
  - <=50K
  - >50K  

### Feature Examples:

- Age
- Workclass
- Education
- Occupation
- Marital Status
- Capital Gain
- Capital Loss
- Hours Per Week
- Native Country

---

## 🔹 Machine Learning Models Implemented

The following classification models were implemented and evaluated on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Boosting Model)

---

## 🔹 Evaluation Metrics

Each model was evaluated using the following performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 🔹 Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|-----------|------|------------|----------|------------|------|
| Logistic Regression | (Add Output) |
| Decision Tree | (Add Output) |
| KNN | (Add Output) |
| Naive Bayes | (Add Output) |
| Random Forest | (Add Output) |
| XGBoost | (Add Output) |

*(Values obtained from model training output)*

---

## 🔹 Model Performance Observations

| Model | Observation |
|----------|--------------|
| Logistic Regression | Performs well for linearly separable relationships and provides stable baseline performance. |
| Decision Tree | Easy to interpret but prone to overfitting on training data. |
| KNN | Sensitive to feature scaling and computationally intensive for large datasets. |
| Naive Bayes | Fast and efficient but assumes independence between features. |
| Random Forest | Provides strong accuracy and reduces overfitting using ensemble learning. |
| XGBoost | Produces high performance using gradient boosting and regularization techniques. |

---

## 🔹 Streamlit Application Features

The interactive Streamlit dashboard provides the following user-friendly features:

### 📌 Sidebar Controls
- Dataset download option
- CSV test dataset upload
- Machine learning model selection

### 📊 Visualization Features
- Dataset preview section
- Performance summary metrics (Accuracy, Precision, Recall, F1 Score)
- Detailed classification report table
- Confusion matrix heatmap visualization

### 📥 Additional Functionalities
- Download prediction results as CSV
- Interactive expandable sections
- Real-time prediction processing

---

## 🔹 Project Folder Structure

