import streamlit as st
import pandas as pd
import joblib

from preprocess import preprocess_data
from sklearn.metrics import classification_report, confusion_matrix

st.title("Adult Income Classification Models")

uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

model_name = st.selectbox(
    "Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    X, y = preprocess_data(df, training=False)

    model = joblib.load(f"models/{model_name}.pkl")

    predictions = model.predict(X)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, predictions))
