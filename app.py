import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# ---------------- Page Setup ---------------- #

st.set_page_config(page_title="Adult Income ML Dashboard", layout="wide")
st.title("📊 Adult Income Classification Dashboard")


# ---------------- Sidebar ---------------- #

st.sidebar.header("📌 User Guide")

st.sidebar.markdown("""
### Step 1
Download dataset

### Step 2
Upload CSV file

### Step 3
Select ML model

### Step 4
View predictions & performance
""")

# Dataset Download
try:
    with open("adult.csv", "rb") as file:
        st.sidebar.download_button(
            label="⬇ Download Adult Dataset",
            data=file,
            file_name="adult.csv"
        )
except:
    st.sidebar.warning("Dataset not found in project folder")


# Upload
uploaded_file = st.sidebar.file_uploader("📤 Upload Test Dataset", type="csv")

# Model Selection
model_name = st.sidebar.selectbox(
    "🤖 Select Model",
    ["logistic", "decision_tree", "knn", "naive_bayes", "random_forest", "xgboost"]
)


# ---------------- Main Panel ---------------- #

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    # Preview Section
    with st.expander("📂 Dataset Preview", expanded=True):
        st.dataframe(df.head())

    # Preprocess
    X, y = preprocess_data(df, training=False)

    model = joblib.load(f"models/{model_name}.pkl")

    with st.spinner("Running prediction..."):
        predictions = model.predict(X)

    st.success("Prediction completed!")

    # ---------------- Metric Cards ---------------- #

    acc = accuracy_score(y, predictions)
    prec = precision_score(y, predictions)
    rec = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    st.subheader("📊 Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")


    # ---------------- Classification Report ---------------- #

    report_dict = classification_report(y, predictions, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    with st.expander("📋 Detailed Classification Report", expanded=True):
        st.dataframe(report_df.style.format("{:.2f}"))

    # ---------------- Confusion Matrix ---------------- #

    cm = confusion_matrix(y, predictions)

    st.subheader("📉 Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


    # ---------------- Download Predictions ---------------- #

    result_df = df.copy()
    result_df["Predicted Income"] = predictions

    csv = result_df.to_csv(index=False)

    st.download_button(
        "⬇ Download Prediction Results",
        csv,
        "prediction_results.csv",
        "text/csv"
    )

else:
    st.info("👈 Please upload dataset from sidebar to begin.")
