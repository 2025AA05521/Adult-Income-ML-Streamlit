import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def preprocess_data(df, training=True):

    df = df.copy()
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)

    encoders = {}

    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    if training:
        joblib.dump(encoders, "models/encoders.pkl")

    X = df.drop("income", axis=1)
    y = df["income"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if training:
        joblib.dump(scaler, "models/scaler.pkl")

    return X, y
