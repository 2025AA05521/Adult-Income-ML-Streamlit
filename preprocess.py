import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def preprocess_data(df, training=True):

    df = df.copy()
    df.replace(" ?", np.nan, inplace=True)
    df.dropna(inplace=True)

    if training:
        encoders = {}

        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        joblib.dump(encoders, "models/encoders.pkl")

        X = df.drop("income", axis=1)
        y = df["income"]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        joblib.dump(scaler, "models/scaler.pkl")

        return X, y

    else:
        encoders = joblib.load("models/encoders.pkl")

        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        X = df.drop("income", axis=1)
        y = df["income"]

        scaler = joblib.load("models/scaler.pkl")
        X = scaler.transform(X)

        return X, y
