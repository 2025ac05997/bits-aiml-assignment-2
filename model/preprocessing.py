import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(uploaded_file):
    """Load CSV from uploaded_file (file-like or path) and return X, X_scaled, y.
    For the Absenteeism dataset the target is 'Absenteeism time in hours'.
    Converts target to binary: 0 -> not absent, >0 -> absent (1).
    """
    if hasattr(uploaded_file, "read"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Ensure consistent column names
    df.columns = [c.strip() for c in df.columns]

    # Identify target column
    if "Absenteeism time in hours" in df.columns:
        target_col = "Absenteeism time in hours"
    else:
        target_col = df.columns[-1]

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Create binary target: absent if > 0 hours
    y = (df[target_col] > 0).astype(int)

    # Features: drop target
    X = df.drop(columns=[target_col])

    # Fill missing values
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown")
        else:
            X[col] = X[col].fillna(X[col].median())

    # Encode categorical columns using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y
