"""
train_model.py
----------------
Trains a customer churn prediction model on the Telco Customer Churn dataset.

Dataset: Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Place it in the data/ folder before running this script.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV and clean it up."""
    df = pd.read_csv(path)

    # TotalCharges has some blank strings instead of NaN — fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customer ID — not a predictive feature
    df = df.drop(columns=["customerID"])

    return df


def preprocess(df: pd.DataFrame):
    """Encode categorical columns and split features/target."""
    df = df.copy()

    # Target column: Yes/No -> 1/0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y, encoders


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}.\n"
            "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from "
            "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and place it in the data/ folder."
        )

    print("Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)

    print("Preprocessing...")
    X, y, encoders = preprocess(df)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss"
        ),
    }

    results = {}
    best_model_name = None
    best_f1 = -1
    best_model = None

    print("\nTraining and evaluating models...\n" + "=" * 50)
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": auc}

        print(f"\n{name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print("\n" + "=" * 50)
    print(f"Best model: {best_model_name} (F1-score: {best_f1:.4f})")

    print("\nClassification report for best model:")
    best_preds = best_model.predict(X_test_scaled)
    print(classification_report(y_test, best_preds))

    # Save everything needed for the Streamlit app
    joblib.dump(best_model, MODEL_DIR / "churn_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(
            {"best_model": best_model_name, "results": results}, f, indent=2
        )

    print(f"\nModel, scaler, encoders, and metrics saved to '{MODEL_DIR}/'")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
