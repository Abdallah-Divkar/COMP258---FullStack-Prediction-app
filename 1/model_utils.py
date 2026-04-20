'''import os
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "Student data.csv"
MODEL_FILE = "student_persistence_pipeline.pkl"

COLUMN_NAMES = [
    "first_term_gpa",
    "second_term_gpa",
    "first_language",
    "funding",
    "school",
    "fasttrack",
    "coop",
    "residency",
    "gender",
    "previous_education",
    "age_group",
    "high_school_avg",
    "math_score",
    "english_grade",
    "first_year_persistence",
]

NUMERIC_FEATURES = [
    "first_term_gpa",
    "second_term_gpa",
    "high_school_avg",
    "math_score",
    "english_grade",
]

CATEGORICAL_FEATURES = [
    "first_language",
    "funding",
    "school",
    "fasttrack",
    "coop",
    "residency",
    "gender",
    "previous_education",
    "age_group",
]


def get_base_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))

#Data
def load_dataset() -> pd.DataFrame:
    full_path = os.path.join(get_base_path(), DATA_FILE)
    df = pd.read_csv(full_path, sep=",", skiprows=24, header=None)
    df.columns = COLUMN_NAMES

    df = df.replace("?", np.nan)

    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["first_year_persistence"]).copy()
    df["first_year_persistence"] = df["first_year_persistence"].astype(int)

    return df

#Training
def train_and_save_model():
    df = load_dataset()
    X = df.drop("first_year_persistence", axis=1)
    y = df["first_year_persistence"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "samples": int(len(df)),
    }

    payload = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }

    model_path = os.path.join(get_base_path(), MODEL_FILE)
    joblib.dump(payload, model_path)
    return payload

def load_or_train_model():
    model_path = os.path.join(get_base_path(), MODEL_FILE)
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return train_and_save_model()

def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    network = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("network", network),
        ]
    )

#Prediction
def make_prediction(input_data: dict):
    payload = load_or_train_model()
    pipeline = payload["pipeline"]

    input_df = pd.DataFrame([input_data])
    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return {
        "prediction": prediction,
        "probability_of_persistence": probability,
    }

#Metrics
def get_form_options():
    df = load_dataset()

    options = {}
    for col in CATEGORICAL_FEATURES:
        options[col] = sorted([int(x) for x in df[col].dropna().unique().tolist()])

    defaults = {
        "student_id": "STU-2049",
        "first_term_gpa": float(df["first_term_gpa"].median()),
        "second_term_gpa": float(df["second_term_gpa"].median()),
        "high_school_avg": float(df["high_school_avg"].median()),
        "math_score": float(df["math_score"].median()),
        "english_grade": float(df["english_grade"].median()),
    }

    for col in CATEGORICAL_FEATURES:
        defaults[col] = int(df[col].dropna().mode()[0])

    return {"options": options, "defaults": defaults}

def get_dashboard_data():
    payload = load_or_train_model()
    pipeline = payload["pipeline"]
    metrics = payload["metrics"]

    df = load_dataset().copy()
    X_all = df.drop("first_year_persistence", axis=1)

    probabilities = pipeline.predict_proba(X_all)[:, 1]
    predictions = pipeline.predict(X_all)

    df["predicted_probability"] = probabilities
    df["predicted_class"] = predictions

    at_risk_df = df[df["predicted_probability"] < 0.50].copy()
    at_risk_df = at_risk_df.sort_values("predicted_probability", ascending=True).head(8)

    risk_rows = []
    for i, (_, row) in enumerate(at_risk_df.iterrows(), start=1001):
        risk_rows.append({
            "student": f"STU-{i}",
            "program": f"School {int(row['school'])}",
            "first_term_gpa": round(float(row["first_term_gpa"]), 2) if pd.notna(row["first_term_gpa"]) else 0,
            "persistence_probability": round(float(row["predicted_probability"]) * 100, 1),
            "risk": (
                "High" if row["predicted_probability"] < 0.35
                else "Medium" if row["predicted_probability"] < 0.50
                else "Low"
            ),
        })

    band_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    band_values = [0, 0, 0, 0, 0]

    for p in probabilities:
        if p < 0.2:
            band_values[0] += 1
        elif p < 0.4:
            band_values[1] += 1
        elif p < 0.6:
            band_values[2] += 1
        elif p < 0.8:
            band_values[3] += 1
        else:
            band_values[4] += 1

    program_summary = (
        df.groupby("school")["predicted_probability"]
        .mean()
        .reset_index()
        .sort_values("predicted_probability", ascending=False)
    )

    program_rows = []
    for _, row in program_summary.iterrows():
        program_rows.append({
            "program": f"School {int(row['school'])}",
            "score": round(float(row["predicted_probability"]) * 100, 1)
        })

    dashboard = {
        "summary_cards": {
            "accuracy": round(metrics["accuracy"] * 100, 1),
            "students_analyzed": int(len(df)),
            "at_risk_flagged": int((df["predicted_probability"] < 0.50).sum()),
            "predicted_persistence_rate": round(float(df["predicted_probability"].mean()) * 100, 1),
        },
        "risk_rows": risk_rows,
        "probability_bands": {
            "labels": band_labels,
            "values": band_values,
        },
        "program_rows": program_rows,
    }

    return dashboard
'''