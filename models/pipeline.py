#models/pipeline.py

import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from services.data_service import load_dataset

MODEL_FILE = "student_persistence_pipeline.pkl"

def load_model():
    path = os.path.join(os.path.dirname(__file__), "..", MODEL_FILE)
    return joblib.load(path)

'''NUMERIC_FEATURES = [
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



def save_model(model):
    model_path = os.path.join(get_base_path(), MODEL_FILE)
    joblib.dump(model, model_path)

def train_model():
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

    model = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    }
    save_model(model)
    return model

from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, model):
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

MODEL_PAYLOAD = load_model()'''