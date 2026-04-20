
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "Student data.csv"
MODEL_FILE = "student_persistence_model.pkl"

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


def load_dataset() -> pd.DataFrame:
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, DATA_FILE)

    df = pd.read_csv(full_path, sep=",", skiprows=24, header=None)
    df.columns = COLUMN_NAMES

    df = df.replace("?", np.nan)

    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["first_year_persistence"]).copy()
    df["first_year_persistence"] = df["first_year_persistence"].astype(int)

    return df


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

    model = MLPClassifier(
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

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


@st.cache_data
def prepare_data():
    df = load_dataset()
    X = df.drop("first_year_persistence", axis=1)
    y = df["first_year_persistence"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return df, X_train, X_test, y_train, y_test


@st.cache_resource
def train_model():
    df, X_train, X_test, y_train, y_test = prepare_data()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    return pipeline, metrics


def save_model(pipeline: Pipeline):
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, MODEL_FILE)
    joblib.dump(pipeline, model_path)
    return model_path


def get_default_input(df: pd.DataFrame) -> dict:
    return {
        "first_term_gpa": float(df["first_term_gpa"].median()),
        "second_term_gpa": float(df["second_term_gpa"].median()),
        "first_language": int(df["first_language"].dropna().mode()[0]),
        "funding": int(df["funding"].dropna().mode()[0]),
        "school": int(df["school"].dropna().mode()[0]),
        "fasttrack": int(df["fasttrack"].dropna().mode()[0]),
        "coop": int(df["coop"].dropna().mode()[0]),
        "residency": int(df["residency"].dropna().mode()[0]),
        "gender": int(df["gender"].dropna().mode()[0]),
        "previous_education": int(df["previous_education"].dropna().mode()[0]),
        "age_group": int(df["age_group"].dropna().mode()[0]),
        "high_school_avg": float(df["high_school_avg"].median()),
        "math_score": float(df["math_score"].median()),
        "english_grade": float(df["english_grade"].median()),
    }


def build_input_form(df: pd.DataFrame) -> pd.DataFrame:
    defaults = get_default_input(df)

    st.subheader("Enter student data for prediction")

    col1, col2 = st.columns(2)

    with col1:
        first_term_gpa = st.number_input("First Term GPA", 0.0, 5.0, defaults["first_term_gpa"], 0.01)
        second_term_gpa = st.number_input("Second Term GPA", 0.0, 5.0, defaults["second_term_gpa"], 0.01)
        high_school_avg = st.number_input("High School Average", 0.0, 100.0, defaults["high_school_avg"], 1.0)
        math_score = st.number_input("Math Score", 0.0, 100.0, defaults["math_score"], 1.0)
        english_grade = st.number_input("English Grade", 0.0, 12.0, defaults["english_grade"], 1.0)
        first_language = st.selectbox("First Language (coded)", sorted(df["first_language"].dropna().unique().astype(int)), index=0)
        funding = st.selectbox("Funding (coded)", sorted(df["funding"].dropna().unique().astype(int)), index=0)

    with col2:
        school = st.selectbox("School (coded)", sorted(df["school"].dropna().unique().astype(int)), index=0)
        fasttrack = st.selectbox("FastTrack (coded)", sorted(df["fasttrack"].dropna().unique().astype(int)), index=0)
        coop = st.selectbox("Coop (coded)", sorted(df["coop"].dropna().unique().astype(int)), index=0)
        residency = st.selectbox("Residency (coded)", sorted(df["residency"].dropna().unique().astype(int)), index=0)
        gender = st.selectbox("Gender (coded)", sorted(df["gender"].dropna().unique().astype(int)), index=0)
        previous_education = st.selectbox("Previous Education (coded)", sorted(df["previous_education"].dropna().unique().astype(int)), index=0)
        age_group = st.selectbox("Age Group (coded)", sorted(df["age_group"].dropna().unique().astype(int)), index=0)

    input_df = pd.DataFrame(
        [
            {
                "first_term_gpa": first_term_gpa,
                "second_term_gpa": second_term_gpa,
                "first_language": first_language,
                "funding": funding,
                "school": school,
                "fasttrack": fasttrack,
                "coop": coop,
                "residency": residency,
                "gender": gender,
                "previous_education": previous_education,
                "age_group": age_group,
                "high_school_avg": high_school_avg,
                "math_score": math_score,
                "english_grade": english_grade,
            }
        ]
    )
    return input_df


def main():
    st.set_page_config(page_title="Student Persistence Predictor", layout="wide")
    st.title("Student Persistence Prediction System")
    st.caption("COMP258 Group Project - Full-Stack Intelligent App Demo")

    st.markdown(
        """
        This application predicts whether a student is likely to persist in the first year.
        It uses a neural network model (**MLPClassifier**) trained on the provided student dataset.
        """
    )

    df, X_train, X_test, y_train, y_test = prepare_data()
    pipeline, metrics = train_model()

    with st.expander("Dataset preview and project summary", expanded=False):
        st.write("Dataset shape:", df.shape)
        st.dataframe(df.head(10))
        st.write("Target balance:")
        st.write(df["first_year_persistence"].value_counts())

    st.subheader("Model Evaluation")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    m2.metric("Precision", f"{metrics['precision']:.2%}")
    m3.metric("Recall", f"{metrics['recall']:.2%}")
    m4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    m5.metric("ROC-AUC", f"{metrics['roc_auc']:.2%}")

    st.write("Confusion Matrix")
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    st.dataframe(cm)

    st.text("Classification Report")
    st.code(metrics["classification_report"])

    input_df = build_input_form(df)

    if st.button("Predict Persistence", type="primary"):
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"Prediction: Persist (1) ✅")
        else:
            st.error(f"Prediction: Not Persist (0) ❌")

        st.info(f"Probability of persistence: {probability:.2%}")
        st.dataframe(input_df)

    if st.button("Save trained model"):
        model_path = save_model(pipeline)
        st.success(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
