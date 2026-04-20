#services/dashboard_service
import pandas as pd
from models.pipeline import load_model
from services.data_service import load_dataset

def get_dashboard():
    model = load_model()
    pipeline = model["pipeline"]
    metrics = model["metrics"]

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

'''def get_dashboard():
    model = load_model()
    pipeline = model["pipeline"]
    metrics = model["metrics"]

    df = load_dataset()

    X = df.drop("first_year_persistence", axis=1)
    probabilities = pipeline.predict_proba(X)[:, 1]

    return {
        "metrics": metrics,
        "average_risk": float(probabilities.mean())
    }'''


