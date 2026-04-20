#services/form_service
from services.data_service import load_dataset

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

#Metrics
def get_form_data():
    df = load_dataset()

    options = {
        col: sorted([int(x) for x in df[col].dropna().unique().tolist()])
        for col in CATEGORICAL_FEATURES
    }

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


