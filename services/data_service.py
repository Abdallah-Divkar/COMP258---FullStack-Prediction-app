#services/data_service
import os
import pandas as pd
import numpy as np

DATA_FILE = "Student data.csv"

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


def get_base_path() -> str:
    return os.path.dirname(os.path.abspath(__file__))

#Data
def load_dataset() -> pd.DataFrame:
    full_path = os.path.join(get_base_path(), "..", DATA_FILE)
    df = pd.read_csv(full_path, sep=",", skiprows=24, header=None)
    df.columns = COLUMN_NAMES

    df = df.replace("?", np.nan)

    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["first_year_persistence"]).copy()
    df["first_year_persistence"] = df["first_year_persistence"].astype(int)

    return df