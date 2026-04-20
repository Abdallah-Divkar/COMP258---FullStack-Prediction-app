#services/prediction_service
import pandas as pd
from models.pipeline import load_model

#Prediction
def make_prediction(data: dict):
    model = load_model()
    pipeline = model["pipeline"]

    df = pd.DataFrame([data])
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
    }