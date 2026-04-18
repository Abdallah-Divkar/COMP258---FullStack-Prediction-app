import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from model_utils import (
    load_or_train_model,
    make_prediction,
    get_form_options,
    get_dashboard_data,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Student Persistence Dashboard API")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

MODEL_PAYLOAD = load_or_train_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return MODEL_PAYLOAD["metrics"]


@app.get("/dashboard-data")
def dashboard_data():
    return get_dashboard_data()


@app.post("/predict")
def predict_api(payload: dict):
    result = make_prediction(payload)
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    ui_data = get_form_options()
    dashboard = get_dashboard_data()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "options": ui_data["options"],
            "defaults": ui_data["defaults"],
            "metrics": MODEL_PAYLOAD["metrics"],
            "dashboard": dashboard,
            "result": None,
            "active_tab": "predict",
        },
    )


@app.post("/", response_class=HTMLResponse)
def predict_from_form(
    request: Request,
    student_id: str = Form("STU-DEMO"),
    first_term_gpa: float = Form(...),
    second_term_gpa: float = Form(...),
    high_school_avg: float = Form(...),
    math_score: float = Form(...),
    english_grade: float = Form(...),
    first_language: int = Form(...),
    funding: int = Form(...),
    school: int = Form(...),
    fasttrack: int = Form(...),
    coop: int = Form(...),
    residency: int = Form(...),
    gender: int = Form(...),
    previous_education: int = Form(...),
    age_group: int = Form(...),
):
    form_data = {
        "first_term_gpa": first_term_gpa,
        "second_term_gpa": second_term_gpa,
        "high_school_avg": high_school_avg,
        "math_score": math_score,
        "english_grade": english_grade,
        "first_language": first_language,
        "funding": funding,
        "school": school,
        "fasttrack": fasttrack,
        "coop": coop,
        "residency": residency,
        "gender": gender,
        "previous_education": previous_education,
        "age_group": age_group,
    }

    result = make_prediction(form_data)
    ui_data = get_form_options()
    dashboard = get_dashboard_data()

    result["student_id"] = student_id

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "options": ui_data["options"],
            "defaults": {
                "student_id": student_id,
                **form_data
            },
            "metrics": MODEL_PAYLOAD["metrics"],
            "dashboard": dashboard,
            "result": result,
            "active_tab": "predict",
        },
    )