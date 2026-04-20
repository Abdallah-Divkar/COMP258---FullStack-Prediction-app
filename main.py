from fastapi import FastAPI
from controllers.prediction_controller import router

app = FastAPI(title="Student Persistence API")

app.include_router(router)