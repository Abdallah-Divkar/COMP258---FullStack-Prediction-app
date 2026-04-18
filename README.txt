FastAPI Student Persistence Predictor

Files
- api_main.py -> backend API + web routes
- model_utils.py -> neural network training and prediction logic
- templates/index.html -> frontend
- requirements.txt -> install dependencies
- Student data.csv -> dataset

Install
pip install -r requirements.txt

Run
uvicorn api_main:app --reload

Open in browser
http://127.0.0.1:8000

Endpoints
- GET /health
- GET /metrics
- POST /predict

Notes
- Backend: FastAPI
- Frontend: HTML + Jinja2
- Neural network: sklearn MLPClassifier