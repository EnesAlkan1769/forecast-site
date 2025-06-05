from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware   # ← EKLEDİK
import joblib, numpy as np
model = joblib.load("model.pkl")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inputs(BaseModel):
    x1: float; x2: float; x3: float
    x4: float; x5: float; x6: float; x7: float

@app.post("/predict")
def predict(inp: Inputs):
    X = np.array([[inp.x1, inp.x2, inp.x3,
                   inp.x4, inp.x5, inp.x6, inp.x7]])
    y_hat = model.predict(X)[0]
    return {"result": "A" if y_hat else "B"}

