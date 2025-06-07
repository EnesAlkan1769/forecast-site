from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os
print("Render içeriği:", os.listdir())

# Doğru modeli yükle
model = joblib.load("model_catdv01.pkl")

app = FastAPI()

# CORS middleware (formdan veri gönderimini destekler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kullanıcıdan gelen inputlar
class Inputs(BaseModel):
    x1: float  # IV19
    x2: float  # IV18
    x3: float  # IV15
    x4: float  # LOGIV17
    x5: float  # LOGIV14
    x6: float  # LOGIV20 (CATDV01'e özel)
    x7: float  # IV01 (CATDV03'e özel → burada kullanılmayacak)
    x8: float  # IV09

@app.post("/predict_catdv01")

def predict_catdv01(inp: Inputs):
    log_x4 = np.log(inp.x4)
    log_x5 = np.log(inp.x5)
    log_x6 = np.log(inp.x6)

    X = np.array([[inp.x1, inp.x2, inp.x3, log_x4, log_x5, log_x6, inp.x8]])
    y_hat = model.predict(X)[0]

    print("RandomForest predicted:", y_hat)

    return {"result": "A" if y_hat else "B"}

# Load CATDV03 model
model_catdv03 = joblib.load("model_catdv03.pkl")

@app.post("/predict_catdv03")
def predict_catdv03(inp: Inputs):
    log_x4 = np.log(inp.x4)
    log_x5 = np.log(inp.x5)

    # CATDV03 modeline uygun input dizisi
    X = np.array([[inp.x1, inp.x2, inp.x3, log_x4, log_x5, inp.x7, inp.x8]])

    y_hat = model_catdv03.predict(X)[0]

    print("CATDV03 predicted:", y_hat)

    return {"result": "A" if y_hat else "B"}


