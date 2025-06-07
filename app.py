from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Modeli yükle
model = joblib.load("model_catdv01.pkl")

app = FastAPI()

# CORS ayarı – Netlify frontend ile iletişim kurmak için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 8 inputlu yapı
class Inputs(BaseModel):
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    x6: float  # LOGIV20
    x7: float  # IV01
    x8: float  # IV09

@app.post("/predict_catdv01")
def predict_catdv01(inp: Inputs):
    # Log dönüşüm gereken alanlar (x4, x5, x6)
    log_x4 = np.log(inp.x4)
    log_x5 = np.log(inp.x5)
    log_x6 = np.log(inp.x6)

    # x7 ve x8 doğrudan alınacak
    X = np.array([[inp.x1, inp.x2, inp.x3, log_x4, log_x5, log_x6, inp.x8]])

    y_hat = model.predict(X)[0]
    return {"result": "A" if y_hat else "B"}
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, numpy as np

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

# Tahmin endpointi
@app.post("/predict_catdv01")
def predict(inp: Inputs):
    # Sadece CATDV01 için gerekli inputları modele veriyoruz
    X = np.array([[inp.x1, inp.x2, inp.x3,
                   inp.x4, inp.x5, inp.x6,
                   inp.x8]])  # inp.x7 (CATDV03) burada yok

    y_hat = model.predict(X)[0]
    return {"result": "A" if y_hat else "B"}


