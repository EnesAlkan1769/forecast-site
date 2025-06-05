from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import joblib

# 7 girişli yapay veri üret
X, y = make_classification(
        n_samples=300,
        n_features=7,        #  <<< 7 giriş
        n_informative=5,
        n_redundant=0,
        random_state=42)

model = LogisticRegression(max_iter=500).fit(X, y)
joblib.dump(model, "model.pkl")
print("model.pkl (7-feature) kaydedildi 🚀")
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, (y > 1).astype(int))
joblib.dump(model, "model.pkl")
print("model.pkl kaydedildi 🚀")


