"""Load saved model and predict house prices."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

def load_model(model_name="ridge"):
    paths = {
        "linear": "models/linear_regression.joblib",
        "ridge":  "models/ridge_regression.joblib",
    }
    path = paths.get(model_name, "models/ridge_regression.joblib")
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")
    return joblib.load(path)

def predict_price(features: dict, model_name="ridge") -> dict:
    model = load_model(model_name)
    df = pd.DataFrame([features])
    log_pred = model.predict(df)[0]
    price = np.expm1(log_pred)
    return {
        "predicted_price": round(float(price), 2),
        "formatted": f"${price:,.0f}",
        "model": model_name,
    }

if __name__ == "__main__":
    sample = {
        "GrLivArea": 1800, "OverallQual": 7,
        "TotalBsmtSF": 800, "GarageCars": 2,
        "YearBuilt": 2005, "Neighborhood": "CollgCr",
    }
    result = predict_price(sample)
    print(f"Predicted: {result['formatted']}")
