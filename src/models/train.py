"""Model training — Linear Regression and Ridge Regression pipelines."""
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

def get_feature_types(df, target="SalePrice"):
    X = df.drop(columns=[target], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
    return num_cols, cat_cols

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

def build_linear_pipeline(num_cols, cat_cols):
    return Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", LinearRegression()),
    ])

def build_ridge_pipeline(num_cols, cat_cols, alpha=10.0):
    return Pipeline([
        ("preprocessor", build_preprocessor(num_cols, cat_cols)),
        ("model", Ridge(alpha=alpha)),
    ])

def find_best_alpha(X_train, y_train, num_cols, cat_cols):
    alphas = [0.01,0.1,1,5,10,20,50,100,200,500,1000]
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_t = preprocessor.fit_transform(X_train)
    rcv = RidgeCV(alphas=alphas, cv=5)
    rcv.fit(X_t, y_train)
    logger.info(f"Best alpha: {rcv.alpha_}")
    return rcv.alpha_

def compute_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    adj  = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return {"MAE": round(mae,2), "RMSE": round(rmse,2),
            "R2": round(r2,4), "Adj_R2": round(adj,4)}

def cross_validate(pipeline, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2")
    return {"CV_R2_Mean": round(scores.mean(),4), "CV_R2_Std": round(scores.std(),4)}

def train_and_evaluate(df: pd.DataFrame):
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/metrics", exist_ok=True)

    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    num_cols, cat_cols = get_feature_types(
        pd.concat([X_train, y_train.rename("SalePrice")], axis=1))
    n_features = len(num_cols) + len(cat_cols)
    results = {}

    # Linear Regression
    lr = build_linear_pipeline(num_cols, cat_cols)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    m = compute_metrics(np.expm1(y_test), np.expm1(lr_pred), n_features)
    m.update(cross_validate(lr, X_train, y_train))
    results["LinearRegression"] = m
    joblib.dump(lr, "models/linear_regression.joblib")

    # Ridge Regression
    best_alpha = find_best_alpha(X_train, y_train, num_cols, cat_cols)
    ridge = build_ridge_pipeline(num_cols, cat_cols, alpha=best_alpha)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    m2 = compute_metrics(np.expm1(y_test), np.expm1(ridge_pred), n_features)
    m2.update(cross_validate(ridge, X_train, y_train))
    m2["Alpha"] = best_alpha
    results["RidgeRegression"] = m2
    joblib.dump(ridge, "models/ridge_regression.joblib")

    with open("reports/metrics/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*55)
    print("  MODEL RESULTS")
    print("="*55)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k:<15}: {v}")
    return results

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing.clean import load_data, clean_data
    from src.features.engineer import engineer_features
    df = load_data("data/raw/ames_housing.csv")
    df = clean_data(df)
    df = engineer_features(df)
    train_and_evaluate(df)
