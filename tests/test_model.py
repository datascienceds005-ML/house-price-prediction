import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.models.train import (
    build_linear_pipeline, build_ridge_pipeline,
    compute_metrics, get_feature_types
)

@pytest.fixture
def simple_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "GrLivArea":    np.random.randint(800, 3000, n).astype(float),
        "OverallQual":  np.random.randint(1, 10, n).astype(float),
        "TotalBsmtSF":  np.random.randint(0, 2000, n).astype(float),
        "Neighborhood": np.random.choice(["A","B","C"], n),
        "SalePrice":    np.random.randint(100000, 500000, n).astype(float),
    })

def test_feature_types(simple_df):
    num, cat = get_feature_types(simple_df)
    assert "GrLivArea" in num
    assert "Neighborhood" in cat
    assert "SalePrice" not in num

def test_linear_pipeline(simple_df):
    num, cat = get_feature_types(simple_df)
    pipe = build_linear_pipeline(num, cat)
    X = simple_df.drop("SalePrice", axis=1)
    y = np.log1p(simple_df["SalePrice"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()

def test_ridge_pipeline(simple_df):
    num, cat = get_feature_types(simple_df)
    pipe = build_ridge_pipeline(num, cat, alpha=10.0)
    X = simple_df.drop("SalePrice", axis=1)
    y = np.log1p(simple_df["SalePrice"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)

def test_compute_metrics():
    y_true = np.array([100000, 200000, 150000, 300000], dtype=float)
    y_pred = np.array([110000, 195000, 145000, 305000], dtype=float)
    m = compute_metrics(y_true, y_pred, n_features=10)
    assert 0 <= m["R2"] <= 1
    assert m["MAE"] > 0
    assert m["RMSE"] >= m["MAE"]
