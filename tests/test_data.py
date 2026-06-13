import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")
from src.preprocessing.clean import (
    fill_none_features, fill_zero_features,
    remove_outliers, missing_value_report
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "GrLivArea":   [1500, 4500, 2000, 1800],
        "SalePrice":   [150000, 100000, 200000, 180000],
        "PoolQC":      [None, None, "Ex", None],
        "GarageArea":  [None, 400.0, None, 500.0],
        "Neighborhood":["CollgCr","CollgCr","OldTown","Edwards"],
        "LotFrontage": [None, 80.0, None, 70.0],
    })

def test_fill_none_features(sample_df):
    result = fill_none_features(sample_df)
    assert result["PoolQC"].isna().sum() == 0
    assert "None" in result["PoolQC"].values

def test_fill_zero_features(sample_df):
    result = fill_zero_features(sample_df)
    assert result["GarageArea"].isna().sum() == 0
    assert result["GarageArea"].iloc[0] == 0

def test_remove_outliers(sample_df):
    result = remove_outliers(sample_df)
    assert len(result) == 3

def test_missing_report(sample_df):
    report = missing_value_report(sample_df)
    assert "Missing Count" in report.columns
    assert len(report) > 0
