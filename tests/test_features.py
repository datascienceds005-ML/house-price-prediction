import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")
from src.features.engineer import (
    add_house_age, add_total_bathrooms,
    add_total_sf, add_quality_score,
    add_binary_flags, engineer_features
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "YearBuilt":    [2000, 1990, 2010],
        "YrSold":       [2010, 2008, 2010],
        "YearRemodAdd": [2005, 1990, 2010],
        "FullBath":     [2, 1, 3],
        "HalfBath":     [1, 0, 0],
        "BsmtFullBath": [1, 0, 1],
        "BsmtHalfBath": [0, 1, 0],
        "TotalBsmtSF":  [800.0, 0.0, 1200.0],
        "GrLivArea":    [1500, 1200, 2000],
        "OverallQual":  [7, 5, 8],
        "OverallCond":  [5, 6, 7],
        "GarageArea":   [400.0, 0.0, 600.0],
        "PoolArea":     [0, 0, 200],
        "Fireplaces":   [1, 0, 2],
        "TotRmsAbvGrd": [7, 5, 9],
    })

def test_house_age(sample_df):
    result = add_house_age(sample_df)
    assert "HouseAge" in result.columns
    assert result["HouseAge"].iloc[0] == 10
    assert result["HouseAge"].min() >= 0

def test_total_bathrooms(sample_df):
    result = add_total_bathrooms(sample_df)
    assert "TotalBathrooms" in result.columns
    assert result["TotalBathrooms"].iloc[0] == pytest.approx(3.5)

def test_total_sf(sample_df):
    result = add_total_sf(sample_df)
    assert "TotalSF" in result.columns
    assert result["TotalSF"].iloc[0] == 2300

def test_quality_score(sample_df):
    result = add_quality_score(sample_df)
    assert "QualityScore" in result.columns
    assert result["QualityScore"].iloc[0] == 35

def test_binary_flags(sample_df):
    result = add_binary_flags(sample_df)
    assert result["HasGarage"].iloc[0] == 1
    assert result["HasGarage"].iloc[1] == 0
    assert result["HasPool"].iloc[2] == 1

def test_engineer_features_complete(sample_df):
    result = engineer_features(sample_df)
    for col in ["HouseAge","TotalBathrooms","TotalSF","QualityScore","HasGarage"]:
        assert col in result.columns
