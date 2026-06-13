"""Feature engineering module for Ames Housing dataset."""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

QUALITY_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
QUALITY_COLS = [
    "ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
    "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC",
]

def add_house_age(df):
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    return df

def add_remodel_age(df):
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    return df

def add_total_bathrooms(df):
    cols = ["FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"]
    if all(c in df.columns for c in cols):
        df["TotalBathrooms"] = (
            df["FullBath"] + 0.5*df["HalfBath"] +
            df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]
        )
    return df

def add_total_sf(df):
    if "TotalBsmtSF" in df.columns and "GrLivArea" in df.columns:
        df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]
    return df

def add_total_porch_sf(df):
    porch_cols = ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","WoodDeckSF"]
    available = [c for c in porch_cols if c in df.columns]
    if available:
        df["TotalPorchSF"] = df[available].sum(axis=1)
    return df

def add_quality_score(df):
    if "OverallQual" in df.columns and "OverallCond" in df.columns:
        df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
    return df

def add_qual_area(df):
    if "OverallQual" in df.columns and "GrLivArea" in df.columns:
        df["QualArea"] = df["OverallQual"] * df["GrLivArea"]
    return df

def add_binary_flags(df):
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    if "Fireplaces" in df.columns:
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    if "YearBuilt" in df.columns and "YrSold" in df.columns:
        df["IsNewHouse"] = (df["YearBuilt"] == df["YrSold"]).astype(int)
    if "YearBuilt" in df.columns and "YearRemodAdd" in df.columns:
        df["WasRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    return df

def add_total_rooms(df):
    if "TotRmsAbvGrd" in df.columns and "FullBath" in df.columns:
        df["TotalRooms"] = df["TotRmsAbvGrd"] + df["FullBath"]
    return df

def encode_ordinal_quality(df):
    for col in QUALITY_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_house_age(df)
    df = add_remodel_age(df)
    df = add_total_bathrooms(df)
    df = add_total_sf(df)
    df = add_total_porch_sf(df)
    df = add_quality_score(df)
    df = add_qual_area(df)
    df = add_binary_flags(df)
    df = add_total_rooms(df)
    df = encode_ordinal_quality(df)
    logger.info("Feature engineering complete")
    return df
