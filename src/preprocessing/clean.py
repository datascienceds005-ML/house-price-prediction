"""
Data cleaning module for Ames Housing dataset.
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NONE_FEATURES = [
    "PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
    "GarageType","GarageFinish","GarageQual","GarageCond",
    "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
    "MasVnrType",
]
ZERO_FEATURES = [
    "GarageYrBlt","GarageArea","GarageCars",
    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
    "BsmtFullBath","BsmtHalfBath","MasVnrArea",
]
MODE_FEATURES = [
    "MSZoning","Electrical","KitchenQual",
    "Exterior1st","Exterior2nd","Functional","SaleType","Utilities",
]

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df

def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)
    return df

def fill_none_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in NONE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("None")
    return df

def fill_zero_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in ZERO_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

def fill_mode_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in MODE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def fill_lot_frontage(df: pd.DataFrame) -> pd.DataFrame:
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )
    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    if "GrLivArea" in df.columns and "SalePrice" in df.columns:
        before = len(df)
        df = df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 200000))]
        logger.info(f"Removed {before - len(df)} outliers")
    return df

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["YearBuilt","YearRemodAdd","GarageYrBlt","YrSold"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    return pd.DataFrame({
        "Missing Count": missing,
        "Missing %": pct.round(2)
    }).query("`Missing Count` > 0").sort_values("Missing %", ascending=False)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_id_column(df)
    df = fill_none_features(df)
    df = fill_zero_features(df)
    df = fill_mode_features(df)
    df = fill_lot_frontage(df)
    df = remove_outliers(df)
    df = fix_data_types(df)
    logger.info(f"Cleaning done. Remaining nulls: {df.isnull().sum().sum()}")
    return df
