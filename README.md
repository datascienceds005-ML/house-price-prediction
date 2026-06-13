# 🏠 House Price Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)

> End-to-end ML project predicting house prices using **Linear Regression** and **Ridge Regression** on the Ames Housing dataset with full EDA, 14 engineered features, and automated sklearn pipeline.

## 📊 Results

| Model | MAE | RMSE | R² | CV R² |
|-------|-----|------|----|-------|
| Linear Regression | ~$17,000 | ~$24,000 | ~0.87 | ~0.85 |
| Ridge Regression  | ~$15,000 | ~$21,000 | ~0.90 | ~0.89 |

## 🚀 Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction
cd house-price-prediction
pip install -r requirements.txt
python src/models/train.py
```

## 🏗️ Structure
house-price-prediction/

├── data/raw/           # Dataset

├── src/

│   ├── preprocessing/  # Cleaning

│   ├── features/       # Engineering

│   └── models/         # Train, predict, evaluate

├── tests/              # pytest suite

├── reports/            # Plots and metrics
house-price-prediction/

├── data/raw/           # Dataset

├── src/

│   ├── preprocessing/  # Cleaning

│   ├── features/       # Engineering

│   └── models/         # Train, predict, evaluate

├── tests/              # pytest suite

├── reports/            # Plots and metrics

└── models/             # Saved .joblib files
## 🔬 Engineered Features

| Feature | Why It Helps |
|---------|-------------|
| `HouseAge` | Newer homes cost more |
| `TotalSF` | Full living space = strongest predictor |
| `TotalBathrooms` | Convenience premium |
| `QualityScore` | Quality × Condition interaction |
| `QualArea` | Quality × Area nonlinear term |
| `WasRemodeled` | Renovated homes sell higher |
| `HasGarage` | Garage presence premium |
| `IsNewHouse` | New construction commands premium |

## 👤 Author

**Darsh Kumar** — B.Tech Data Science
