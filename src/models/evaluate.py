"""Evaluation plots for model results."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from sklearn.metrics import r2_score

def plot_actual_vs_predicted(y_true, y_pred, model_name, save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#4C72B0", s=40, edgecolors="white", lw=0.5)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=2, label="Perfect")
    r2 = r2_score(y_true, y_pred)
    ax.set(xlabel="Actual Price ($)", ylabel="Predicted Price ($)",
           title=f"{model_name} — Actual vs Predicted\nR² = {r2:.4f}")
    ax.legend()
    plt.tight_layout()
    fname = f"{save_dir}/{model_name.lower().replace(' ','_')}_actual_vs_pred.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

def plot_residuals(y_true, y_pred, model_name, save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#DD8452", s=40)
    axes[0].axhline(0, color="red", linestyle="--", lw=2)
    axes[0].set(xlabel="Predicted ($)", ylabel="Residuals ($)",
                title=f"{model_name} — Residuals vs Fitted")
    sns.histplot(residuals, kde=True, ax=axes[1], color="#55A868")
    axes[1].set(xlabel="Residual ($)", title=f"{model_name} — Residual Distribution")
    plt.tight_layout()
    fname = f"{save_dir}/{model_name.lower().replace(' ','_')}_residuals.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

def plot_model_comparison(results_path="reports/metrics/results.json",
                          save_dir="reports/figures"):
    os.makedirs(save_dir, exist_ok=True)
    with open(results_path) as f:
        results = json.load(f)
    models = list(results.keys())
    metrics = ["MAE", "RMSE", "R2", "CV_R2_Mean"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#4C72B0", "#DD8452"]
    for i, metric in enumerate(metrics):
        vals = [results[m].get(metric, 0) for m in models]
        bars = axes[i].bar(models, vals, color=colors, edgecolor="white")
        axes[i].set_title(metric, fontweight="bold")
        for bar, val in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.01,
                        f"{val:,.3f}", ha="center", va="bottom", fontsize=9)
        axes[i].set_xticklabels([m.replace("Regression","\nRegression") for m in models])
    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"{save_dir}/model_comparison.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
