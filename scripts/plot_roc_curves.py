"""
Script to compute and plot ROC curves for all candidate models and the Weighted Ensemble.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_data
from src.features import create_features
from src.validation import get_cv_splits
from src.models import (
    train_logistic_regression,
    train_lightgbm,
    train_xgboost,
    train_catboost,
    ensemble_probability_average
)


def plot_roc_curves():
    print("Loading data & generating features...")
    train_df, _, _ = load_raw_data()
    train_feat, feature_cols = create_features(train_df)

    X = train_feat[feature_cols]
    y = train_feat["PitNextLap"].values

    print("Running 5-fold CV to collect Out-Of-Fold predictions...")
    splits = list(get_cv_splits(train_feat, target_col="PitNextLap", group_col="EventID", n_splits=5))

    model_names = ["LogisticRegression", "LightGBM", "XGBoost", "CatBoost"]
    oof_predictions = {m: np.zeros(len(train_feat)) for m in model_names}

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"  Processing Fold {fold+1}/5...")
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        _, lr_preds = train_logistic_regression(X_tr, y_tr, X_va)
        oof_predictions["LogisticRegression"][val_idx] = lr_preds

        _, lgb_preds = train_lightgbm(X_tr, y_tr, X_va, y_va)
        oof_predictions["LightGBM"][val_idx] = lgb_preds

        _, xgb_preds = train_xgboost(X_tr, y_tr, X_va, y_va)
        oof_predictions["XGBoost"][val_idx] = xgb_preds

        _, cat_preds = train_catboost(X_tr, y_tr, X_va, y_va)
        oof_predictions["CatBoost"][val_idx] = cat_preds

    # Ensemble predictions
    ensemble_weights = {
        "LogisticRegression": 0.05,
        "LightGBM": 0.35,
        "XGBoost": 0.30,
        "CatBoost": 0.30,
    }
    oof_predictions["Weighted Ensemble"] = ensemble_probability_average(oof_predictions, weights=ensemble_weights)

    # Plot ROC Curves
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(10, 8))

    colors = {
        "LogisticRegression": "#7f7f7f", # Gray
        "LightGBM": "#2ca02c",           # Green
        "XGBoost": "#1f77b4",            # Blue
        "CatBoost": "#ff7f0e",           # Orange
        "Weighted Ensemble": "#d62728"   # Red (highlight)
    }

    linestyles = {
        "LogisticRegression": "--",
        "LightGBM": "-",
        "XGBoost": "-",
        "CatBoost": "-",
        "Weighted Ensemble": "-"
    }

    linewidths = {
        "LogisticRegression": 1.5,
        "LightGBM": 2.0,
        "XGBoost": 2.0,
        "CatBoost": 2.0,
        "Weighted Ensemble": 2.5
    }

    for model_name, preds in oof_predictions.items():
        fpr, tpr, _ = roc_curve(y, preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.5f})",
            color=colors[model_name],
            linestyle=linestyles[model_name],
            linewidth=linewidths[model_name]
        )

    # Random chance diagonal line
    plt.plot([0, 1], [0, 1], 'k:', label="Random Chance (AUC = 0.50000)", linewidth=1.2)

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight='bold')
    plt.title("Receiver Operating Characteristic (ROC) Curves - F1 Pit Stop Prediction", fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.6)

    fig_dir = PROJECT_ROOT / "report" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_filepath = fig_dir / "roc_curves.png"
    plt.tight_layout()
    plt.savefig(out_filepath, dpi=300)
    plt.close()
    print(f"\nROC curves plot saved successfully to {out_filepath}")


if __name__ == "__main__":
    plot_roc_curves()
