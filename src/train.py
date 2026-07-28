"""
End-to-end training, validation, and submission generation script.
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in python path
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
    evaluate_predictions,
    ensemble_probability_average,
)


def run_pipeline(n_splits: int = 5, save_submission: bool = True) -> Dict[str, Any]:
    """
    Run full cross-validation pipeline across candidate models, evaluate OOF performance,
    ensemble predictions, and generate test set submission file.
    """
    print("=" * 60)
    print("Starting F1 Pit Stop Prediction Training Pipeline")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/5] Loading raw data...")
    train_df, test_df, sub_df = load_raw_data()
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # 2. Feature Engineering
    print("\n[2/5] Engineering features...")
    train_feat, feature_cols = create_features(train_df)
    test_feat, _ = create_features(test_df)
    print(f"Total features created: {len(feature_cols)}")

    X = train_feat[feature_cols]
    y = train_feat["PitNextLap"].values
    X_test = test_feat[feature_cols]

    # 3. Cross-Validation Setup
    print(f"\n[3/5] Setting up {n_splits}-fold StratifiedGroupKFold validation by EventID...")
    splits = list(get_cv_splits(train_feat, target_col="PitNextLap", group_col="EventID", n_splits=n_splits))

    model_names = ["LogisticRegression", "LightGBM", "XGBoost", "CatBoost"]
    oof_predictions = {m: np.zeros(len(train_feat)) for m in model_names}
    test_predictions = {m: np.zeros(len(test_feat)) for m in model_names}

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # 4. Training Loop
    print("\n[4/5] Training models across folds...")
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        # A. Logistic Regression Baseline
        lr_model, lr_preds = train_logistic_regression(X_tr, y_tr, X_va)
        oof_predictions["LogisticRegression"][val_idx] = lr_preds
        test_predictions["LogisticRegression"] += lr_model.predict_proba(X_test)[:, 1] / n_splits
        print(f"  LogisticRegression ROC-AUC: {evaluate_predictions(y_va, lr_preds)['roc_auc']:.5f}")

        # B. LightGBM
        lgb_model, lgb_preds = train_lightgbm(X_tr, y_tr, X_va, y_va)
        oof_predictions["LightGBM"][val_idx] = lgb_preds
        test_predictions["LightGBM"] += lgb_model.predict_proba(X_test)[:, 1] / n_splits
        print(f"  LightGBM           ROC-AUC: {evaluate_predictions(y_va, lgb_preds)['roc_auc']:.5f}")

        # C. XGBoost
        xgb_model, xgb_preds = train_xgboost(X_tr, y_tr, X_va, y_va)
        oof_predictions["XGBoost"][val_idx] = xgb_preds
        test_predictions["XGBoost"] += xgb_model.predict_proba(X_test)[:, 1] / n_splits
        print(f"  XGBoost            ROC-AUC: {evaluate_predictions(y_va, xgb_preds)['roc_auc']:.5f}")

        # D. CatBoost
        cat_model, cat_preds = train_catboost(X_tr, y_tr, X_va, y_va)
        oof_predictions["CatBoost"][val_idx] = cat_preds
        test_predictions["CatBoost"] += cat_model.predict_proba(X_test)[:, 1] / n_splits
        print(f"  CatBoost           ROC-AUC: {evaluate_predictions(y_va, cat_preds)['roc_auc']:.5f}")

    # 5. Out-Of-Fold Evaluation & Ensembling
    print("\n" + "=" * 60)
    print("OUT-OF-FOLD (OOF) EVALUATION RESULTS")
    print("=" * 60)

    oof_metrics = {}
    for m in model_names:
        metrics = evaluate_predictions(y, oof_predictions[m])
        oof_metrics[m] = metrics
        print(f"{m:20s} | ROC-AUC: {metrics['roc_auc']:.5f} | LogLoss: {metrics['log_loss']:.5f} | PR-AUC: {metrics['pr_auc']:.5f}")

    # Weighted Ensemble (Equal weights for GBDT models, lower weight for Logistic Regression)
    ensemble_weights = {
        "LogisticRegression": 0.05,
        "LightGBM": 0.35,
        "XGBoost": 0.30,
        "CatBoost": 0.30,
    }
    oof_ensemble = ensemble_probability_average(oof_predictions, weights=ensemble_weights)
    ensemble_metrics = evaluate_predictions(y, oof_ensemble)
    oof_metrics["WeightedEnsemble"] = ensemble_metrics
    print(f"{'WeightedEnsemble':20s} | ROC-AUC: {ensemble_metrics['roc_auc']:.5f} | LogLoss: {ensemble_metrics['log_loss']:.5f} | PR-AUC: {ensemble_metrics['pr_auc']:.5f}")
    print("=" * 60)

    # 6. Generate Submission File
    if save_submission and sub_df is not None:
        print("\n[5/5] Generating Kaggle submission file...")
        test_ensemble = ensemble_probability_average(test_predictions, weights=ensemble_weights)

        sub_dir = PROJECT_ROOT / "submissions"
        sub_dir.mkdir(exist_ok=True)
        sub_filepath = sub_dir / "submission.csv"

        submission = pd.DataFrame({
            "id": test_df["id"],
            "PitNextLap": test_ensemble
        })
        submission.to_csv(sub_filepath, index=False)
        print(f"Submission saved to {sub_filepath}")
        print(f"Submission shape: {submission.shape}")
        print("Sample predictions:\n", submission.head())

    return {
        "oof_metrics": oof_metrics,
        "oof_predictions": oof_predictions,
        "test_predictions": test_predictions,
    }


if __name__ == "__main__":
    run_pipeline(n_splits=5, save_submission=True)
