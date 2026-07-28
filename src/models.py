"""
Modeling module for F1 Pit Stop Prediction.
Provides model wrappers for Logistic Regression, LightGBM, XGBoost, CatBoost, and Ensembling.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def evaluate_predictions(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for binary classification.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_pred_prob)),
        "log_loss": float(log_loss(y_true, y_pred_prob)),
        "pr_auc": float(average_precision_score(y_true, y_pred_prob)),
    }


def train_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame) -> Tuple[Any, np.ndarray]:
    """Train scaled Logistic Regression baseline model."""
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    pipeline.fit(X_train, y_train)
    val_preds = pipeline.predict_proba(X_val)[:, 1]
    return pipeline, val_preds


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any] = None
) -> Tuple[LGBMClassifier, np.ndarray]:
    """Train LightGBM model with early stopping."""
    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    model = LGBMClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[]
    )
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any] = None
) -> Tuple[XGBClassifier, np.ndarray]:
    """Train XGBoost model."""
    default_params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "early_stopping_rounds": 50,
    }
    if params:
        default_params.update(params)

    model = XGBClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Dict[str, Any] = None
) -> Tuple[CatBoostClassifier, np.ndarray]:
    """Train CatBoost model."""
    default_params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 50,
    }
    if params:
        default_params.update(params)

    model = CatBoostClassifier(**default_params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
    )
    val_preds = model.predict_proba(X_val)[:, 1]
    return model, val_preds


def ensemble_rank_average(preds_dict: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
    """
    Weighted rank averaging of prediction probabilities across multiple models.
    """
    if weights is None:
        weights = {model: 1.0 / len(preds_dict) for model in preds_dict}

    total_weight = sum(weights.values())
    normalized_weights = {m: w / total_weight for m, w in weights.items()}

    final_preds = np.zeros(len(next(iter(preds_dict.values()))))
    for model_name, preds in preds_dict.items():
        w = normalized_weights[model_name]
        # Rank transform
        ranks = pd.Series(preds).rank(pct=True).values
        final_preds += w * ranks

    return final_preds


def ensemble_probability_average(preds_dict: Dict[str, np.ndarray], weights: Dict[str, float] = None) -> np.ndarray:
    """
    Weighted probability averaging across multiple models.
    """
    if weights is None:
        weights = {model: 1.0 / len(preds_dict) for model in preds_dict}

    total_weight = sum(weights.values())
    normalized_weights = {m: w / total_weight for m, w in weights.items()}

    final_preds = np.zeros(len(next(iter(preds_dict.values()))))
    for model_name, preds in preds_dict.items():
        w = normalized_weights[model_name]
        final_preds += w * preds

    return final_preds
