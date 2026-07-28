"""
Validation strategy module for F1 Pit Stop Prediction.
Provides StratifiedKFold and StratifiedGroupKFold splitting logic.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from typing import Generator, Tuple


def get_cv_splits(
    df: pd.DataFrame,
    target_col: str = "PitNextLap",
    group_col: str = "EventID",
    n_splits: int = 5,
    random_state: int = 42,
    use_groups: bool = True,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train/validation indices for cross-validation.
    
    Args:
        df: Input dataframe containing target and group columns
        target_col: Target variable column name ('PitNextLap')
        group_col: Column used for grouping (e.g. 'EventID' or 'Race')
        n_splits: Number of CV folds (default 5)
        random_state: Random seed
        use_groups: If True, uses StratifiedGroupKFold; otherwise StratifiedKFold
    
    Returns:
        Generator yielding (train_idx, val_idx)
    """
    y = df[target_col].values

    if use_groups and group_col in df.columns:
        groups = df[group_col].values
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return sgkf.split(df, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf.split(df, y)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_raw_data

    train, _, _ = load_raw_data()
    train["EventID"] = train["Year"].astype(str) + "_" + train["Race"]

    print("Testing StratifiedGroupKFold splits:")
    splits = list(get_cv_splits(train, n_splits=5, use_groups=True))
    for fold, (trn_idx, val_idx) in enumerate(splits):
        val_events = train.iloc[val_idx]["EventID"].nunique()
        val_pos_rate = train.iloc[val_idx]["PitNextLap"].mean()
        print(f"Fold {fold+1}: Train={len(trn_idx)}, Val={len(val_idx)}, Val Unique Events={val_events}, Target Positive Rate={val_pos_rate:.4f}")
