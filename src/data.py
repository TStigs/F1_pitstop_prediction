"""
Data loading and validation module for F1 Pit Stop Prediction project.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, Any


def load_raw_data(data_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw train, test, and sample_submission CSV files from data_dir.
    
    Returns:
        train (pd.DataFrame): Training dataset including target 'PitNextLap'
        test (pd.DataFrame): Test dataset for predictions
        sub (pd.DataFrame): Sample submission file format
    """
    path = Path(data_dir)
    if not path.exists():
        # Fallback to project root / data/raw if called from notebooks/ or scripts/
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "data" / "raw"

    train_path = path / "train.csv"
    test_path = path / "test.csv"
    sub_path = path / "sample_submission.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sub = pd.read_csv(sub_path) if sub_path.exists() else None

    return train, test, sub


def get_data_summary(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Generate basic data validation metrics for a dataframe.
    """
    summary = {
        "name": name,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "missing_count": int(df.isnull().sum().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    if "PitNextLap" in df.columns:
        summary["target_distribution"] = df["PitNextLap"].value_counts(normalize=True).to_dict()

    return summary


if __name__ == "__main__":
    train, test, sub = load_raw_data()
    print("Train summary:", get_data_summary(train, "Train"))
    print("Test summary:", get_data_summary(test, "Test"))
