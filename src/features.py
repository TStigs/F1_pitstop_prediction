"""
Feature engineering module for F1 Pit Stop Prediction.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


COMPOUND_MAP = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5
}


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate domain-specific features for F1 pit stop prediction.
    
    Args:
        df: Input raw or cleaned dataframe.
        
    Returns:
        df_feat: Dataframe with added engineered features.
        feature_cols: List of feature column names to be used for modeling.
    """
    res = df.copy()

    # 1. Identifier / Grouping column
    res["EventID"] = res["Year"].astype(str) + "_" + res["Race"]

    # 2. Compound Encoding
    res["Compound_Ordinal"] = res["Compound"].map(COMPOUND_MAP).fillna(0).astype(int)

    # 3. Tire & Stint Dynamics
    res["TyreLife_Squared"] = res["TyreLife"] ** 2
    res["Degradation_Per_Lap"] = res["Cumulative_Degradation"] / (res["TyreLife"] + 1e-5)
    res["TyreLife_Degradation_Interaction"] = res["TyreLife"] * res["Cumulative_Degradation"]
    res["Compound_TyreLife_Interaction"] = res["Compound_Ordinal"] * res["TyreLife"]

    # 4. Race Progress & Pit Window Context
    res["RaceRemaining"] = 1.0 - res["RaceProgress"]
    res["RaceProgress_Squared"] = res["RaceProgress"] ** 2
    res["Estimated_Total_Laps"] = res["LapNumber"] / (res["RaceProgress"] + 1e-5)
    res["Estimated_Laps_Remaining"] = res["Estimated_Total_Laps"] - res["LapNumber"]

    # Pit Window Indicator (Pit stops usually peak between 30% and 85% race progress)
    res["In_Primary_Pit_Window"] = ((res["RaceProgress"] >= 0.25) & (res["RaceProgress"] <= 0.85)).astype(int)
    res["Late_Race_Flag"] = (res["RaceProgress"] > 0.85).astype(int)

    # 5. Position & Delta Features
    res["Position_Squared"] = res["Position"] ** 2
    res["Is_Podium_Position"] = (res["Position"] <= 3).astype(int)
    res["Is_Points_Position"] = (res["Position"] <= 10).astype(int)
    res["LapTime_Pace_Ratio"] = res["LapTime_Delta"] / (res["LapTime (s)"] + 1e-5)

    # List of all numeric feature columns for model input
    feature_cols = [
        "Compound_Ordinal",
        "Year",
        "PitStop",
        "LapNumber",
        "Stint",
        "TyreLife",
        "TyreLife_Squared",
        "Degradation_Per_Lap",
        "TyreLife_Degradation_Interaction",
        "Compound_TyreLife_Interaction",
        "Position",
        "Position_Squared",
        "Position_Change",
        "Is_Podium_Position",
        "Is_Points_Position",
        "LapTime (s)",
        "LapTime_Delta",
        "LapTime_Pace_Ratio",
        "Cumulative_Degradation",
        "RaceProgress",
        "RaceProgress_Squared",
        "RaceRemaining",
        "Estimated_Total_Laps",
        "Estimated_Laps_Remaining",
        "In_Primary_Pit_Window",
        "Late_Race_Flag",
    ]

    return res, feature_cols


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.data import load_raw_data

    train, test, _ = load_raw_data()
    train_feat, fcols = create_features(train)
    test_feat, _ = create_features(test)

    print(f"Engineered features shape - Train: {train_feat.shape}, Test: {test_feat.shape}")
    print(f"Number of modeling features: {len(fcols)}")
    print("Features:", fcols)
