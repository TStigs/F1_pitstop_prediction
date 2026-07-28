# 🏎️ F1 Pit Stop Prediction: Machine Learning Workflow & Strategy Analysis

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-green)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-blue)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)](https://catboost.ai/)
[![Quarto](https://img.shields.io/badge/Quarto-Report-75AADB?style=flat&logo=quarto&logoColor=white)](https://quarto.org/)

A complete, end-to-end data science portfolio project based on the Kaggle competition **Playground Series S6E5: Predicting F1 Pit Stops**.

This repository demonstrates data auditing, domain-specific feature engineering, leak-free cross-validation, multi-GBDT model ensembling, SHAP explainability, and reproducible Quarto technical reporting.

---

## 📌 Project Overview

Predicting when a Formula 1 driver will pit on the next lap (`PitNextLap`) is crucial for live telemetry analysis, race strategy modeling, and undercut/overcut detection.

- **Training Dataset**: 439,140 rows x 16 columns across 104 Grand Prix events (2022–2025).
- **Test Dataset**: 188,165 rows x 15 columns.
- **Target Class**: `PitNextLap` (~19.90% positive class).
- **Primary Metric**: **ROC-AUC** & **LogLoss**.

---

## 📊 Key Results Benchmark

| Model | ROC-AUC | LogLoss | PR-AUC |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.82695 | 0.38864 | 0.50320 |
| **CatBoost Classifier** | 0.93044 | 0.26095 | 0.73403 |
| **LightGBM Classifier** | 0.93154 | 0.25897 | 0.73527 |
| **XGBoost Classifier** | **0.93238** | **0.25753** | **0.73909** |
| **Weighted Multi-GBDT Ensemble** | **0.93193** | **0.25909** | **0.73857** |

---

## 🛠️ Repository Architecture

```text
f1-pit-stop-prediction/
│
├── data/
│   ├── raw/                       # Raw Kaggle CSV datasets (train.csv, test.csv)
│   └── processed/                 # Processed feature caches
│
├── notebooks/
│   ├── 01_data_audit.ipynb        # Data audit & missing value inspection
│   ├── 02_exploratory_analysis.ipynb # Feature interactions & stint distributions
│   ├── 03_baseline_models.ipynb   # Logistic Regression baseline benchmark
│   ├── 04_feature_engineering.ipynb # GBDT cross-validation experiments
│   └── 05_model_interpretation.ipynb # SHAP explainability & PDP plots
│
├── src/
│   ├── data.py                    # Data loading & summary tools
│   ├── features.py                # F1 stint, race progress & pace features
│   ├── validation.py              # StratifiedGroupKFold event splitting
│   ├── models.py                  # Model training wrappers & ensemble functions
│   └── train.py                   # Full end-to-end retraining & submission pipeline
│
├── report/
│   ├── report.qmd                 # Quarto technical report
│   └── figures/                   # Rendered plots (SHAP, feature importances)
│
├── submissions/
│   └── submission.csv             # Kaggle test set predictions (188,165 rows)
│
├── .vscode/                       # VS Code workspace settings & recommendations
├── requirements.txt               # Dependencies list
└── README.md                      # Public project documentation
```

---

## ⚙️ Quickstart & Reproduction

### 1. Clone & Setup Environment

```bash
git clone https://github.com/your-username/f1-pit-stop-prediction.git
cd f1-pit-stop-prediction

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Kaggle Dataset

Place `train.csv`, `test.csv`, and `sample_submission.csv` inside `data/raw/`, or fetch via Kaggle CLI:

```bash
kaggle competitions download -c playground-series-s6e5 -p data/raw/
```

### 3. Run Training & Submission Pipeline

```bash
python3 src/train.py
```

This script will:
1. Load raw data from `data/raw/`
2. Generate 26 domain features
3. Execute 5-fold `StratifiedGroupKFold` cross-validation grouped by `EventID`
4. Print Out-Of-Fold evaluation metrics
5. Export `submissions/submission.csv`

---

## 📈 Visualizations & SHAP Interpretation

![SHAP Summary Plot](report/figures/shap_summary.png)
*Figure 1: SHAP feature impact on pit stop probability (TyreLife & Degradation are top drivers).*

![Feature Importance](report/figures/feature_importance.png)
*Figure 2: LightGBM top 15 feature importances.*

---

## 📄 Technical Report

Checkout the github-page

Or

Render the full Quarto technical report:

```bash
quarto render report/report.qmd
```
