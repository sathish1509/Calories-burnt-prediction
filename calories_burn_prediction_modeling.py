"""
Calories Burnt Prediction — Model Training & Evaluation (Python Script)

Features:
- Train/test split
- Models: Linear Regression (with scaling), Random Forest, XGBoost (optional)
- Metrics: MAE, RMSE, R^2
- Clear comparison printout
- Feature importance plots for tree-based models (and saved PNGs)
- Save best model to .pkl for deployment

Usage (from terminal):
    python calories_burn_prediction_modeling.py \
        --data-path "d:/AIML project/calories_burn_prediction_dataset.csv" \
        --model-output-path "d:/AIML project/best_model.pkl" \
        --plots-output-dir "d:/AIML project/plots" \
        --test-size 0.2

Notes:
- Assumes preprocessing is complete and the dataset is numeric-ready.
- Target column is auto-detected among common names or defaults to the last numeric column.
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 6)

try:
    from xgboost import XGBRegressor  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

import joblib


# -------------------------- Utility Functions -------------------------- #

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load preprocessed dataset from CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Loaded DataFrame is empty. Check the CSV content.")
    return df


def detect_target_column(df: pd.DataFrame, candidates: List[str] | None = None) -> str:
    """Detect likely target column for calories burnt.

    Tries common names; if not found, uses the last numeric column.
    """
    if candidates is None:
        candidates = [
            "Calories_Burned",
            "Calories_Burnt",
            "CaloriesBurned",
            "CaloriesBurnt",
            "Calories",
        ]

    df_cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in df_cols_lower:
            return df_cols_lower[c.lower()]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to use as a target.")
    return numeric_cols[-1]


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column {target_col} not in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_models(random_state: int = 42) -> Dict[str, object]:
    """Return a dictionary of model name -> estimator.

    Linear Regression is wrapped with StandardScaler via Pipeline.
    """
    models: Dict[str, object] = {}

    models["LinearRegression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("reg", LinearRegression()),
        ]
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=400, max_depth=None, random_state=random_state, n_jobs=-1
    )

    if XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )
    return models


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_evaluate(
    models: Dict[str, object], X_train, y_train, X_test, y_test
) -> Tuple[Dict[str, object], pd.DataFrame]:
    results = []
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        results.append({"Model": name, **metrics})
        trained_models[name] = model
        print(
            f"{name} -> MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | R2: {metrics['R2']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(by="RMSE").reset_index(drop=True)
    return trained_models, results_df


def _extract_estimator(model: object) -> object:
    """If a Pipeline, return the last step estimator, else the model itself."""
    if isinstance(model, Pipeline):
        try:
            return model.steps[-1][1]
        except Exception:
            return model
    return model


def plot_feature_importance(
    model: object, feature_names: List[str], title: str, top_n: int = 15, save_path: str | None = None
) -> None:
    """Plot feature importance for tree-based models (RandomForest, XGBoost)."""
    estimator = _extract_estimator(model)

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(getattr(estimator, "feature_importances_"))
    elif hasattr(estimator, "get_booster"):
        try:
            booster = estimator.get_booster()
            score = booster.get_score(importance_type="gain")
            importances = np.zeros(len(feature_names))
            for k, v in score.items():
                idx = int(k[1:])  # 'f0' -> 0
                if idx < len(importances):
                    importances[idx] = v
        except Exception:
            importances = None

    if importances is None:
        print("Feature importances not available for this model.")
        return

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values(by="importance", ascending=False).head(min(top_n, len(imp_df)))

    plt.figure(figsize=(9, max(4, int(min(top_n, len(imp_df)) * 0.35))))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved feature importance plot to: {save_path}")

    # Show interactive window if environment supports it
    try:
        plt.show()
    except Exception:
        pass


def save_best_model(trained_models: Dict[str, object], results_df: pd.DataFrame, output_path: str) -> Tuple[str, object]:
    """Save the best model based on lowest RMSE to a pickle file."""
    if results_df.empty:
        raise ValueError("Results DataFrame is empty. No models to select.")
    best_row = results_df.sort_values("RMSE").iloc[0]
    best_name = str(best_row["Model"])
    best_model = trained_models[best_name]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f"Saved best model '{best_name}' to: {output_path}")
    return best_name, best_model


# ------------------------------ Main Flow ------------------------------ #

def parse_args() -> argparse.Namespace:
    default_data = r"d:\\AIML project\\calories_burn_prediction_dataset.csv"
    default_model = r"d:\\AIML project\\best_model.pkl"
    default_plots = r"d:\\AIML project\\plots"

    parser = argparse.ArgumentParser(
        description="Calories Burnt Prediction — Model Training & Evaluation"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=default_data,
        help=f"Path to preprocessed CSV (default: {default_data})",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        default=default_model,
        help=f"Where to save best model .pkl (default: {default_model})",
    )
    parser.add_argument(
        "--plots-output-dir",
        type=str,
        default=default_plots,
        help=f"Directory to save plots (default: {default_plots})",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="",
        help="Optional explicit target column name; if empty, auto-detect",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for train/test split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Configuration:")
    print(f"  Data path          : {args.data_path}")
    print(f"  Model output path  : {args.model_output_path}")
    print(f"  Plots output dir   : {args.plots_output_dir}")
    print(f"  Test size          : {args.test_size}")
    print(f"  Random state       : {args.random_state}")

    # Load data
    df = load_dataset(args.data_path)
    print("Data shape:", df.shape)

    # Detect and set up target
    target_col = args.target_col if args.target_col else detect_target_column(df)
    print("Detected target column:", target_col)

    # Features/target and split
    X, y = split_features_target(df, target_col)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print("Train/Test shapes:", X_train.shape, X_test.shape)

    # Build, train, evaluate
    models = build_models(random_state=args.random_state)
    if not XGB_AVAILABLE:
        print("XGBoost unavailable — proceeding without it. To enable, install 'xgboost'.")

    trained_models, results_df = train_and_evaluate(
        models, X_train, y_train, X_test, y_test
    )

    print("\nModel Comparison (sorted by RMSE):")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(results_df.to_string(index=False))

    # Plot feature importances
    os.makedirs(args.plots_output_dir, exist_ok=True)

    if "RandomForest" in trained_models:
        rf_plot_path = os.path.join(args.plots_output_dir, "feature_importance_random_forest.png")
        plot_feature_importance(
            trained_models["RandomForest"],
            feature_names,
            title="Random Forest — Top Feature Importances",
            top_n=min(15, len(feature_names)),
            save_path=rf_plot_path,
        )

    if "XGBoost" in trained_models:
        xgb_plot_path = os.path.join(args.plots_output_dir, "feature_importance_xgboost.png")
        plot_feature_importance(
            trained_models["XGBoost"],
            feature_names,
            title="XGBoost — Top Feature Importances",
            top_n=min(15, len(feature_names)),
            save_path=xgb_plot_path,
        )

    # Save best model
    best_name, _ = save_best_model(trained_models, results_df, args.model_output_path)
    print("Best model:", best_name)


if __name__ == "__main__":
    main()
