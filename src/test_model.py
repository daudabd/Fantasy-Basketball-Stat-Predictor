# src/test_model.py
#
# Purpose:
#   Evaluate a trained model on the held-out test set and compare it
#   against a simple baseline (5-game rolling average of fantasy points).
#
# Outputs:
#   1) A JSON file with MAE/RMSE for both the model and the baseline
#   2) A CSV file with per-game predictions (for plotting / inspection)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocess import (
    PLAYER_ID_COL,
    PLAYER_FIRST_COL,
    PLAYER_LAST_COL,
    GAME_DATE_COL,
    TARGET_COL,      # "fantasy_points"
    BASELINE_COL,    # 5-game rolling average of fantasy points
)
from train_model import get_feature_cols


def eval_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression:
      - MAE: mean absolute error
      - RMSE: root mean squared error

    Both are standard for measuring prediction error of continuous targets.
    Lower values are better.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def main(args):
    # ---------------------------------------------------------------------
    # 1. Load test data
    # ---------------------------------------------------------------------
    # We parse GAME_DATE_COL as datetime so it can be used for time-based
    # analysis or plotting if needed.
    test_df = pd.read_csv(args.test_csv, parse_dates=[GAME_DATE_COL])

    # Determine which columns are used as input features.
    # This ensures test-time features match training-time features.
    feature_cols = get_feature_cols(test_df)
    print(f"[+] Test rows: {len(test_df)}, features: {len(feature_cols)}")

    # X_test: model inputs
    # y_test: ground-truth fantasy points
    # y_base: baseline prediction (5-game moving average)
    X_test = test_df[feature_cols].values
    y_test = test_df[TARGET_COL].values
    y_base = test_df[BASELINE_COL].values

    # ---------------------------------------------------------------------
    # 2. Load trained model and generate predictions
    # ---------------------------------------------------------------------
    # model_path can point to either:
    #   - models/linear_reg.pkl  (Linear Regression)
    #   - models/mlp_reg.pkl     (MLP / Neural Network)
    model = load(args.model_path)
    y_pred = model.predict(X_test)

    # ---------------------------------------------------------------------
    # 3. Compute metrics for model and baseline
    # ---------------------------------------------------------------------
    # We compare:
    #   - test_model_mae / rmse: errors from our ML model
    #   - test_baseline_mae / rmse: errors from simple 5-game average
    model_mae, model_rmse = eval_metrics(y_test, y_pred)
    base_mae, base_rmse = eval_metrics(y_test, y_base)

    metrics = {
        "test_model_mae": float(model_mae),
        "test_model_rmse": float(model_rmse),
        "test_baseline_mae": float(base_mae),
        "test_baseline_rmse": float(base_rmse),
        "n_test": int(len(test_df)),
    }

    # Save metrics as a JSON file so they can be inspected or included
    # directly in the report.
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[+] Test metrics:")
    print(json.dumps(metrics, indent=2))

    # ---------------------------------------------------------------------
    # 4. Save per-game predictions
    # ---------------------------------------------------------------------
    # This CSV is useful for:
    #   - plotting predicted vs actual
    #   - analyzing errors for specific players / dates
    preds_df = test_df[
        [PLAYER_ID_COL, PLAYER_FIRST_COL, PLAYER_LAST_COL, GAME_DATE_COL]
    ].copy()
    preds_df["y_true"] = y_test
    preds_df["y_pred"] = y_pred
    preds_df["baseline_5g"] = y_base

    preds_df.to_csv(args.preds_out, index=False)
    print(f"[+] Saved predictions to {args.preds_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set and compare to a 5-game baseline."
    )
    parser.add_argument(
        "--test-csv",
        default="processed/test.csv",
        help="Path to processed test split (with features and targets).",
    )
    parser.add_argument(
        "--model-path",
        default="models/linear_reg.pkl",
        help="Path to trained model (.pkl) to evaluate.",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/test_metrics_linear.json",
        help="Where to save JSON metrics (MAE, RMSE, baseline comparison).",
    )
    parser.add_argument(
        "--preds-out",
        default="results/test_predictions_linear.csv",
        help="Where to save CSV with per-row predictions.",
    )
    args = parser.parse_args()
    main(args)
