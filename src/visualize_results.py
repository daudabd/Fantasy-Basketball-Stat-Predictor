# src/visualize_results.py
#
# Purpose:
#   Generate visualizations to help evaluate model performance on the test set.
#   This script reads:
#     - test predictions (CSV with y_true, y_pred, baseline)
#     - test metrics (JSON with MAE/RMSE)
#
#   It produces:
#     1. A scatter plot of predicted vs actual fantasy points
#     2. A histogram comparing model error vs baseline error
#
# Outputs are saved to a specified folder (e.g., plots_linear/ or plots_mlp/).

import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from preprocess import GAME_DATE_COL


# ---------------------------------------------------------------------
# 1. Predicted vs Actual Scatter Plot
# ---------------------------------------------------------------------
def plot_preds_vs_actual(preds_df, out_path):
    """
    Create a scatter plot comparing:
      - Model predictions (y_pred)
      - Baseline predictions (baseline_5g)
      - Ground truth values (y_true)

    This plot visually shows how close predictions are to actual fantasy
    points. The diagonal dashed line represents perfect accuracy.
    """
    plt.figure(figsize=(7, 7))

    # Model vs actual
    plt.scatter(
        preds_df["y_true"],
        preds_df["y_pred"],
        alpha=0.4,
        label="Model Prediction"
    )

    # Baseline vs actual
    plt.scatter(
        preds_df["y_true"],
        preds_df["baseline_5g"],
        alpha=0.4,
        label="Baseline (5-game avg)"
    )

    # Perfect prediction reference line
    max_val = max(preds_df["y_true"].max(), preds_df["y_pred"].max())
    plt.plot([0, max_val], [0, max_val], "k--", label="Perfect prediction")

    plt.xlabel("Actual Fantasy Points")
    plt.ylabel("Predicted Fantasy Points")
    plt.title("Predicted vs Actual Fantasy Points")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# 2. Error Histogram (Model vs Baseline)
# ---------------------------------------------------------------------
def plot_error_hist(preds_df, out_path):
    """
    Create a histogram showing the distribution of prediction errors:
        error = prediction - actual

    We plot two histograms:
      - Model error
      - Baseline (5-game average) error

    This visualization tells us:
      • Does our model make smaller errors on average?
      • Are large errors more or less frequent than the baseline?
    """
    model_err = preds_df["y_pred"] - preds_df["y_true"]
    baseline_err = preds_df["baseline_5g"] - preds_df["y_true"]

    plt.figure(figsize=(10, 5))

    # Model error distribution
    plt.hist(
        model_err,
        bins=50,
        alpha=0.6,
        label="Model",
        color="blue"
    )

    # Baseline error distribution
    plt.hist(
        baseline_err,
        bins=50,
        alpha=0.6,
        label="Baseline",
        color="orange"
    )

    plt.xlabel("Error (Prediction - Actual)")
    plt.ylabel("Frequency")
    plt.title("Error Distribution: Model vs Baseline")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------
# 3. Main Script: Load Inputs, Print Metrics, Save Plots
# ---------------------------------------------------------------------
def main(args):
    # Ensure output directory exists
    out_dir = Path(args.plots_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the test prediction CSV (contains y_true, y_pred, baseline_5g)
    preds_df = pd.read_csv(args.preds_csv, parse_dates=[GAME_DATE_COL])

    # Load test metrics (MAE/RMSE)
    with open(args.test_metrics, "r") as f:
        metrics = json.load(f)

    print("[+] Test metrics:")
    print(json.dumps(metrics, indent=2))

    # Generate all plots
    plot_preds_vs_actual(preds_df, out_dir / "preds_vs_actual.png")
    plot_error_hist(preds_df, out_dir / "error_hist.png")

    print(f"[+] Saved plots to folder: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate plots comparing model predictions vs actuals on test set."
    )
    parser.add_argument(
        "--preds-csv",
        default="results/test_predictions_linear.csv",
        help="CSV file containing y_true, y_pred, and baseline_5g columns."
    )
    parser.add_argument(
        "--test-metrics",
        default="results/test_metrics_linear.json",
        help="JSON file containing test MAE/RMSE."
    )
    parser.add_argument(
        "--plots-out",
        default="plots/",
        help="Directory where plots will be saved."
    )
    args = parser.parse_args()
    main(args)
