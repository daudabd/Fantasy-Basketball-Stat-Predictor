# src/train_model.py
#
# Purpose:
#   Train regression models to predict fantasy basketball points using
#   the processed train/validation splits produced by preprocess.py.
#
#   Models supported:
#     - Linear Regression (baseline ML model)
#     - MLPRegressor (simple neural network to capture nonlinear patterns)
#
# Outputs:
#   - Trained model saved as a .pkl file in models/
#   - Training/validation metrics saved as JSON in results/
#
# These files are later used by:
#   - test_model.py (for final evaluation on the test set)
#   - visualize_results.py (for plotting predictions vs actuals)

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import (
    PLAYER_ID_COL,
    PLAYER_FIRST_COL,
    PLAYER_LAST_COL,
    GAME_DATE_COL,
    TEAM_COL,
    OPP_COL,
    MIN_COL,
    PTS_COL,
    REB_COL,
    AST_COL,
    STL_COL,
    BLK_COL,
    TOV_COL,
    FGA_COL,
    FGM_COL,
    FG3M_COL,
    FTA_COL,
    FTM_COL,
    TARGET_COL,
    BASELINE_COL,
)


def get_feature_cols(df: pd.DataFrame):
    """
    Determine which columns should be used as input features (X)
    for the model.

    We:
      - DROP:
          * ID-like columns (player name, gameId, date, etc.)
          * raw stats from the current game (we don't want to "cheat")
          * target + baseline columns
      - KEEP:
          * engineered features such as:
              - rolling averages (points_roll3, etc.)
              - days_rest
              - is_home
              - team_*/opp_* one-hot columns

    This ensures we only use information that would be known
    **before** the game is played.
    """
    # ID / meta columns that we do NOT want to feed to the model
    id_cols = [
        PLAYER_ID_COL,
        PLAYER_FIRST_COL,
        PLAYER_LAST_COL,
        "gameId",
        GAME_DATE_COL,
        "playerteamCity",
        "opponentteamCity",
        "gameType",
        "gameLabel",
        "gameSubLabel",
        "seriesGameNumber",
        "win",
        "home",
    ]

    # Raw box score stats from the game itself (leakage if included)
    raw_stat_cols = [
        MIN_COL,
        PTS_COL,
        REB_COL,
        AST_COL,
        STL_COL,
        BLK_COL,
        TOV_COL,
        FGA_COL,
        FGM_COL,
        FG3M_COL,
        FTA_COL,
        FTM_COL,
    ]

    drop_cols = set(
        id_cols
        + raw_stat_cols
        + [
            TARGET_COL,   # fantasy_points (label)
            BASELINE_COL, # rolling 5-game fantasy baseline
        ]
    )

    # All remaining columns are treated as features
    feature_cols = [c for c in df.columns if c not in drop_cols]
    return feature_cols


def eval_metrics(y_true, y_pred):
    """
    Compute standard regression metrics:
      - MAE: mean absolute error
      - RMSE: root mean squared error

    Both are reported for train and validation sets.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def build_model(model_type: str):
    """
    Build a scikit-learn Pipeline that:
      1) Standardizes features with StandardScaler
      2) Applies the chosen regression model

    Supported model types:
      - "linear": LinearRegression
      - "mlp":    MLPRegressor (two hidden layers: 64 -> 32)
    """
    if model_type == "linear":
        reg = LinearRegression()
    elif model_type == "mlp":
        # Simple feedforward neural network with early stopping.
        # This allows us to capture nonlinear relationships between features.
        reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Pipeline ensures that the same scaling is applied during training
    # and when we later call model.predict().
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", reg),
        ]
    )


def main(args):
    # ------------------------------------------------------------------
    # 1. Load train and validation splits
    # ------------------------------------------------------------------
    train_df = pd.read_csv(args.train_csv, parse_dates=[GAME_DATE_COL])
    val_df = pd.read_csv(args.val_csv, parse_dates=[GAME_DATE_COL])

    # Decide which columns are used as features
    feature_cols = get_feature_cols(train_df)
    print(f"[+] Using {len(feature_cols)} features")
    print(f"[+] Model type: {args.model_type}")

    # Build input (X) and target (y) matrices
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values

    X_val = val_df[feature_cols].values
    y_val = val_df[TARGET_COL].values

    # ------------------------------------------------------------------
    # 2. Initialize model (Linear or MLP)
    # ------------------------------------------------------------------
    model = build_model(args.model_type)

    # ------------------------------------------------------------------
    # 3. Fit model on training data
    # ------------------------------------------------------------------
    print("[+] Fitting model...")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 4. Evaluate model on train and validation sets
    # ------------------------------------------------------------------
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_mae, train_rmse = eval_metrics(y_train, y_train_pred)
    val_mae, val_rmse = eval_metrics(y_val, y_val_pred)

    metrics = {
        "model_type": args.model_type,
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "val_mae": float(val_mae),
        "val_rmse": float(val_rmse),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "feature_cols": feature_cols,
    }

    # ------------------------------------------------------------------
    # 5. Save trained model and metrics
    # ------------------------------------------------------------------
    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"[+] Saved model to {model_path}")

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Saved metrics to {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a regression model (Linear or MLP) to predict fantasy points "
            "using processed train/validation splits."
        )
    )
    parser.add_argument(
        "--train-csv",
        default="processed/train.csv",
        help="Path to processed training split.",
    )
    parser.add_argument(
        "--val-csv",
        default="processed/val.csv",
        help="Path to processed validation split.",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "mlp"],
        default="linear",
        help="Which model to train: 'linear' for LinearRegression, 'mlp' for MLPRegressor.",
    )
    parser.add_argument(
        "--model-out",
        default="models/linear_reg.pkl",
        help="Where to save the trained model (.pkl).",
    )
    parser.add_argument(
        "--metrics-out",
        default="results/train_val_metrics_linear.json",
        help="Where to save train/val metrics as JSON.",
    )
    args = parser.parse_args()
    main(args)
