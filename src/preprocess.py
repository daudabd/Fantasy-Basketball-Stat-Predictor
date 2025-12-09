# src/preprocess.py
#
# Purpose:
#   Take the raw Kaggle PlayerStatistics.csv file and turn it into
#   clean, feature-rich datasets for model training:
#     - processed/train.csv
#     - processed/val.csv
#     - processed/test.csv
#
#   These processed files contain:
#     - Custom fantasy point target (fantasy_points)
#     - Rolling averages (3-game, 5-game) for recent form
#     - Days of rest between games
#     - Home/away indicator
#     - 5-game fantasy baseline (baseline_5g_fp)
#     - One-hot encodings for team and opponent
#
#   This is the "data & feature engineering" stage of the project.

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ================================
# COLUMN NAMES FOR YOUR DATASET
# (match exactly what appears in PlayerStatistics.csv)
# ================================
PLAYER_ID_COL = "personId"
PLAYER_FIRST_COL = "firstName"
PLAYER_LAST_COL = "lastName"
GAME_DATE_COL = "gameDateTimeEst"

TEAM_COL = "playerteamName"
OPP_COL = "opponentteamName"

MIN_COL = "numMinutes"
PTS_COL = "points"
REB_COL = "reboundsTotal"
AST_COL = "assists"
STL_COL = "steals"
BLK_COL = "blocks"
TOV_COL = "turnovers"

FGA_COL = "fieldGoalsAttempted"
FGM_COL = "fieldGoalsMade"
FG3M_COL = "threePointersMade"
FTA_COL = "freeThrowsAttempted"
FTM_COL = "freeThrowsMade"

# Name of our regression target and the simple baseline
TARGET_COL = "fantasy_points"
BASELINE_COL = "baseline_5g_fp"


# ================================
# Custom Fantasy Scoring Formula
#
# Based on the scoring system for this project:
#   PTS = 1
#   3PM = 1
#   FGA = -1
#   FGM = 2
#   FTA = -1
#   FTM = 1
#   REB = 1
#   AST = 2
#   STL = 4
#   BLK = 4
#   TOV = -2
#
# We apply this row-wise to compute fantasy_points for each game.
# ================================
def compute_fantasy_points(row: pd.Series) -> float:
    return (
        row[PTS_COL] * 1 +
        row[FG3M_COL] * 1 +
        row[FGA_COL] * -1 +
        row[FGM_COL] * 2 +
        row[FTA_COL] * -1 +
        row[FTM_COL] * 1 +
        row[REB_COL] * 1 +
        row[AST_COL] * 2 +
        row[STL_COL] * 4 +
        row[BLK_COL] * 4 +
        row[TOV_COL] * -2
    )


# ================================
# Feature builder
# ================================
def build_features(df: pd.DataFrame, min_history_games: int = 5) -> pd.DataFrame:
    """
    Add all engineered features needed for modeling:

      - fantasy_points (target variable)
      - is_home (home/away flag)
      - games_played_before (per-player game index)
      - days_rest (days since previous game)
      - rolling averages (3-game and 5-game) for:
          points, reboundsTotal, assists, numMinutes
      - baseline_5g_fp (5-game rolling average of fantasy_points)
      - one-hot team_*/opp_* columns

    We also enforce a minimum amount of history per player so that
    rolling-window features are meaningful.
    """

    # Sort by player and date so that rolling windows make sense
    df = df.sort_values([PLAYER_ID_COL, GAME_DATE_COL])

    # ------------------------------------------------------------------
    # 1. Target: fantasy points (label for supervised learning)
    # ------------------------------------------------------------------
    df[TARGET_COL] = df.apply(compute_fantasy_points, axis=1)

    # ------------------------------------------------------------------
    # 2. Home/away indicator
    # ------------------------------------------------------------------
    # The raw dataset provides a "home" column (True/False or 1/0).
    # We convert it to int to use directly as a numeric feature.
    df["is_home"] = df["home"].astype(int)

    # ------------------------------------------------------------------
    # 3. Games played before this one (per player)
    # ------------------------------------------------------------------
    # cumcount() gives 0, 1, 2, ... for each player's game history,
    # which we use both as context and for history filtering.
    df["games_played_before"] = df.groupby(PLAYER_ID_COL).cumcount()

    # ------------------------------------------------------------------
    # 4. Days of rest since previous game
    # ------------------------------------------------------------------
    # prev_date: the previous GAME_DATE for this player
    df["prev_date"] = df.groupby(PLAYER_ID_COL)[GAME_DATE_COL].shift(1)
    # days_rest: difference in days between current game and previous game
    df["days_rest"] = (df[GAME_DATE_COL] - df["prev_date"]).dt.days
    # For the first game (no previous date), we fill with a default value (3)
    df["days_rest"] = df["days_rest"].fillna(3)

    # ------------------------------------------------------------------
    # 5. Rolling averages for recent form (no leakage by using shift(1))
    # ------------------------------------------------------------------
    # For each player, we compute rolling means over the *previous*
    # 3 and 5 games. We always shift by 1 so we never peek at the
    # current game's stats when building features.
    for stat in [PTS_COL, REB_COL, AST_COL, MIN_COL]:
        df[f"{stat}_roll3"] = (
            df.groupby(PLAYER_ID_COL)[stat]
              .shift(1)                       # previous games only
              .rolling(3, min_periods=1)
              .mean()
        )
        df[f"{stat}_roll5"] = (
            df.groupby(PLAYER_ID_COL)[stat]
              .shift(1)
              .rolling(5, min_periods=1)
              .mean()
        )

    # ------------------------------------------------------------------
    # 6. Rolling fantasy baseline (5-game average of fantasy_points)
    # ------------------------------------------------------------------
    # This baseline is what we compare our ML model against:
    #   baseline_5g_fp = average fantasy points over the player's
    #   last 5 games before the current one.
    df[BASELINE_COL] = (
        df.groupby(PLAYER_ID_COL)[TARGET_COL]
          .shift(1)                           # only past games
          .rolling(5, min_periods=1)
          .mean()
    )

    # ------------------------------------------------------------------
    # 7. Filter out very early games with not enough history
    # ------------------------------------------------------------------
    # We require at least `min_history_games` previous games for a row
    # to be included. This stabilizes the rolling stats and baseline.
    df = df[df["games_played_before"] >= min_history_games].copy()

    # ------------------------------------------------------------------
    # 8. One-hot encode team and opponent
    # ------------------------------------------------------------------
    # Creates columns like team_Lakers, team_Celtics, opp_Heat, etc.
    df = pd.get_dummies(df, columns=[TEAM_COL, OPP_COL], prefix=["team", "opp"])

    # Drop helper column not needed by the model
    df = df.drop(columns=["prev_date"])

    return df


# ================================
# Train/Val/Test Split
# ================================
def split_by_date(df: pd.DataFrame, train_end: str, val_end: str):
    """
    Split the full dataset into train, validation, and test sets
    based on GAME_DATE_COL.

    - Train: dates < train_end
    - Val:   train_end <= dates < val_end
    - Test:  dates >= val_end

    Using chronological splits helps mimic real-world forecasting,
    where we only train on past data and evaluate on future games.
    """
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)

    train = df[df[GAME_DATE_COL] < train_end_ts].copy()
    val = df[(df[GAME_DATE_COL] >= train_end_ts) & (df[GAME_DATE_COL] < val_end_ts)].copy()
    test = df[df[GAME_DATE_COL] >= val_end_ts].copy()

    return train, val, test


def main(args):
    # ------------------------------------------------------------------
    # 1. Load raw CSV from Kaggle
    # ------------------------------------------------------------------
    print("[+] Loading raw CSV...")
    df = pd.read_csv(args.raw_csv, low_memory=False)

    # ------------------------------------------------------------------
    # 2. Parse game dates and normalize timezones
    # ------------------------------------------------------------------
    # The gameDateTimeEst column may contain timezone offsets
    # (e.g., "2022-10-25 19:30:00-04:00").
    # We parse them as UTC-aware datetimes, then drop the timezone info
    # so that all rows are consistent.
    print("[+] Parsing game dates (with timezone handling)...")
    df[GAME_DATE_COL] = pd.to_datetime(
        df[GAME_DATE_COL],
        utc=True,
        errors="coerce",
    )
    # Drop rows where the date could not be parsed
    df = df.dropna(subset=[GAME_DATE_COL])
    # Convert from timezone-aware to naive (no tz) timestamps
    df[GAME_DATE_COL] = df[GAME_DATE_COL].dt.tz_convert(None)

    # ------------------------------------------------------------------
    # 3. Basic cleaning: ensure minutes are numeric and > 0
    # ------------------------------------------------------------------
    print("[+] Cleaning data (filtering out 0 minutes)...")
    df[MIN_COL] = pd.to_numeric(df[MIN_COL], errors="coerce")
    df = df[df[MIN_COL] > 0].copy()

    # ------------------------------------------------------------------
    # 4. Build engineered features
    # ------------------------------------------------------------------
    print("[+] Building features...")
    df = build_features(df, min_history_games=args.min_history)

    # ------------------------------------------------------------------
    # 5. Chronological train/val/test split
    # ------------------------------------------------------------------
    print("[+] Splitting by date...")
    train, val, test = split_by_date(df, args.train_end, args.val_end)

    # ------------------------------------------------------------------
    # 6. Save processed splits to disk
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"[+] Saved processed files in {out_dir}/")
    print(f"    Train: {len(train)} rows")
    print(f"    Val:   {len(val)} rows")
    print(f"    Test:  {len(test)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PlayerStatistics.csv into train/val/test with engineered features."
    )
    parser.add_argument(
        "--raw-csv",
        required=True,
        help="Path to PlayerStatistics.csv (raw Kaggle export).",
    )
    parser.add_argument(
        "--out-dir",
        default="processed",
        help="Output directory for processed train/val/test splits.",
    )
    parser.add_argument(
        "--train-end",
        default="2017-01-01",
        help="Date that marks the end of the training set (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--val-end",
        default="2020-01-01",
        help="Date that marks the end of the validation set (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=5,
        help="Minimum number of previous games required for a player.",
    )
    args = parser.parse_args()
    main(args)
