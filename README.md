# Fantasy Basketball Stat Predictor

🏀 Fantasy Basketball Stat Predictor

CS 549 – Final Project
Daud Abdinasir & Buchard Joseph

📌 Project Summary

This repository contains the full machine learning pipeline developed for forecasting fantasy basketball points using historical NBA player statistics. The project includes:

Data preprocessing and feature engineering

Model training (Linear Regression and MLP Neural Network)

Baseline comparison using rolling averages

Test-time evaluation

Visualization of prediction accuracy and error distributions

The goal is to build a reproducible and interpretable workflow demonstrating how engineered features and machine learning models can improve fantasy point projections over simple heuristics.

📁 Repository Structure
Fantasy-Basketball-Stat-Predictor/
│
├── src/ # All project scripts
│ ├── preprocess.py # Data cleaning, feature engineering, train/val/test split
│ ├── train_model.py # Linear + MLP model training
│ ├── test_model.py # Test set evaluation and baseline comparison
│ └── visualize_results.py # Scatterplots and error histograms
│
├── data/ # Raw Kaggle input (not tracked in repo)
│ └── PlayerStatistics.csv  
│
├── processed/ # Preprocessed train/val/test splits (excluded from repo)
│
├── models/ # Trained model artifacts
│ ├── linear_reg.pkl
│ └── mlp_reg.pkl
│
├── results/ # Training & test metrics and prediction outputs
│ ├── train_val_metrics_linear.json
│ ├── train_val_metrics_mlp.json
│ ├── test_metrics_linear.json
│ ├── test_metrics_mlp.json
│ ├── test_predictions_linear.csv
│ └── test_predictions_mlp.csv
│
├── plots_linear/ # Linear Regression visualization outputs
├── plots_mlp/ # MLP visualization outputs
│
└── README.md

This organization ensures clarity, reproducibility, and compliance with the assignment’s requirements.

⚙️ Environment Setup

1. Create and activate a virtual environment
   python3 -m venv venv
   source venv/bin/activate # macOS/Linux

# or

venv\Scripts\activate # Windows

2. Install dependencies
   pip install -r requirements.txt

Required packages:

pandas

numpy

scikit-learn

matplotlib

joblib

🧹 Data Preprocessing

The preprocessing stage:

Cleans the raw Kaggle dataset

Normalizes datetime formats

Computes custom fantasy scoring

Generates rolling player statistics (3-game & 5-game averages)

Computes rest days and home/away indicators

Builds a 5-game rolling fantasy baseline

Produces chronological train/validation/test splits

Ensures no future information leaks into training features

Run preprocessing:

python src/preprocess.py \
 --raw-csv data/PlayerStatistics.csv \
 --out-dir processed \
 --train-end 2017-01-01 \
 --val-end 2020-01-01

Output:

processed/train.csv
processed/val.csv
processed/test.csv

🤖 Model Training

Two models are implemented:

1. Linear Regression

Serves as a simple, interpretable baseline ML model.

2. MLP Neural Network

A feed-forward network that captures nonlinear interactions in the feature set.

Both models use a standardized feature pipeline (StandardScaler → Model).

Train linear regression:

python src/train_model.py \
 --model-type linear \
 --model-out models/linear_reg.pkl \
 --metrics-out results/train_val_metrics_linear.json

Train MLP neural network:

python src/train_model.py \
 --model-type mlp \
 --model-out models/mlp_reg.pkl \
 --metrics-out results/train_val_metrics_mlp.json

Metrics include MAE and RMSE on training and validation splits.

🧪 Model Evaluation

Models are evaluated on an unseen test set and compared against a simple baseline:

Baseline:

5-game rolling average of fantasy points per player.

Evaluate linear model:
python src/test_model.py \
 --model-path models/linear_reg.pkl \
 --metrics-out results/test_metrics_linear.json \
 --preds-out results/test_predictions_linear.csv

Evaluate MLP model:
python src/test_model.py \
 --model-path models/mlp_reg.pkl \
 --metrics-out results/test_metrics_mlp.json \
 --preds-out results/test_predictions_mlp.csv

Metrics written to JSON include:

test_model_mae

test_model_rmse

test_baseline_mae

test_baseline_rmse

These values directly support the Experimental Results section of the report.

📈 Visualization

Use the visualization script to generate:

Predicted vs Actual scatter plot

Error distribution histogram (Model vs Baseline)

Example (Linear model):

python src/visualize_results.py \
 --preds-csv results/test_predictions_linear.csv \
 --test-metrics results/test_metrics_linear.json \
 --plots-out plots_linear/

Example (MLP model):

python src/visualize_results.py \
 --preds-csv results/test_predictions_mlp.csv \
 --test-metrics results/test_metrics_mlp.json \
 --plots-out plots_mlp/

Plots are stored in the specified output folder and may be included in the final report.

🔁 Reproducibility

This project satisfies the assignment’s reproducibility requirements:

Each step (preprocessing, training, testing, visualization) is handled by a standalone script.

Scripts can be executed in sequence to regenerate all results.

Parameters (dates, model type, paths) are configurable via command-line flags.

All models, metrics, and plots are stored in version-controlled folders.

👥 Contributors

Daud Abdinasir

Data acquisition

Preprocessing & feature engineering

Repository organization

Buchard Joseph

Model development

Training & evaluation

Visualization & analysis
