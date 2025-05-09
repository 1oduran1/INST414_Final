"""
This script loads test data, trains a classification model using TimeSeriesSplit cross-validation,
and saves the trained model to a specified file.

Arguments:
features_path: The path to the CSV file containing the feature data. Default is the 
  features.csv located in the PROCESSED_DATA_DIR.
labels_path: The path to the CSV file containing the labels. Default is the labels.csv located 
  in the PROCESSED_DATA_DIR.
model_path: The path where the trained model will be saved as a pickle file. Default is model.pkl
  located in the MODELS_DIR.
model_type: The type of model to train. Options are 'xgboost'(default) or 'random_forest'.

Output:
model.pkl: The trained model is saved as a .pkl file to the specified model_path.

Usage:
Run the script from the command line using 'make train'.
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib

from INST414_FP.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    model_type: str = "xgboost" or "random_forest" or "both",
):
    logger.info("Loading data...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Make it a Series

    models = {
        "xgboost": XGBClassifier(n_estimators=1200, max_depth=3, learning_rate=1, random_state=0),
        "random_forest": RandomForestClassifier(n_estimators=700, random_state=0),
    }
    
    if model_type not in ["xgboost", "random_forest", "both"]:
        raise ValueError(f"Unsupported model type: {model_type}")
      
    selected_models = models.keys() if model_type == "both" else [model_type]
    
    for m_type in selected_models:
        model = models[m_type]
        tscv = TimeSeriesSplit(n_splits=5)

        logger.info(f"Training {m_type} model with TimeSeriesSplit...")
        for train_idx, test_idx in tqdm(tscv.split(X), total=5):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        save_path = model_path / f"model_{m_type}.pkl"
        logger.info(f"Saving {m_type} model to {save_path}")
        joblib.dump(model, save_path)

    logger.success("Selected model(s) trained and saved.")

if __name__ == "__main__":
    app()
