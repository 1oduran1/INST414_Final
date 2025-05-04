from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
from sklearn.linear_model import LinearRegression
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
):
    logger.info("Loading data...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Make it a Series

    model = LinearRegression()
    tscv = TimeSeriesSplit(n_splits=5)

    logger.info("Training model with TimeSeriesSplit...")
    
    for (train_idx, test_idx) in enumerate(tqdm(tscv.split(X), total=5)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    logger.success("Model training complete and saved.")

if __name__ == "__main__":
    app()
