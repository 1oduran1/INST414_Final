"""
This script loads test data, applies a trained machine learning model to generate predictions,
and saves the results to a specified file.

Arguments:
features_path: The path to the CSV file containing the test features. Default is the 
  features.csv located in the PROCESSED_DATA_DIR.
model_path: The path to the saved trained model file. Default is the model.pkl located in
  the MODELS_DIR.
predictions_path: The path where the predictions will be saved as a CSV file. Default is 
  predictions.csv located in the MODELS_DIR.

Output:
predictions_csv: The generated predictions are saved as a .csv file to the specified predictions_path.

Usage:
Run the script from the command line using 'make predict'"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import joblib

from INST414_FP.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "holdout_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "holdout_labels.csv",
    model_path: Path = MODELS_DIR,
    predictions_path: Path = MODELS_DIR,
    model_type: str = "both",
):
  
    logger.info("Loading test features...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Make it a Series

    logger.info("Loading trained model...")
    if model_type not in ["xgboost", "random_forest", "both"]:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    models = {
        "xgboost": model_path / "model_xgboost.pkl",
        "random_forest": model_path / "model_random_forest.pkl",
    }
    
    selected_models = models.keys() if model_type == "both" else [model_type]
    
    for m_type in selected_models:
        model = joblib.load(models[m_type])

        logger.info("Generating predictions...")
        y_pred = model.predict(X)

        logger.info("Saving predictions...")
        pd.Series(y_pred, name="Affordability Predictions").to_csv(predictions_path, index=False)

        save_path = predictions_path / f"predictions_{m_type}.csv"
        logger.success(f"Inference complete. Predictions saved to: {save_path}")

if __name__ == "__main__":
    app()
