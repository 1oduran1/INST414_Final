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
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = MODELS_DIR / "predictions.csv",
):
    logger.info("Loading test features...")
    X_test = pd.read_csv(features_path)

    logger.info("Loading trained model...")
    model = joblib.load(model_path)

    logger.info("Generating predictions...")
    y_pred = model.predict(X_test)

    logger.info("Saving predictions...")
    pd.Series(y_pred, name="Affordability").to_csv(predictions_path, index=False)

    logger.success(f"Inference complete. Predictions saved to: {predictions_path}")

if __name__ == "__main__":
    app()
