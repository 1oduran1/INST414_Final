"""
This script evaluates a trained classification model using TimeSeriesSplit cross-validation,
computes the average accuracy, and generates a confusion matrix heatmap for visualization.

Arguments:
features_path: Path to the CSV file containing feature data. Default is features.csv in PROCESSED_DATA_DIR.
labels_path: Path to the CSV file containing label data. Default is labels.csv in PROCESSED_DATA_DIR.
model_path: Path to the trained model file (.pkl). Default is model.pkl in MODELS_DIR.
output_path: Path where the confusion matrix heatmap image will be saved. Default is confusion_matrix.png in FIGURES_DIR.

Output:
Confusion matrix heatmap plot (PNG format), visualizing model prediction performance.

Usage:
Run the script from the command line using make plots
"""


from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from INST414_FP.config import FIGURES_DIR, PROCESSED_DATA_DIR, MODELS_DIR

app = typer.Typer()


@app.command()
def main(
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
):
    logger.info("Loading data...")
    all_preds = pd.read_csv(predictions_path).squeeze()
    all_true = pd.read_csv(labels_path).squeeze()
    logger.info("Generating confusion matrix...")

    cm = confusion_matrix(all_true, all_preds)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)

    labels = [[f"{count}\n({percent:.1%})" for count, percent in zip(row_c, row_p)]
              for row_c, row_p in zip(cm, cm_percent)]

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f'Confusion Matrix (TimeSeriesSplit)\nAvg Accuracy: {average_score:.2%}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    logger.info(f"Saving confusion matrix to {output_path}...")
    plt.savefig(output_path)
    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()