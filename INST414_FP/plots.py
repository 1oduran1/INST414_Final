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
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
):
    logger.info("Loading data...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Convert to Series

    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Running TimeSeriesSplit evaluation and generating confusion matrix...")

    tscv = TimeSeriesSplit()
    scores = []
    all_preds = []
    all_true = []
    for train_index, test_index in tqdm(tscv.split(X), total=tscv.get_n_splits()):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

        all_preds.extend(y_pred)
        all_true.extend(y_test)

    average_score = np.mean(scores)
    logger.info(f"Average accuracy score: {average_score:.4f}")

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