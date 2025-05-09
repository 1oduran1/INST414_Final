"""
This script loads a dataset, generates features and labels, and saves them to specified files.

Arguments:
input_path: The path to the CSV file containing the raw dataset. Default is 'Clean_Final_Full_Housing_Classification.csv'
  located in the PROCESSED_DATA_DIR.
output_path: The directory where the generated features and labels will be saved. Default is the PROCESSED_DATA_DIR.

Output:
features.csv: The features extracted from the dataset, saved to the specified output_path.
labels.csv: The labels extracted from the dataset, saved to the specified output_path.

Usage:
Run the script from the command line using 'make features'.
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd


from INST414_FP.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    
    datasets = {'train': pd.read_csv(input_path/'clean_training_data.csv'),
                'holdout': pd.read_csv(input_path/'clean_holdout_data.csv')}

    for name, data in datasets.items():
        data.sort_values(by=['Date'])
        features = data.drop(columns=['Affordability', 'Date'])
        labels = data['Affordability']
        features.to_csv(output_path/ f"{name}_features.csv", index=False)
        logger.info(f"Saved {name} to {output_path/ f'{name}_features.csv'}")
        labels.to_csv(output_path/ f"{name}_labels.csv", index=False)
        logger.info(f"Saved {name} to {output_path/ f'{name}_labels.csv'}")
    # -----------------------------------------


if __name__ == "__main__":
    app()
