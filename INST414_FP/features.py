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
    input_path: Path = PROCESSED_DATA_DIR / "Clean_Final_Full_Housing_Classification.csv",
    output_path: Path = PROCESSED_DATA_DIR,
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")
    
    full_data = pd.read_csv(input_path)
    
    full_data.sort_values(by=['Date'])
    
    features = full_data.drop(columns=['Affordability', 'Date'])
    labels = full_data['Affordability']
    
    features.to_csv(output_path/ "features.csv", index=False)
    labels.to_csv(output_path/ "labels.csv", index=False)
    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
