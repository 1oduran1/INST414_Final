"""
This script cleans and transforms a housing dataset by filtering, handling missing values, computing derived columns,
and saving the processed data to a specified file.

Arguments:
input_path: The path to the CSV file containing the joined interim dataset. Default is 
  Inner_Join_Affordability_Payment.csv located in the INTERIM_DATA_DIR.
output_path: The path where the cleaned and transformed dataset will be saved. Default is 
  Clean_Final_Full_Housing_Classification.csv located in the PROCESSED_DATA_DIR.

Output:
Clean_Final_Full_Housing_Classification.csv: A cleaned and transformed version of the input dataset.

Usage:
Run the script from the command line using make data.
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from INST414_FP.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "Inner_Join_Affordability_Payment.csv",
    output_path: Path = PROCESSED_DATA_DIR,
):
    logger.info("Starting data cleaning and transformation process...")

    # Load full joined dataset
    full_df = pd.read_csv(input_path)
    logger.info(f"Loaded data with shape: {full_df.shape}")

    # Filter out national rows
    full_df = full_df.query("`Region Name` != 'United States'")

    # Null handling
    null_df = full_df[full_df.isnull().any(axis=1)]
    region_nulls = null_df.groupby('Region Name').size().sort_values(ascending=False).to_dict()

    # Drop regions with > 6 nulls
    drop_regions = {k for k, v in region_nulls.items() if v > 6}
    full_df = full_df[~full_df['Region Name'].isin(drop_regions)]

    # Drop remaining rows with NA in key columns
    full_df = full_df.dropna(axis=0, subset=['Affordability Proportion', 'Monthly Payment'])

    # Round monthly payment
    full_df['Monthly Payment'] = full_df['Monthly Payment'].round(0).astype(int)

    # Income and affordability calculation
    def get_real_income(payment, affordability_proportion):
        monthly_income = payment / affordability_proportion
        annual_income = int(round(monthly_income * 12, 0))
        affordable_bool = 0 if affordability_proportion > 0.3 else 1
        return pd.Series([annual_income, affordable_bool])

    full_df[['Income', 'Affordability']] = full_df.apply(
        lambda x: get_real_income(x["Monthly Payment"], x["Affordability Proportion"]),
        axis=1
    )

    # Drop unused columns
    full_df = full_df.drop(columns=['Region Name', 'Monthly Payment', 'Affordability Proportion'])
    logger.info(f"Data cleaning complete. Cleaned data shape: {full_df.shape}")
    
    #create training and holdout sets
    full_df['Date'] = pd.to_datetime(full_df['Date'])
    train_df = full_df[full_df['Date'] < '2024-12-31']
    holdout_df = full_df[full_df['Date'] == '2024-12-31']
    
    # Export cleaned dataset
    train_df.to_csv(output_path/'clean_training_data.csv', index=False)
    holdout_df.to_csv(output_path/'clean_holdout_data.csv', index=False)
    
    logger.success(f"Cleaned datasets saved to: {output_path}")

if __name__ == "__main__":
    app()