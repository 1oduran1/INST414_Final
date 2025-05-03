from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np

from INST414_FP.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "Inner_Join_Affordability_Payment.csv",
    output_path: Path = PROCESSED_DATA_DIR / "Clean_Final_Full_Housing_Classification.csv",
):
    logger.info("Starting data cleaning and transformation process...")

    # Load full joined dataset
    df = pd.read_csv(input_path)
    logger.info(f"Loaded data with shape: {df.shape}")

    # Filter out national rows
    df = df.query("`Region Name` != 'United States'")

    # Null handling
    null_df = df[df.isnull().any(axis=1)]
    region_nulls = null_df.groupby('Region Name').size().sort_values(ascending=False).to_dict()

    # Drop regions with > 6 nulls
    drop_regions = {k for k, v in region_nulls.items() if v > 6}
    df = df[~df['Region Name'].isin(drop_regions)]

    # Drop remaining rows with NA in key columns
    df = df.dropna(axis=0, subset=['Affordability Proportion', 'Monthly Payment'])

    # Round monthly payment
    df['Monthly Payment'] = df['Monthly Payment'].round(0).astype(int)

    # Income and affordability calculation
    def get_real_income(payment, affordability_proportion):
        monthly_income = payment / affordability_proportion
        annual_income = int(round(monthly_income * 12, 0))
        affordable_bool = 0 if affordability_proportion > 0.3 else 1
        return pd.Series([annual_income, affordable_bool])

    df[['Income', 'Affordability']] = df.apply(
        lambda x: get_real_income(x["Monthly Payment"], x["Affordability Proportion"]),
        axis=1
    )

    # Drop unused columns
    df = df.drop(columns=['Region Name', 'Monthly Payment', 'Affordability Proportion'])

    # Export cleaned dataset
    df.to_csv(output_path, index=False)
    logger.success(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    app()