# INST414_FP

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         INST414_FP and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── INST414_FP   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes INST414_FP a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

Project description:

List all dependencies:
    sklearn requires Python (>= 3.9), NumPy (>= 1.19.5), SciPy (>= 1.6.0), joblib (>= 1.2.0), threadpoolctl (>= 3.1.0)
    pandas has optional dependencies numexpr (>= 2.8.4), bottleneck (>= 1.3.6), numba (>=0.56.4)
        Improves speed, handling of large datasets


Include instructions for:
Setting up the environment:
    Set up a new environment using Python 3.9. Install sklearn and pandas. 

Running the data processing pipeline:
    Run "make data" in the terminal to process the dataset "Inner_Join_Affordability_Payment.csv" from the interim data folder 
    into "Clean_Final_Full_Housing_Classification.csv" which will be in the processed data folder. 

    Run "make features" in the terminal to split "Clean_Final_Full_Housing_Classification.csv" into a dataset containing features and 
    another containing labels. Both datasets will also be in the processed data folder.

Training models: 
    Run "make train" in the terminal. Specify the argument make train MODEL_TYPE=
    Options for the MODEL_TYPE argument include "xgboost" or "random_forest" or "both" to train one or both models
    The model.pkl files get saved to the models folder. Default is "both".

Evaluating models:
    Run "make predict" in the terminal. Specify the argument make train MODEL_TYPE=
    Options for the MODEL_TYPE argument include "xgboost" or "random_forest" or "both" to train one or both models
    The model.pkl files get saved to the models folder. Default is "both".

    Run "make plots"

Reproducing results:
    Run "make all" to run 