#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = INST414_Final
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python

TRAIN_FEATURES_PATH = data/processed/train_features.csv
TEST_FEATURES_PATH = data/processed/holdout_features.csv
TRAIN_LABELS_PATH = data/processed/train_labels.csv
TEST_LABELS_PATH = data/processed/holdout_labels.csv

MODEL_PATH = models
MODEL_TYPE = both

PLOTS_PATH = reports/figures

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset (for training)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) INST414_FP/dataset.py

## Generate features and labels from dataset
.PHONY: features
features: requirements
	$(PYTHON_INTERPRETER) INST414_FP/features.py

## Train the model
.PHONY: train
train:
	$(PYTHON_INTERPRETER) INST414_FP/modeling/train.py \
		--features-path $(TRAIN_FEATURES_PATH) \
		--labels-path $(TRAIN_LABELS_PATH) \
		--model-path $(MODEL_PATH) \
		--model-type $(MODEL_TYPE)

## Run prediction script
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m INST414_FP.modeling.predict \
		--features-path $(TEST_FEATURES_PATH) \
		--model-path $(MODEL_PATH) \
		--predictions-path $(MODEL_PATH) \
		--model-type $(MODEL_TYPE)

## Generate plots
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) INST414_FP/plots.py \
		--labels-path $(TEST_LABELS_PATH) \
		--predictions-path $(MODEL_PATH) \
		--output-path $(PLOTS_PATH) \
		--model-type $(MODEL_TYPE)

## Run all tasks
.PHONY: all
all: data features train predict plots
	@echo "All tasks completed successfully!"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
