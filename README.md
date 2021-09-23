# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This projects presents a full (but simplified) machine learning pipeline to create simple models i.e., Logistic Regression (LR) and Random Forest (RF) for customer churn prediction on [Kaggle's data](https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv). In particular, below steps are included:

1. Exploratory Data Analysis
2. Feature Engineering
3. Model Training
4. Prediction
5. Model Evaluation

The code is designed not to produce a complex solution and state-of-the-art results but rather to resemble best practices from production level environment with coding standards following the PEP 8 guidelines, tests written for each function and clear log file for info and errors.<br>Functions required to run this project (all aforementioned steps) are available in `churn_library.py` while corresponding tests are stored in `churn_script_logging_and_tests.py`.

## Installation
To clone this repo please use:
```
git clone https://github.com/mrstelmach/Predict-Customer-Churn.git
```
All packages required for conducting this project can be installed using `requirements.txt` file and running:
```
pip install -r requirements.txt
```
Project workflow has been tested with **Python 3.6.3**.

## Running Files
To run tests and reproduce results from `logs/churn_library.log` run the follwing:
```
python churn_script_logging_and_tests.py
```
The whole ml pipeline can be run with:
```
python churn_library.py
```
This will produce the following results:
1. EDA results for sample variables in `images/eda/` folder
2. Two model objects (LR and RF) available in `models/` folder
3. Model evaluation in `images/results/` (ROC curve and classification report) with additional feature importance plot
