# -*- coding: utf-8 -*-
"""
This module contains tests for functions within churn_library.py file.

Author: Marek Stelmach
Date: Spetember, 2021
"""

import glob
import logging
import os

import pandas as pd

import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df, 'Marital_Status', 'Customer_Age', 
                    corr_heatmap=True)
        num_plots = len(glob.glob("./images/eda/*.png"))
        assert num_plots == 3
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_eda: Three png plots required, found {}".format(num_plots))
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    try:
        df = encoder_helper(df, cat_columns, 'Churn')
        assert all(['{}_{}'.format(col, 'Churn') 
                    in df.columns for col in cat_columns])
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        msg = ("Testing encoder_helper: Encoded categorical "
               + "columns not available in data frame")
        logging.error(msg)
        raise err
        
    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    # test for returning exactly four objects
    datasets = perform_feature_engineering(
        df, 'Churn', ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'])
    try:
        assert len(datasets) == 4
        logging.info("Testing perform_feature_engineering: SUCCESS, four objects returned")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Four objects required to be returned")
        raise err
    
    # test for comparing dimensions of X and y with original df
    X_train, X_test, y_train, y_test = datasets
    X_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
              'Total_Relationship_Count', 'Months_Inactive_12_mon',
              'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
              'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
              'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
              'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
              'Income_Category_Churn', 'Card_Category_Churn']
    
    try:
        assert pd.concat([y_train, y_test]).shape == (df.shape[0],)
        logging.info("Testing perform_feature_engineering: SUCCESS, y has correct shape")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Incorrect shape of y")
        raise err

    try:
        assert pd.concat([X_train, X_test]).shape == df[X_cols].shape
        logging.info("Testing perform_feature_engineering: SUCCESS, X has correct shape")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Incorrect shape of X")
        raise err
        
    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)
    
    model_files = os.listdir('./models')
    try:
        assert all([file in model_files for file 
                    in ['rf_model.pkl', 'lr_model.pkl']])
        logging.info("Testing test_train_models: SUCCESS, all models found")
    except AssertionError as err:
        logging.error("Testing test_train_models: Incomplete models list")
        raise err
    
    result_files = os.listdir('./images/results')
    try:
        assert all([file in result_files for file 
                    in ['classification_report.png',
                        'feature_importance.png',
                        'roc_curve.png']])
        logging.info("Testing test_train_models: SUCCESS, all results files found")
    except AssertionError as err:
        logging.error("Testing test_train_models: Incomplete results list")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    churn_data = cls.import_data("./data/bank_data.csv")
    churn_data['Churn'] = churn_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_eda(cls.perform_eda, churn_data)
    churn_data = test_encoder_helper(cls.encoder_helper, churn_data)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, churn_data)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
