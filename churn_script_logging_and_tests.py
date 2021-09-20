# -*- coding: utf-8 -*-
"""
This module contains tests for functions within churn_library.py file.

Author: Marek Stelmach
Date: Spetember, 2021
"""

import logging
import os

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


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    pass


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    pass


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    pass


def test_train_models(train_models):
    '''
    test train_models
    '''
    pass


if __name__ == "__main__":
    test_import(cls.import_data)








