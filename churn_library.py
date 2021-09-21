# -*- coding: utf-8 -*-
"""
This module contains a set of functions for performing customer 
churn prediction analysis.

Author: Marek Stelmach
Date: Spetember, 2021
"""

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    
    return df


def perform_eda(df, cat_col, qnt_col, corr_heatmap=True):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
            cat_col: column name with categorical variable for bar plot
            qnt_col: column name with quantitative variable for histogram
            corr_heatmap: if True, a heatmap with correlation matrix is stored

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df[cat_col].value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/{}_cat_plot.png".format(cat_col))
    plt.close()
    
    plt.figure(figsize=(20,10)) 
    df[qnt_col].hist();
    plt.savefig("./images/eda/{}_qnt_plot.png".format(qnt_col))
    plt.close()
    
    if corr_heatmap:
        plt.figure(figsize=(20,10)) 
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig("./images/eda/corr_plot.png")
        plt.close()


def encoder_helper(df, category_lst, response, drop_cat=True):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]
            drop_cat: if True, original categorical columns are dropped from the data frame

    output:
            df: pandas dataframe with new column for each categorical feature
    '''
    for col in category_lst:
        col_lst = []
        col_groups = df.groupby(col).mean()[response]

        for val in df[col]:
            col_lst.append(col_groups.loc[val])

        df['{}_{}'.format(col, response)] = col_lst
    
    if drop_cat:
        df.drop(category_lst, axis=1, inplace=True)
    
    return df


def perform_feature_engineering(df, response, drop_cols=None, 
                                test_size=0.3, random_state=42):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]
              drop_cols: list of columns to drop from the data frame before any further engineering
              test_size: float, size of a test set
              random_state: int, random seed

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    all_to_drop = [response]
    if drop_cols is not None:
        all_to_drop += drop_cols
        
    y = df[response].copy()
    X = df.drop(all_to_drop, axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass


if __name__ == '__main__':
    churn_data = import_data("./data/bank_data.csv")
    churn_data['Churn'] = churn_data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    
    perform_eda(df=churn_data, cat_col='Marital_Status', 
                qnt_col='Customer_Age', corr_heatmap=True)
    
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    churn_data = encoder_helper(churn_data, cat_columns, 'Churn')
    
    columns_to_drop = ['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag']
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        churn_data, 'Churn', drop_cols=columns_to_drop)
