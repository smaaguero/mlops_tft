import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_absolute_error)
from sklearn.model_selection import GridSearchCV, train_test_split
from typing import Tuple, Callable

def scoring(rf_grid, X_train, y_train, X_test, y_test):
    print(f'Train Accuracy - : {rf_grid.score(X_train, y_train):.3f}')
    print(f'Test Accuracy - : {rf_grid.score(X_test, y_test):.3f}')

    y_pred = rf_grid.best_estimator_.predict(X_test)

    # mean absolute error

    mae = mean_absolute_error(y_test, y_pred)

    # confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_mat, annot = True, fmt = 'g')
    plt.title('Confusion Matrix of Placement Predictor')
    plt.ylabel('Real Place')
    plt.xlabel('Predicted Place')
    plt.show()

    # accuracy score
    print("Accuracy of model:", accuracy_score(y_test, y_pred))
    print("Mean Average Error: ", mae)


def filter_columns(X_full: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the necessary columns from the input DataFrame based on specific patterns.

    Args:
        X_full (pd.DataFrame): The full input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing only the filtered columns.
    """
    features = ['level', 'placement']

    augs = re.compile("augments.")
    trait_names = re.compile("traits_._name")
    trait_nums = re.compile("traits_._num")
    units_id = re.compile("units_._character")
    units_rarity = re.compile("units_._rarity")
    units_tier = re.compile("units_._tier")
    itemnames = re.compile("units_._itemNames")
    needed_columns = [augs, trait_names, trait_nums, units_id, units_rarity, units_tier, itemnames]

    for filters in needed_columns:
        features += list(filter(filters.match, X_full.columns))

    return X_full[features]

def convert_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the data types of columns from float64 to Int64.

    Args:
        X (pd.DataFrame): The input DataFrame with columns to be converted.

    Returns:
        pd.DataFrame: The DataFrame with converted data types.
    """
    for colname in list(X.select_dtypes("float64")):
        X[colname] = X[colname].astype(float).astype("Int64")
    return X

def create_total_item_feature(X: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature representing the total number of items and drops the original item columns.

    Args:
        X (pd.DataFrame): The input DataFrame containing item columns.

    Returns:
        pd.DataFrame: The DataFrame with the new feature and without the original item columns.
    """
    item_columns = [column for column in X.columns if 'item' in column]
    X['total_items'] = X[item_columns].count(axis='columns').copy()
    X = X.drop(item_columns, axis='columns')
    return X

def fill_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in string columns with '0' and in numeric columns with 0.

    Args:
        X (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: The DataFrame with filled missing values.
    """
    X[X.select_dtypes(include='string').columns] = X.select_dtypes(include='string').fillna('0')
    X[X.select_dtypes(include=['number']).columns] = X.select_dtypes(include=['number']).fillna(0)
    return X

def one_hot_encode(X: pd.DataFrame) -> pd.DataFrame:
    """
    Performs one-hot encoding on categorical features.

    Args:
        X (pd.DataFrame): The input DataFrame with categorical features.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded categorical features.
    """
    categoricals = [column for column in X.columns if ('augments' in column) or ('name' in column) or ('id' in column)]
    X = pd.get_dummies(X, columns=categoricals)
    X = X.replace(np.nan, 0)
    return X


def prepare_sets_data(
        df: pd.DataFrame,
        test_size: float,
        random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares data for 8, 4, and 2 class classifications.

    Args:
        X (pd.DataFrame): The input feature DataFrame.
        y (pd.Series): The target Series.

    Returns:
        Tuple: A tuple containing DataFrames and Series for 8, 4, and 2 class classifications.
    """
    y = df['placement']
    X = df.drop('placement', axis='columns')

    # X8 = X4 = X2 = X
    # y8 = y  # 8 possible classifications
    # y4 = y.replace({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4})  # 4 possible classifications
    # y2 = y.replace({1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2})  # 2 possible classifications

    # return X8, X4, X2, y8, y4, y2

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )

    return  X_train, X_test, y_train, y_test

def train_models(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series, 
        # param_grid: dict, 
) -> GridSearchCV:
    """
    Trains a RandomForest model using grid search and evaluates it.

    Args:
        X_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.Series): The training target Series.
        X_test (pd.DataFrame): The validation feature DataFrame.
        y_test (pd.Series): The validation target Series.
        param_grid (dict): The grid of hyperparameters to search.

    Returns:
        GridSearchCV: The trained model after grid search.
    """
    param_grid = {
        'n_estimators': [10, 100, 200],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [None],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }  

    forest_model = RandomForestClassifier()

    rf_grid = GridSearchCV(
        estimator=forest_model, 
        param_grid=param_grid, 
        cv=3, 
        verbose=2, 
        n_jobs=4
    )
    rf_grid.fit(X_train, y_train)
    
    print("Best parameters:", rf_grid.best_params_)
    scoring(rf_grid, X_train, y_train, X_test, y_test)
    
    return rf_grid