import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_absolute_error)
from sklearn.model_selection import GridSearchCV, train_test_split
from typing import Tuple
import joblib
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from .optuna_utils import Optimizer

def scoring(model, X_train, y_train, X_test, y_test):
    dtest = xgb.DMatrix(X_test)

    y_pred = model.predict(dtest)
    # confusion matrix
    conf_mat = confusion_matrix(y_test, (y_pred >= 0.5).astype(int))
    sns.heatmap(conf_mat, annot = True, fmt = 'g')
    plt.title('Confusion Matrix of Placement Predictor')
    plt.ylabel('Real Place')
    plt.xlabel('Predicted Place')
    plt.show()

    # accuracy score
    print("Accuracy of model:", accuracy_score(
        y_test, 
        (y_pred >= 0.5).astype(int))
    )


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
    y = y.replace({1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0})  # 2 possible classifications

    # return X8, X4, X2, y8, y4, y2

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state
    )

    # Save the data
    X_train.to_parquet(
        'data/X_train.parquet', 
        index = False
    )
    y_train.to_frame().to_parquet('data/y_train.parquet', index = False)
    X_test.to_parquet(
        'data/X_test.parquet', 
        index = False
    )
    y_test.to_frame().to_parquet('data/y_test.parquet', index = False)

    return  X_train, X_test, y_train.to_frame(), y_test.to_frame()

def train_models(
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame, 
        y_test: pd.Series,
        n_trials: int,
        grid_to_search: dict
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
    # Get the train and validation data sets.
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    v_X_train, v_X_valid, v_y_train, v_y_valid = [], [], [], []
    for train_index, valid_index in kf.split(X_train, y_train):
        v_X_train.append(X_train.iloc[train_index])
        v_X_valid.append(X_train.iloc[valid_index])
        v_y_train.append(y_train.iloc[train_index])
        v_y_valid.append(y_train.iloc[valid_index])

    data = {
        'v_X_train': v_X_train,
        'v_X_valid': v_X_valid,
        'v_y_train': v_y_train,
        'v_y_valid': v_y_valid
    }


    optimizer = Optimizer(grid_to_search, data)
    best_trial = optimizer.optimize(direction='maximize', n_trials=n_trials)

    print("Best trial parameters:", best_trial.params)
    print("Best trial score:", best_trial.value)

    # Train the final model using the best hyperparameters
    dtrain = xgb.DMatrix(X_train, label=y_train)
    final_model = xgb.train(best_trial.params, dtrain)


    scoring(final_model, X_train, y_train, X_test, y_test)


    joblib.dump(final_model, 'etc/xgboost.pkl')    

    return final_model


