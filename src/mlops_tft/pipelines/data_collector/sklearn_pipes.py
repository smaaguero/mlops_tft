from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Optional, List

class NaNDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer for dropping rows and columns with all NaN values.

    This transformer removes rows that contain only NaN values and columns that
    contain only NaN values from the input DataFrame.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.

    transform(X):
        Transforms the input DataFrame by dropping rows and columns with all NaN values.

    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with all-NaN rows and columns removed.
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NaNDropper':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.dropna(
            how="all"
        ).dropna(
            axis="columns", how="all"
        )


class CorruptedDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer for dropping columns known to contain corrupted data.

    This transformer removes a predefined list of columns that are identified
    as corrupted from the input DataFrame.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by dropping predefined corrupted columns.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with corrupted columns removed.
    """
    def __init__(
            self, 
            corrupted_features: Optional[List[str]] = None
        ):
        if corrupted_features is None:
            self.corrupted_features = [
                "units_5_items_0", "units_5_items_1", 
                "units_5_items_2", "units_6_items_0", 
                "units_6_items_1", "units_6_items_2",
                "units_7_items_0", "units_7_items_1", 
                "units_7_items_2", "units_3_items_0", 
                "units_3_items_1", "units_0_items_0",
                "units_1_items_0", "units_1_items_1", 
                "units_2_items_0", "units_2_items_1", 
                "units_2_items_2", "units_1_items_2",
                "units_4_items_0", "units_4_items_1", 
                "units_4_items_2", "units_0_items_1", 
                "units_3_items_2", "units_0_items_2",
                "units_8_items_0", "units_8_items_1", 
                "units_8_items_2"
            ]
        else:
            self.corrupted_features = corrupted_features

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorruptedDropper':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.drop(
            self.corrupted_features, 
            axis="columns", 
            errors="ignore"
        )

        return X


class ResetIndex(BaseEstimator, TransformerMixin):
    """
    A custom transformer for resetting the index of a DataFrame.

    This transformer resets the index of the input DataFrame, optionally
    dropping the existing index.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by resetting its index.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with the index reset.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ResetIndex':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reset_index(drop=True)


class DescribeMissing(BaseEstimator, TransformerMixin):
    """
    A custom transformer for describing missing data in a DataFrame.

    This transformer calculates and prints the percentage of missing data in the
    input DataFrame but does not alter the DataFrame.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by calculating and printing the percentage of missing data.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to analyze.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame, unaltered.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DescribeMissing':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # get number of missing data points per column
        missing_values_count = X.isnull().sum()

        # how many missing values do we have?
        total_cells = np.prod(X.shape)
        total_missing = missing_values_count.sum()

        # percent of missing data
        percent_missing = (total_missing / total_cells) * 100
        print("Percent Missing of Data: ", str(percent_missing))

        return X


### Data Pipeline for ML
class TrainDropper(BaseEstimator, TransformerMixin):
    """
    A custom transformer for dropping non-training 
    features from a DataFrame.

    This transformer removes columns that are not relevant for 
    training from the input DataFrame.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by dropping non-training features.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with non-training features removed.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TrainDropper':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # remove features that don't help with training the data
        non_training_features = [
            "companion_content_ID",
            "companion_item_ID",
            "companion_skin_ID",
            "companion_species",
            "gold_left",
            "players_eliminated",
        ]

        for feature in non_training_features:
            try:
                X = X.drop(feature, axis="columns")
            except:  # noqa: E722
                continue

        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    A custom transformer for removing outliers from a DataFrame.

    This transformer removes columns from the input DataFrame that have a
    significant number of missing values (above a specified threshold).

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by removing columns with excessive missing values.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with columns removed if they have excessive missing values.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierRemover':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # remove outliers (10% threshold to not remove level 8 data)
        threshold = int(len(X) * 0.1)
        X = X.dropna(axis=1, thresh=threshold)

        return X


class GetAugmentDummies(BaseEstimator, TransformerMixin):
    """
    A custom transformer for creating dummy variables for augment features.

    This transformer converts categorical augment features into dummy/indicator
    variables in the input DataFrame.

    Methods
    -------
    fit(X, y=None):
        This method does nothing and is included for compatibility.
    
    transform(X):
        Transforms the input DataFrame by creating dummy variables for augment features.
    
    Parameters
    ----------
    X : pandas.DataFrame
        The input data to transform.
    y : None, optional
        Ignored parameter, present for compatibility.

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame with dummy variables for augment features.
    """
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'GetAugmentDummies':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        augments = ["augments_0", "augments_1", "augments_2"]
        X = pd.get_dummies(X, columns=augments)

        return X
