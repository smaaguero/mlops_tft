from typing import Dict, Any, List, Union
import optuna
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


class Optimizer:
    """
    A class used to optimize hyperparameters using Optuna for an XGBoost model.

    Attributes:
    -----------
    param_dic : Dict[str, Dict[str, Union[List[int], List[float]]]]
        A dictionary containing the hyperparameter ranges to be optimized.
    data : Dict[str, List[np.ndarray]]
        A dictionary containing the training and validation datasets.
    """

    def __init__(
        self, 
        param_dict: Dict[str, Dict[str, Union[List[int], List[float]]]], 
        data: Dict[str, List[np.ndarray]]
    ) -> None:
        """
        Initializes the Optimizer with a dictionary of parameters and data.

        Parameters:
        -----------
        param_dict : Dict[str, Dict[str, Union[List[int], List[float]]]]
            A dictionary specifying the types and ranges of parameters to optimize.
        data : Dict[str, List[np.ndarray]]
            A dictionary containing the training and validation datasets.
        """
        self.param_dic = param_dict
        self.data = data
        
    def create_param_dict(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Creates a dictionary of suggested hyperparameters for the trial.

        Parameters:
        -----------
        trial : optuna.Trial
            The trial object that suggests values for the hyperparameters.

        Returns:
        --------
        Dict[str, Any]
            A dictionary containing the suggested hyperparameters.
        """
        suggested_param = {}  # Param storage

        # Add fixed parameters if any
        one_param = self.param_dic.get('contexto', None)
        if one_param is not None:     
            for k, v in one_param.items():
                suggested_param.update({k: v})
                
        # Add integer parameters
        int_param = self.param_dic.get('int', None)
        if int_param is not None:     
            for k, v in int_param.items():
                suggested = trial.suggest_int(k, v[0], v[1])
                suggested_param.update({k: suggested})
        
        # Add float parameters
        float_param = self.param_dic.get('float', None)
        if float_param is not None:     
            for k, v in float_param.items():
                suggested = trial.suggest_float(k, v[0], v[1])
                suggested_param.update({k: suggested})

        # Add log-uniform parameters
        loguniform_param = self.param_dic.get('loguniform', None)
        if loguniform_param is not None:
            for k, v in loguniform_param.items():
                suggested = trial.suggest_float(k, v[0], v[1], log=True)
                suggested_param.update({k: suggested})
                
        return suggested_param
    
    def optimization_function(self, trial: optuna.Trial) -> float:
        """
        The objective function to optimize. It trains the XGBoost model with 
        suggested hyperparameters and calculates the mean AUC score across 
        cross-validation folds.

        Parameters:
        -----------
        trial : optuna.Trial
            The trial object that suggests values for the hyperparameters.

        Returns:
        --------
        float
            The mean AUC score from cross-validation.
        """
        # Create the dictionary with XGBoost parameters         
        suggested_param = self.create_param_dict(trial)

        # List to store AUC scores for each cross-validation fold
        auc_scores = []

        for index in range(len(self.data['v_X_train'])):
            # Get the training and validation data for the current fold
            X_sub_train, X_val = self.data['v_X_train'][index], self.data['v_X_valid'][index]
            y_sub_train, y_val = self.data['v_y_train'][index], self.data['v_y_valid'][index]

            # Prepare the data for XGBoost
            dtrain = xgb.DMatrix(X_sub_train, label=y_sub_train.placement)
            dval = xgb.DMatrix(X_val, label=y_val.placement)

            # Train the XGBoost model
            model = xgb.train(suggested_param, dtrain, evals=[(dval, 'valid')], verbose_eval=False)
            
            # Predict and calculate the AUC score
            y_pred = model.predict(dval)
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)

        # Return the average AUC score
        return np.mean(auc_scores)


    def optimize(
        self, 
        direction: str, 
        n_trials: int
    ) -> optuna.trial.FrozenTrial:
        """
        Runs the hyperparameter optimization using Optuna.

        Parameters:
        -----------
        direction : str
            The optimization direction, either 'minimize' or 'maximize'.
        n_trials : int
            The number of trials to run in the optimization.

        Returns:
        --------
        optuna.trial.FrozenTrial
            The best trial with the optimal set of hyperparameters.
        """
        study = optuna.create_study(direction=direction)
        study.optimize(self.optimization_function, n_trials=n_trials, n_jobs=2)
        return study.best_trial
