import logging
from abc import ABC,abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """
    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that used Mean Squared Error
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculation info")
            mse= mean_squared_error(y_true,y_pred)
            logging.info("MSE : {}".format(mse))
            return mse
        except Exception as e:
            logging.info("Error in calculation MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that used R2 Score
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculation info")
            r2= r2_score(y_true,y_pred)
            logging.info("R2 Score : {}".format(r2))
            return r2
        except Exception as e:
            logging.info("Error in calculation R2 Score: {}".format(e))
            raise e

class RMSE(Evaluation):
    """
    Evaluation Strategy that used root mean squared
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            logging.info("Calculation info")
            rmse= root_mean_squared_error(y_true,y_pred)
            logging.info("RMSE : {}".format(rmse))
            return rmse
        except Exception as e:
            logging.info("Error in calculation RMSE: {}".format(e))
            raise e