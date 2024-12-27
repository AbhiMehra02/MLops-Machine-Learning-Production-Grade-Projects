import logging
import pandas as pd
from zenml import step
from src.evaluation import R2,RMSE,MSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin,
                   X_test:pd.DataFrame,
                   y_test:pd.Series)->Tuple[
                       Annotated[float,"r2"],
                       Annotated[float,"rmse"]
                   ]:
    """
    Evaluates the model on the ingested data.
    Args:
        model: The trained model (should be a regressor)
        X_test: The test feature data
        y_test: The test labels
    Returns:
        r2: The R2 score of the model
        rmse: The RMSE score of the model
    """
    try:
        # Ensure X_test and y_test are not empty
        if X_test.empty or y_test.empty:
            raise ValueError("Test data (X_test or y_test) cannot be empty.")

        # Make predictions
        logging.info("Making predictions...")
        prediction = model.predict(X_test)
        mse_class=MSE()
        mse=mse_class.calculate_scores(y_test,prediction)

        r2_class=R2()
        r2=r2_class.calculate_scores(y_test,prediction)

        rmse_class=RMSE()
        rmse=rmse_class.calculate_scores(y_test,prediction)

        return r2,rmse
    except Exception as e:
        logging.error("Error in Evaluating model: {}".format(e))
        raise e