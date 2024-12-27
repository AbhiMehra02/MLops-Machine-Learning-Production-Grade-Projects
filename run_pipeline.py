from pipelines import training_pipeline
from steps.config import ModelNameConfig

if __name__=="__main__":
    "Run the Pipeline"
    config = ModelNameConfig(model_name="LinearRegression", model_kwargs={"fit_intercept": True})
    training_pipeline.training_pipeline(data_path="data/olist_customers_dataset.csv",config=config)