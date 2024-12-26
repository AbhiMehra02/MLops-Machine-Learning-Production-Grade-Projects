from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model

# cache = True means it will used previous pipeline data if it wouldn't change in code and (by default it is true)
# cache = False in any case it will run pipeline from the start either it will changing or not.
@pipeline(enable_cache=False)
def training_pipeline(data_path:str):

    df = ingest_df(data_path)
    clean_df(df)
    train_model(df)
    evaluate_model(df)