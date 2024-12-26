from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


# cache = True means it will use previous pipeline data if the inputs haven't changed (by default, it is true).
# cache = False means it will run the pipeline from the start every time, regardless of any changes.
@pipeline(enable_cache=True)
def training_pipeline(data_path:str):

    df = ingest_df(data_path)
    X_train,X_test,y_train,y_test = clean_df(df)
    model = train_model(X_train,X_test,y_train,y_test)
    r2_score , rmse = evaluate_model(model,X_test,y_test)