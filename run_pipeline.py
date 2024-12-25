from pipelines import training_pipeline

if __name__=="__main__":
    "Run the Pipeline"
    training_pipeline.training_pipeline(data_path="data/olist_customers_dataset.csv")