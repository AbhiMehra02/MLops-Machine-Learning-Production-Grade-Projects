# MLops-Machine-Learning-Production-Grade-Projects

## Setup
1. Create a new Conda environment
```bash
conda create -n mlops python=3.11  
```

2. Activate the environment
```bash
conda activate mlops  
```

3. Install required packages
Install the core dependencies:
```bash
pip install scikit-learn
pip install pandas  
pip install zenml  
pip install "zenml[server]"  
```
## Initialize ZenML
1. Initialize ZenML
```bash
zenml init  
```

2. Downgrade ZenML (if required)
```bash
zenml downgrade  
```

## Running the Pipeline (testing_purpose)
1. Run the pipeline
```bash
python run_pipeline.py  
```
### Results
```
Making predictions...
Calculation info
MSE : 1.8640770533975461
Calculation info
R2 Score : 0.017729030402296564
Calculation info
RMSE : 1.365312071798073
Step evaluate_model has finished in 0.993s.
```

2. Login to ZenML (local mode)
```bash
zenml login --local --blocking  
```