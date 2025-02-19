import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score





# Load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Read data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(config['data']['raw_data_path'], header=None, delimiter=r"\s+", names=column_names)



# Split data into X and y
X = df.drop('MEDV', axis=1)
y = df['MEDV']



# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=config['data']['test_size'], 
                                                    random_state=config['data']['random_state'])



# Mapping model names from string to actual class
model_mapping = {
    "StandardScaler": StandardScaler,
    "LinearRegression": LinearRegression
}


# Train models
models = [
    {
        'name': config['model1']['name'],
        'model': model_mapping[config['model1']['type']]        
    },
    {
        'name': config['model2']['name'],
        'model': model_mapping[config['model2']['type']]        
    }
]

# Get evaluation metrics from config
metrics = config['evaluation']['metrics']


# Track using DagsHub & MLFlow
import dagshub
import mlflow
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True) # collaborators can access this repo and experiments will be tracked under the same experiment name


# Set experiment name
mlflow.set_experiment(config['app']['name'])


# Train the models
for i, m in enumerate(models):
    model_name = m['name']
    model = m['model']()

    if model_name=='Standard Scaler':
        X_train = model.fit_transform(X_train)
        X_test = model.transform(X_test)
    else:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        r2score = r2_score(y_test, y_test_pred)

    with mlflow.start_run(run_name = model_name):
        # Log params
        mlflow.log_param('model', model_name)
       
        # Log metrics
        if model_name!='Standard Scaler':
            for metric in metrics:
                mlflow.log_metric(metric, r2score)
        # Log model
        mlflow.sklearn.log_model(model, f'{model_name}')