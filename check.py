import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

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
# print('Sample input data:\n',X_test[:1])

# # Create a sample based on above
features = np.array([0.09178,0.0,4.05,0,0.51,6.416,84.1,2.6463,5,296.0,16.6,395.5,9.04])
features_df = pd.DataFrame([features], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
# print('Test data:\n', features_df)


# Test predicting on logged mlflow model
import dagshub
import mlflow
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True)


scaling_model = 'runs:/dc23053edf584478aa0a894871287fd4/Standard Scaler'
logged_model = 'runs:/57236f5c745e45eea1dfbc3d9aab69fe/Linear Regression'

# Load models
scaled_model = mlflow.sklearn.load_model(scaling_model)
regression_model = mlflow.sklearn.load_model(logged_model)
print(type(scaled_model))
print(type(regression_model))



# Make prediction
features_scaled = scaled_model.transform(features_df)
prediction = regression_model.predict(features_scaled)
print('Prediction:\n', prediction)




