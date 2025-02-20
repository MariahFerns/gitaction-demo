import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template   # request: for http request, jsonify: convert request to json

import dagshub
import mlflow
dagshub.init(repo_owner='MariahFerns', repo_name='dagshub-demo', mlflow=True)


scaling_model = 'runs:/dc23053edf584478aa0a894871287fd4/Standard Scaler'
logged_model = 'runs:/57236f5c745e45eea1dfbc3d9aab69fe/Linear Regression'

# Load models
scaled_model = mlflow.sklearn.load_model(scaling_model)
regression_model = mlflow.sklearn.load_model(logged_model)


# Initialize Flask app
app = Flask(__name__)

# Define home route
@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict", methods=["POST"])
def predict():
    # Get the data from the request
    data = [float(x) for x in request.form.values()]
    
    # Convert features to dataframe
    features = np.array(data)
    features_df = pd.DataFrame([features], columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    
    # Make prediction
    features_scaled = scaled_model.transform(features_df)
    prediction = regression_model.predict(features_scaled)
    
    # Send back the prediction as JSON
    return render_template("home.html", prediction_text="The House price prediction is {}".format(prediction[0]))


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
