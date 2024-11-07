
import json
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

def init():
    global model
    global transformations

    # Load the trained Keras model
    global model
    # Get the path to the deployed model file and load it
    
    model_dir =os.getenv('AZUREML_MODEL_DIR')
    print("model_dir", model_dir)
    keras_model_dir = os.listdir(model_dir)[0]
    # personalization_model_demo_tf
    full_model_path = os.path.join(model_dir,keras_model_dir,"azure_deployable_model.keras")
    print(full_model_path)
    model = load_model(full_model_path)
    print("model: sop last step")
    # model = mlflow.sklearn.load_model(model_path) 
    # Load transformations
    with open('transformations.json', 'r') as f:
        transformations = json.load(f)
    print("loaded transformations json")

def preprocess(data):
    # Convert JSON data to DataFrame
    print("In preprocess land")
    df = pd.DataFrame(data)
    
    # Apply transformations based on metadata
    processed_data = {}
    
    # Normalize continuous variables
    for feature in transformations["continuous"]:
        processed_data[feature["name"]] = (df[feature["name"]] - df[feature["name"]].mean()) / df[feature["name"]].std()

    # One-hot encode categorical variables
    for feature in transformations["categorical"]:
        one_hot = pd.get_dummies(df[feature["name"]], prefix=feature["name"], drop_first=True)
        for col in one_hot.columns:
            processed_data[col] = one_hot[col]
    
    return pd.DataFrame(processed_data)

def run(raw_data):
    # Parse the input JSON data
    print("in run raw_data land")
    data = json.loads(raw_data)['data']
    
    # Preprocess data
    input_data = preprocess(data)

    # Predict using the model
    predictions = model.predict(input_data)
    
    # Return predictions
    return {"predictions": predictions.tolist()}
