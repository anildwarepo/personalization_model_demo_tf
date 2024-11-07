import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Normalization, CategoryEncoding
from tensorflow.keras.optimizers import Adam

# Step 2: Preprocessing Layers
def preprocess_data(data, transformations):
    numerical_inputs = [Input(shape=(1,), name=f) for f in transformations["continuous"]]
    categorical_inputs = [Input(shape=(1,), name=f, dtype="int32") for f in transformations["categorical"]]

    # Apply normalization to continuous features
    normalized_numerical = [Normalization()(num_input) for num_input in numerical_inputs]

    # Apply one-hot encoding to categorical features
    encoded_categorical = [CategoryEncoding(num_tokens=card)(cat_input) for cat_input, card in zip(categorical_inputs, cardinalities)]

    # Concatenate all processed inputs
    processed_inputs = Concatenate()(normalized_numerical + encoded_categorical)

    # Step 3: Model Definition
    x = Dense(64, activation="relu")(processed_inputs)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    # Build the model
    model = Model(inputs=numerical_inputs + categorical_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, data, continuous_features, categorical_features):
    X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Y']), data['Y'], test_size=0.2, random_state=42)
    
    # Prepare input data in dictionary format for Keras
    X_train_dict = {name: X_train[name].values for name in continuous_features + categorical_features}
    X_test_dict = {name: X_test[name].values for name in continuous_features + categorical_features}

    # Train the model
    model.fit(X_train_dict, y_train, epochs=5, batch_size=128, validation_split=0.2)

    return model, X_test_dict, y_test

def save_model(model, model_path='azure_deployable_model.keras'):
    model.save(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Load data and transformations
    data = pd.read_csv("synthetic_data.csv")
    with open("transformations.json", "r") as f:
        transformations = json.load(f)
    
    # Train the model
    model = preprocess_data(data, transformations)
    model, X_test, y_test = train_model(model, data, continuous_features, categorical_features)
    
    # Save the trained model
    save_model(model)

