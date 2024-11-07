import numpy as np
import pandas as pd
import json

# Step 1: Data Generation
def generate_data(num_samples=100000):
    continuous_features = [f'cont_{i}' for i in range(150)]
    categorical_features = [f'cat_{i}' for i in range(50)]
    cardinalities = [np.random.randint(2, 20) for _ in range(50)]  # random cardinality for each categorical feature

    data = pd.DataFrame({f: np.random.rand(num_samples) * np.random.randint(1, 100) for f in continuous_features})
    for i, f in enumerate(categorical_features):
        data[f] = np.random.randint(0, cardinalities[i], size=num_samples)

    # Generate binary target variable
    data['Y'] = np.random.randint(0, 2, size=num_samples)
    
    return data, continuous_features, categorical_features, cardinalities

def save_transformations(continuous_features, categorical_features, cardinalities):
    transformations = {
        "continuous": [{"name": f"cont_{i}", "type": "normalization"} for i in range(150)],
        "categorical": [{"name": f"cat_{i}", "type": "one_hot_encoding", "cardinality": cardinalities[i]} for i in range(50)]
    }
    with open("transformations.json", "w") as f:
        json.dump(transformations, f)

if __name__ == "__main__":
    # Generate synthetic data
    data, continuous_features, categorical_features, cardinalities = generate_data()

    # Save transformations metadata
    save_transformations(continuous_features, categorical_features, cardinalities)

    # Save data for training (optional)
    data.to_csv("synthetic_data.csv", index=False)
    print("Data and transformations saved successfully.")

