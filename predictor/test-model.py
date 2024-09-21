import joblib
import numpy as np
import pytest

# Load the model
model = joblib.load('model.pkl')


def test_model_performance():
    # Using test data from the CSV
    test_data = np.array([[29.41, 88.5, 0.0], [29.64, 79.0, 3.8]])

    # Get predictions
    predictions = model.predict(test_data)

    # Expected output (approximations for regression)
    expected_output = [0.32, 0.32]

    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predictions - expected_output))

    # Define a tolerance for error
    tolerance = 0.8  # Adjust this based on acceptable error

    assert mae <= tolerance, f"Model MAE exceeded tolerance. MAE: {mae}"
