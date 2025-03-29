import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.exception import CustomException

class StockCrashPrediction:
    def __init__(self, model_path):
        """Load the trained model"""
        try:
            self.model = load_model(model_path)
            print("✅ Model loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def load_test_data(self, test_data_path):
        """Loads and preprocesses test data"""
        try:
            df = pd.read_csv(test_data_path)

            if 'Crash' in df.columns:
                df = df.drop(columns=['Crash'])  # Remove target column if present

            # Convert all data to numeric format
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            # Reshape for LSTM input
            X_test = df.values.reshape((df.shape[0], df.shape[1], 1))

            return X_test
        except Exception as e:
            raise CustomException(e, sys)

    def make_prediction(self, test_data_path):
        """Predicts stock market crash based on test data"""
        try:
            X_test = self.load_test_data(test_data_path)

            predictions = self.model.predict(X_test)

            # Convert predictions to 0 (No Crash) or 1 (Crash)
            predictions = (predictions > 0.5).astype(int)

            return predictions
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    model_path = "artifacts/stock_crash_model.h5"
    test_data_path = "data/features.csv"

    predictor = StockCrashPrediction(model_path)
    predictions = predictor.make_prediction(test_data_path)

    print("🔹 Predictions:", predictions.flatten())  # Print predictions as 1D array
