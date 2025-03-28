import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class TrainModelConfig:
    train_obj_file_path = os.path.join('artifacts', 'train.pkl')

    def load_data(self):
        try:
            df = pd.read_csv("data/features.csv")

            # Check for object columns
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                print(f"Converting object columns to numeric: {object_cols}")
                df[object_cols] = df[object_cols].apply(pd.to_numeric, errors='coerce')

            # Ensure no NaN values exist
            df.fillna(0, inplace=True)

            # Check if 'Crash' column exists
            if 'Crash' not in df.columns:
                raise ValueError("Error: 'Crash' column not found in dataset!")

            # Separate features and target variable
            X = df.drop(columns=['Crash']).values  # Convert to NumPy array
            y = df['Crash'].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Ensure X_train has the correct 3D shape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)


    def build_lstm_model(self, input_shape):
        try:
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(1, activation="sigmoid")
            ])
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            return model
        except Exception as e:
            raise CustomException(sys, e)

    def train_model(self):
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data()

            # Ensure correct input shape for LSTM
            input_shape = (X_train.shape[1], X_train.shape[2])

            # Build and train the model
            model = self.build_lstm_model(input_shape)
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

            # Save trained model
            model.save("artifacts/stock_crash_model.h5")
            print("Model trained and saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainModelConfig()
    obj.train_model()
