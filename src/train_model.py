import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras_tuner import RandomSearch
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class TrainModelConfig:
    train_obj_file_path = os.path.join('artifacts', 'train.pkl')

    def load_data(self):
        """Loads the preprocessed data and ensures it's in the correct format"""
        try:
            df = pd.read_csv("data/features.csv")

            # Ensure all data is numeric
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            if 'Crash' not in df.columns:
                raise ValueError("Error: 'Crash' column not found in dataset!")

            # Separate features and target
            X = df.drop(columns=['Crash']).values
            y = df['Crash'].values

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Reshape for LSTM input (samples, timesteps, features)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def build_model(self, hp, input_shape):
        """Defines an LSTM model with tunable hyperparameters"""
        model = Sequential()
        model.add(LSTM(
            units=hp.Int('units', min_value=32, max_value=256, step=32),
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(LSTM(hp.Int('units_2', min_value=32, max_value=256, step=32)))
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=["accuracy"]
        )
        return model

    def train_model(self):
        """Trains the LSTM model with hyperparameter tuning"""
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            # ✅ Add batch_size to hyperparameter search space
            def model_builder(hp):
                model = self.build_model(hp, (X_train.shape[1], X_train.shape[2]))

                # ✅ Add batch_size explicitly
                batch_size = hp.Choice('batch_size', values=[16, 32, 64])

                return model

            # Define hyperparameter tuner
            tuner = RandomSearch(
                model_builder,
                objective='val_accuracy',
                max_trials=10,
                executions_per_trial=1,
                directory='hyperparameter_tuning',
                project_name='stock_crash_lstm'
            )

            # Run hyperparameter tuning
            tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

            # Get best hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            # ✅ Fix: Ensure 'batch_size' exists
            if 'batch_size' in best_hps.values:
                batch_size = best_hps.get('batch_size')
            else:
                batch_size = 32  # Default value

            print(f"✅ Best Batch Size: {batch_size}")

            # Build and train the best model
            best_model = tuner.hypermodel.build(best_hps)
            best_model.fit(
                X_train, y_train, 
                epochs=20, 
                batch_size=batch_size,  # ✅ Now batch_size is correctly retrieved
                validation_data=(X_test, y_test)
            )

            # Save the trained model
            best_model.save("artifacts/stock_crash_model.h5")
            print("best model is ", best_model)
            print("✅ Best model trained and saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainModelConfig()
    obj.train_model()
