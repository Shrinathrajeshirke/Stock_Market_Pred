import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras_tuner import RandomSearch
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class TrainModelConfig:
    train_obj_file_path = os.path.join('artifacts', 'train.pkl')

    def load_data(self):
        try:
            df = pd.read_csv("data/features.csv")
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

            if 'Crash' not in df.columns:
                raise ValueError("Missing 'Crash' column in dataset!")

            X = df.drop(columns=['Crash']).values
            y = df['Crash'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

    def build_model(self, hp, input_shape):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', 32, 256, step=32), return_sequences=True, input_shape=input_shape))
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(LSTM(units=hp.Int('units_2', 32, 256, step=32)))
        model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        try:
            X_train, X_test, y_train, y_test = self.load_data()

            def model_builder(hp):
                return self.build_model(hp, (X_train.shape[1], X_train.shape[2]))

            tuner = RandomSearch(
                model_builder,
                objective='val_accuracy',
                max_trials=5,
                executions_per_trial=1,
                directory='hyperparameter_tuning',
                project_name='lstm_stock_crash'
            )

            tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

            best_hps = tuner.get_best_hyperparameters(1)[0]
            print("✅ Best hyperparameters found:", best_hps.values)

            model = tuner.hypermodel.build(best_hps)
            model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

            # Predict and calculate accuracy
            y_pred = model.predict(X_test)
            y_pred = (y_pred > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ Model Accuracy on Test Data: {acc:.4f}")

            model.save("artifacts/stock_crash_model.h5")
            print("✅ Model saved at artifacts/stock_crash_model.h5")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = TrainModelConfig()
    obj.train_model()
