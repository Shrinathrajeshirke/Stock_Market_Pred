class PredictModel:
    def __init__(self, model_path="artifacts/stock_crash_model.h5"):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        from tensorflow.keras.models import load_model
        return load_model(self.model_path)

    def make_prediction(self, X_input):
        import numpy as np
        X_input = np.array(X_input).reshape(1, -1, 1)  # Ensure 3D shape for LSTM
        prediction = self.model.predict(X_input)
        return prediction

# ✅ Create an instance and call the method
if __name__ == "__main__":
    predictor = PredictModel()  # Create an instance
    sample_input = [[0.5, 0.2, -0.1, 1.0, 0.8, 0.3, -0.5, 0.7, -0.2, 0.6, 0.4]]  # Example input
    result = predictor.make_prediction(sample_input)
    print("Predicted Crash Probability:", result)
