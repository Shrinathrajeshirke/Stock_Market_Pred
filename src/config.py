import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
LOG_DIR = os.path.join(BASE_DIR, '../logs')

# API Keys (Replace with actual API keys)
YAHOO_FINANCE_API = "your_api_key"
TWITTER_API_KEY = "your_api_key"

# Model Parameters
SEQUENCE_LENGTH = 60  # Days of historical data used for prediction
LSTM_UNITS = 64
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 20

# Training Parameters
TRAIN_TEST_SPLIT = 0.8
