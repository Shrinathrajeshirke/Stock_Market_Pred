import os
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from dataclasses import dataclass
import pandas as pd

@dataclass
class featureEngineeringConfig:
    FE_obj_file_path=os.path.join('artifacts','fe.pkl')

class FeatureEngineering:
    def __init__(self):
        self.feature_engineering_config=featureEngineeringConfig()

    def create_crash_label(self, df, threshold=0.05):
        """
        Create a 'Crash' column that labels a market crash 
        if the stock price drops by more than a certain threshold.

        Args:
        - df (pd.DataFrame): Stock data with 'Close' price.
        - threshold (float): Percentage drop to define a crash (default: 5%).

        Returns:
        - df (pd.DataFrame): Updated dataframe with 'Crash' column.
        """
        df['Crash'] = (df['Scaled_Close'].pct_change() < -threshold).astype(int)
        return df
    
    def create_features(self):
        try:
            logging.info("Importing preprocessed data...")
            stock_data = pd.read_csv("data/preprocessed_stock.csv", index_col="Date", parse_dates=True)
            news_data = pd.read_csv("data/preprocessed_sentiment.csv")

            # Feature engineering: Rolling statistics
            stock_data['Rolling_Mean'] = stock_data['Scaled_Close'].rolling(window=10).mean()
            stock_data['Volatility'] = stock_data['Scaled_Close'].rolling(window=10).std()

            # Add 'Crash' label
            stock_data = self.create_crash_label(stock_data)

            # Merge stock and sentiment data
            merged_df = pd.concat([stock_data, news_data], axis=1)

            # Save the processed features
            merged_df.to_csv("data/features.csv", index=False)
            print("Feature engineering completed. 'Crash' column added successfully.")
        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = FeatureEngineering()
    obj.create_features()
