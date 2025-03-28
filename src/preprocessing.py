import os
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass

@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataPreprocessing:
    def __init__(self):
        self.data_preprocessing_config=DataPreprocessingConfig()

    
    def preprocess_stock_data(self):
        try:
            df = pd.read_csv("artifacts/stock.csv", index_col="Date", parse_dates=True)
            logging.info("exported the stock dataset")
            df['Close']=pd.to_numeric(df['Close'],errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            scaler = MinMaxScaler(feature_range=(0, 1))
            logging.info("Scaling the data using standard scaler method")
            df['Scaled_Close'] = scaler.fit_transform(df[['Close']])
            df.to_csv(f"data/preprocessed_stock.csv")
            print("Stock data preprocessed.")
            logging.info("preprocessing of stock data completed")
        except Exception as e:
            raise CustomException(e,sys)
            
    def preprocess_sentiment_data(self):
            try:
                logging.info("exporting the sentiment data") 
                df = pd.read_csv("artifacts/news.csv")
                df['Sentiment_Score'] = df['sentiment'].apply(lambda x: 1 if x == 'Positive' else 0)
                df.to_csv(f"data/preprocessed_sentiment.csv", index=False)
                print("Sentiment data preprocessed.")
            except Exception as e:
                raise CustomException(e,sys)
            
if __name__=="__main__":
    obj = DataPreprocessing()
    obj.preprocess_stock_data()
    obj.preprocess_sentiment_data()