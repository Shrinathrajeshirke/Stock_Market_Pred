import os
import sys
from src.exception import CustomException
from src.logger import logging
import yfinance as yf
import pandas as pd
import requests
#from textblob import TextBlob  # Install with: pip install textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from config import DATA_DIR
from dataclasses import dataclass

from src.preprocessing import DataPreprocessing
from src.preprocessing import DataPreprocessingConfig

@dataclass
class DataIngestionConfig:
    stock_data_path: str = os.path.join("artifacts", "stock.csv")
    news_data_path: str = os.path.join("artifacts", "news.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def analyze_sentiment(self, text):
        analyzer = SentimentIntensityAnalyzer()
        sentiment_score = analyzer.polarity_scores(text)['compound']  # Score between -1 (neg) and +1 (pos)
        
        if sentiment_score > 0.05:
            return "Positive"
        elif sentiment_score < -0.05:
            return "Negative"
        else:
            return "Neutral"

    def initiate_data_ingestion(self, ticker="^GSPC"):
        logging.info("Enter the data ingestion method")
        try:
            # Fetch stock price data
            stock = yf.download(ticker, start="2010-01-01", end="2025-01-01")
            stock.reset_index(inplace=True)
            print(stock.columns)
            logging.info("Exported the dataset as a dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.stock_data_path), exist_ok=True)
            stock.to_csv(self.ingestion_config.stock_data_path, index=False, header=True)

            # Fetch news data
            url = "https://newsapi.org/v2/everything?q=stock market&apiKey=0587f5650b1b4a729f812d9a25d13382"
            response = requests.get(url)
            news_data = response.json()
            
            # Analyze sentiment
            sentiment_scores = []
            for article in news_data.get("articles", []):  # Avoid KeyError
                sentiment_scores.append({
                    "title": article.get("title", ""),  
                    "sentiment": self.analyze_sentiment(article.get("title", ""))
                })
            
            df = pd.DataFrame(sentiment_scores)
            os.makedirs(os.path.dirname(self.ingestion_config.news_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.news_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.stock_data_path,
                self.ingestion_config.news_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    stock_data, news_data = obj.initiate_data_ingestion()

    