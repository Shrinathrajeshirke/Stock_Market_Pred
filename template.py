import os  ## to create libraries
from pathlib import Path ## handles the / while creating directories in linux/windows
import logging ## to log all info

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "textSummarizer"

list_of_files = [
    "models/__init__.py",
    "logs/__init__py",
    "data/__init__.py",
    "noteboook/__init__.py",
    "deployment/__init__.py"
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
    "src/config.py",
    "src/data_collection.py",
    "src/preprocessing.py",
    "src/feature_engineering.py",
    "src/train_model.py",
    "src/evaluate_model.py",
    "src/predict.py",
    "templates/main.txt",
    "README.md",
    "requirements.txt",
    "setup.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f"creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")


