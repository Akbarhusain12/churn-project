import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(data)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException("Error occurred while predicting", sys) from e


class CustomData:   
    def __init__(self,
                 data: pd.DataFrame):
        self.data = data

    def get_data_as_dataframe(self):
        return self.data