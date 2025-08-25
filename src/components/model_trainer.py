import numpy as np
import pandas as pd
import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "RandomForest": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(),
                "SVC": SVC(),
                "DecisionTree": DecisionTreeClassifier()                
            }

            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # To get best model score and name from model_report
            best_model_name = None
            best_model_score = -1
            for name, report in model_report.items():
                # Assume report is a dict with an 'accuracy' key
                score = report['accuracy'] if isinstance(report, dict) and 'accuracy' in report else report
                if score > best_model_score:
                    best_model_score = score
                    best_model_name = name
            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException("No best model found with accuracy greater than 60%")

            logging.info(f"Best model found: {best_model_name} with accuracy: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy
        
        
        except Exception as e:
            logging.error(f"Error occurred while training model: {e}")
            raise CustomException("Error occurred while training model", sys)
