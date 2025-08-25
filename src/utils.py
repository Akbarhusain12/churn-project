import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from src.logger import logging
import dill


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error(f"Error occurred while saving object at {file_path}")
        raise CustomException(f"Error occurred while saving object at {file_path}", e)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }
        return model_report
    
    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException("Error occurred while evaluating models", e)