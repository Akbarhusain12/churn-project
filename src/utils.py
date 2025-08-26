import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
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


def evaluate_model(X_train, y_train, X_test, y_test, models, param, cv=3):
    """
    Evaluate multiple models with optional hyperparameter tuning using GridSearchCV.
    Returns:
        model_report: dict with metrics for each model
        fitted_models: dict with fitted model objects
    """
    try:
        model_report = {}
        fitted_models = {}

        for model_name, model in models.items():
            param_grid = param.get(model_name, None)

            if param_grid:  # If hyperparameters provided, use GridSearchCV
                gs = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            # Store fitted model
            fitted_models[model_name] = best_model

            # Predictions and metrics
            y_pred = best_model.predict(X_test)
            model_report[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }

            logging.info(f"Model {model_name} evaluated with accuracy: {model_report[model_name]['accuracy']}")

        return model_report, fitted_models

    except Exception as e:
        logging.error("Error occurred while evaluating models")
        raise CustomException("Error occurred while evaluating models", e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error(f"Error occurred while loading object from {file_path}")
        raise CustomException(f"Error occurred while loading object from {file_path}", e, sys)