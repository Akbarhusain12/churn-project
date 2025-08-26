import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(data)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException("Error occurred while predicting", sys) from e


class CustomData:   
    def __init__(self, gender, age, monthly_charges, contract):
        self.gender = gender
        self.age = int(age)
        self.monthly_charges = float(monthly_charges)
        self.contract = contract

    def get_data_as_dataframe(self):
        data = pd.DataFrame({
            # user inputs
            "Gender": [self.gender],
            "Age": [self.age],
            "Monthly Charge": [self.monthly_charges],
            "Contract": [self.contract],

            # defaults
            "Number of Dependents": [0],
            "Zip Code": [12345],
            "Latitude": [0.0],
            "Longitude": [0.0],
            "Population": [1000],
            "Number of Referrals": [0],
            "Tenure in Months": [12],
            "Avg Monthly Long Distance Charges": [20.0],
            "Avg Monthly GB Download": [10.0],
            "Total Charges": [self.monthly_charges * 12],
            "Total Refunds": [0.0],
            "Total Extra Data Charges": [0.0],
            "Total Long Distance Charges": [50.0],
            "Total Revenue": [self.monthly_charges * 12],
            "Satisfaction Score": [3],
            "CLTV": [1000],

            # categorical defaults
            "Customer ID": ["CUST0001"],
            "Under 30": ["Yes" if self.age < 30 else "No"],
            "Senior Citizen": ["Yes" if self.age >= 60 else "No"],
            "Married": ["No"],
            "Dependents": ["No"],
            "Country": ["USA"],
            "State": ["CA"],
            "City": ["San Francisco"],
            "Quarter": ["Q1"],
            "Referred a Friend": ["No"],
            "Offer": ["None"],
            "Phone Service": ["Yes"],
            "Multiple Lines": ["No"],
            "Internet Service": ["Yes"],
            "Internet Type": ["Fiber"],
            "Online Security": ["No"],
            "Online Backup": ["No"],
            "Device Protection Plan": ["No"],
            "Premium Tech Support": ["No"],
            "Streaming TV": ["Yes"],
            "Streaming Movies": ["Yes"],
            "Streaming Music": ["Yes"],
            "Unlimited Data": ["Yes"],
            "Paperless Billing": ["Yes"],
            "Payment Method": ["Credit Card"]
        })
        return data