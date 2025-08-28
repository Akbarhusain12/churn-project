import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging  


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, data: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info(f"Expected features: {preprocessor.feature_names_in_.tolist()}")
            logging.info(f"Received features: {data.columns.tolist()}")

            # Check differences
            missing = set(preprocessor.feature_names_in_) - set(data.columns)
            extra = set(data.columns) - set(preprocessor.feature_names_in_)
            logging.info(f"Missing features: {missing}")
            logging.info(f"Extra features: {extra}")
            logging.info(f"Data dtypes: {data.dtypes}")

            data_scaled = preprocessor.transform(data)  # line 20
            predictions = model.predict(data_scaled)

            return predictions
        except Exception as e:
            raise CustomException("Error occurred while predicting", sys) from e



class CustomData:   
    def __init__(self,
            gender,
            age,
            under_30,
            senior_citizen,
            married,
            dependents,
            num_dependents,
            country,
            state,
            city,
            zip_code,
            latitude,
            longitude,
            population,
            quarter,
            referred_friend,
            num_referrals,
            tenure_months,
            offer,
            phone_service,
            multiple_lines,
            internet_service,
            internet_type,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            streaming_music,
            unlimited_data,
            avg_long_distance,
            avg_gb_download,
            monthly_charges,
            total_charges,
            total_refunds,
            extra_data_charges,
            long_distance_charges,
            total_revenue,
            satisfaction_score,
            cltv,
            contract,
            paperless_billing,
            payment_method):
        
        # categorical
        self.gender = gender
        self.under_30 = under_30
        self.senior_citizen = senior_citizen
        self.married = married
        self.dependents = dependents
        self.country = country
        self.state = state
        self.city = city
        self.quarter = quarter
        self.referred_friend = referred_friend
        self.offer = offer
        self.phone_service = phone_service
        self.multiple_lines = multiple_lines
        self.internet_service = internet_service
        self.internet_type = internet_type
        self.online_security = online_security
        self.online_backup = online_backup
        self.device_protection = device_protection
        self.tech_support = tech_support
        self.streaming_tv = streaming_tv
        self.streaming_movies = streaming_movies
        self.streaming_music = streaming_music
        self.unlimited_data = unlimited_data
        self.contract = contract
        self.paperless_billing = paperless_billing
        self.payment_method = payment_method

        # numeric → explicitly cast to int/float
        self.age = int(age)
        self.num_dependents = int(num_dependents) if num_dependents else 0
        self.zip_code = int(zip_code) if zip_code else 0
        self.latitude = float(latitude) if latitude else 0.0
        self.longitude = float(longitude) if longitude else 0.0
        self.population = int(population) if population else 0
        self.num_referrals = int(num_referrals) if num_referrals else 0
        self.tenure_months = int(tenure_months) if tenure_months else 0
        self.avg_long_distance = float(avg_long_distance) if avg_long_distance else 0.0
        self.avg_gb_download = float(avg_gb_download) if avg_gb_download else 0.0
        self.monthly_charges = float(monthly_charges)
        self.total_charges = float(total_charges) if total_charges else self.monthly_charges * 12
        self.total_refunds = float(total_refunds) if total_refunds else 0.0
        self.extra_data_charges = float(extra_data_charges) if extra_data_charges else 0.0
        self.long_distance_charges = float(long_distance_charges) if long_distance_charges else 0.0
        self.total_revenue = float(total_revenue) if total_revenue else self.monthly_charges * 12
        self.satisfaction_score = int(satisfaction_score) if satisfaction_score else 3
        self.cltv = float(cltv) if cltv else 1000.0


    def get_data_as_dataframe(self):
        data = pd.DataFrame({
            "Customer ID": ["CUST0001"],
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