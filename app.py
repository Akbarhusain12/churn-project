import pickle
from flask import Flask, render_template, request

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline   


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('result.html')
    else:
        # Wrap user inputs with CustomData
        data = CustomData(
            gender=request.form['gender'],
            age=request.form['age'],
            monthly_charges=request.form['monthly_charges'],
            contract=request.form['contract']
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df) 

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(data=pred_df)

        return render_template('result.html', prediction=result[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)