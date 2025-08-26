import pickle
from flask import Flask, render_template, request

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender=request.form['gender'],
            age=request.form['age'],
            monthly_charges=request.form['monthly_charges'],
            contract=request.form['contract']
        )