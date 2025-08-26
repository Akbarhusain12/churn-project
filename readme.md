# 📌 Telco Customer Churn Prediction  

This project is an **end-to-end machine learning solution** for predicting customer churn in a telecommunications company. It features a **modular pipeline** for data processing and model training, and includes a **Flask web application** for real-time inference.  

---

## 🚀 Key Features  
- **Data Ingestion** → Load raw dataset, perform train-test split, save processed files.  
- **Data Transformation** → Handle missing values, encode categorical variables, scale numerical features.  
- **Model Training** → Train multiple classifiers (Logistic Regression, Random Forest, XGBoost), save best model (`.pkl`).  
- **Prediction Pipeline** → Load preprocessor + model for inference on new data.  
- **Web Application** → Flask app with a simple UI for customer churn prediction.  

---

## 📂 Project Structure  

CHURN-PROJECT/
│
├── artifacts/ # Stores outputs like datasets and models
├── logs/ # Stores log files for debugging
├── notebooks/ # Jupyter notebook for experimentation
│ ├── data/ # Contains the raw dataset
│ │ └── telco.csv
│ └── EDA_Churn_prediction.ipynb
│
├── src/ # Source code for the project
│ ├── init.py
│ ├── components/ # Core ML components (modular scripts)
│ │ ├── init.py
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipeline/ # Manages the prediction pipeline
│ │ ├── init.py
│ │ └── predict_pipeline.py
│ │
│ ├── exception.py # Custom exception handling
│ ├── logger.py # Logging configuration
│ └── utils.py # Utility functions
│
├── templates/ # HTML templates for the Flask app
│ ├── index.html
│ └── result.html
│
├── app.py # Main Flask application file
├── .gitignore # Files to be ignored by Git
├── README.md # Project documentation
├── requirements.txt # Project dependencies
└── setup.py # Makes the project installable as a package

yaml
Copy
Edit

---

## 🛠️ Technologies Used  
- **Python 3.12.8**  
- **Flask** → Web application framework  
- **Pandas & NumPy** → Data manipulation  
- **Scikit-learn** → Preprocessing, modeling, evaluation  
- **Matplotlib & Seaborn** → Data visualization  
- **Dill** → Serialize model objects  
- **Jupyter Notebook** → EDA and prototyping  

---

## ⚙️ Setup and How to Run  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/Akbarhusain12/churn-project.git
cd churn-project
2️⃣ Create a virtual environment (recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the training pipeline
Execute the following scripts in order to process data and train the model:

bash
Copy
Edit
python src/components/data_ingestion.py
python src/components/data_transformation.py
python src/components/model_trainer.py
Artifacts (datasets, models) will be saved in the artifacts/ directory.

5️⃣ Run the Flask application
bash
Copy
Edit
python app.py
Then open in browser: 👉 http://127.0.0.1:5000

✨ Future Improvements
Add bulk prediction (CSV upload) in the Flask app.

Integrate SHAP/LIME for explainable AI.

Deploy using Docker + Cloud (Heroku/AWS/GCP).

Track experiments with MLflow or DVC.

✍️ Author
Akbar Husain
📧 [Your Email]
🔗 [Your LinkedIn/GitHub]