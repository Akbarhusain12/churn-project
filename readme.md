Telco Customer Churn Prediction
This project is an end-to-end machine learning solution for predicting customer churn in a telecommunications company. It features a modular pipeline for data processing and model training, and includes a Flask web application for real-time inference.

🚀 Key Features & Workflow
The project is structured into several key stages:

Data Ingestion: Fetches the raw dataset, performs a train-test split, and stores the resulting CSV files in the artifacts directory.

Data Transformation: Applies a preprocessing pipeline to the training data. This includes handling missing values, encoding categorical features, and scaling numerical features.

Model Training: Trains multiple classification models (e.g., Logistic Regression, Random Forest, XGBoost) and saves the best-performing model object (.pkl file) in the artifacts directory.

Prediction Pipeline: A separate pipeline designed for inference. It loads the saved preprocessor and model to make predictions on new data.

Web Application (Flask): A user-friendly interface built with Flask that takes customer data as input and displays the churn prediction result in real-time.

📁 Project Structure
The project follows a modular structure to ensure scalability and maintainability.

CHURN-PROJECT/
│
├── artifacts/              # Stores outputs like datasets and models
├── logs/                   # Stores log files for debugging
├── notebooks/              # Jupyter notebook for experimentation
│   ├── data/               # Contains the raw dataset
│   │   └── telco.csv
│   └── EDA_Churn_prediction.ipynb
│
├── src/                    # Source code for the project
│   ├── __init__.py
│   ├── components/         # Core ML components (modular scripts)
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/           # Manages the prediction pipeline
│   │   ├── __init__.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py        # Custom exception handling
│   ├── logger.py           # Logging configuration
│   └── utils.py            # Utility functions
│
├── templates/              # HTML templates for the Flask app
│   ├── index.html
│   └── result.html
│
├── app.py                  # Main Flask application file
├── .gitignore              # Files to be ignored by Git
├── README.md               # Project documentation
├── requirements.txt        # Project dependencies
└── setup.py                # Makes the project installable as a package

🛠️ Technologies Used
Python 3.12.8

Flask: For the web application framework.

Pandas & NumPy: For efficient data manipulation.

Scikit-learn: For data preprocessing, modeling, and evaluation.

Matplotlib & Seaborn: For data visualization.

Dill: For serializing Python objects.

Jupyter Notebook: For interactive development and EDA.

⚙️ Setup and How to Run
Follow these steps to set up and run the project locally.

1. Clone the repository:

git clone https://github.com/Akbarhusain12/churn-project.git
cd churn-project

2. Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the data and training pipeline:
Execute the following components in order to process the data and train the model. The resulting artifacts will be saved in the artifacts folder.

python src/components/data_ingestion.py
python src/components/data_transformation.py
python src/components/model_trainer.py

5. Run the Flask application:
Start the web server to use the prediction interface.

python app.py

Navigate to http://127.0.0.1:5000 in your web browser to access the application.

✍️ Author
Akbarhusain