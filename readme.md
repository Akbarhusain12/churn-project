# 📌 Telco Customer Churn Prediction

An end-to-end machine learning project to predict customer churn with a modular pipeline and a Flask web application for real-time inference.

-----

## 🚀 Key Features

  - **Data Ingestion**: Loads the raw dataset, performs a train-test split, and saves the processed files.
  - **Data Transformation**: Handles missing values, encodes categorical variables, and scales numerical features using a preprocessing pipeline.
  - **Model Training**: Trains multiple classifiers (Logistic Regression, Random Forest, XGBoost) and saves the best-performing model as a `.pkl` file.
  - **Prediction Pipeline**: Encapsulates the preprocessor and model for seamless inference on new, unseen data.
  - **Web Application**: A user-friendly Flask app with a simple UI for making real-time churn predictions.

-----

## 📂 Project Structure

The project follows a modular structure to ensure scalability and maintainability.

```
CHURN-PROJECT/
│
├── artifacts/              # Stores outputs like datasets and models
├── logs/                   # Stores log files for debugging
├── notebooks/              # Jupyter notebook for experimentation
│   ├── data/
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
```

-----

## 🛠️ Technologies Used

  - **Programming Language**: `Python 3.12`
  - **Web Framework**: `Flask`
  - **Data Manipulation**: `Pandas`, `NumPy`
  - **Machine Learning**: `Scikit-learn`
  - **Data Visualization**: `Matplotlib`, `Seaborn`
  - **Serialization**: `Dill`
  - **Experimentation**: `Jupyter Notebook`

-----

## ⚙️ Setup and How to Run

Follow these steps to set up and run the project on your local machine.

### 1\. Clone the Repository

```bash
git clone https://github.com/Akbarhusain12/churn-project.git
cd churn-project
```

### 2\. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3\. Install Dependencies

Install all the required packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4\. Run the Training Pipeline

Execute the following scripts in order to process the data and train the model. The resulting artifacts (datasets, preprocessor, and model) will be saved in the `artifacts/` directory.

```bash
python src/components/data_ingestion.py
python src/components/data_transformation.py
python src/components/model_trainer.py
```

### 5\. Run the Flask Application

Start the web server to interact with the prediction UI.

```bash
python app.py
```

Open your web browser and navigate to: 👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

-----

## ✨ Future Improvements

  - [ ] Add functionality for bulk prediction via CSV upload in the Flask app.
  - [ ] Integrate SHAP or LIME for model explainability (XAI).
  - [ ] Deploy the application using Docker and a cloud service (e.g., AWS, GCP, Heroku).
  - [ ] Implement experiment tracking with tools like MLflow or DVC.

-----

## ✍️ Author

**Akbar Husain**

  
 - **LinkedIn**: [Akbar Husain](https://www.linkedin.com/in/akbar-kadiwala/)

  
