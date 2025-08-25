# Telco Customer Churn Prediction

This project is an end-to-end machine learning solution for predicting customer churn in a telecommunications company. It follows a modular and scalable structure, from data ingestion and transformation to model training and prediction pipelines.

-----

## 🚀 Workflow

The project is structured into several key stages:

1.  **Data Ingestion**: Fetches the raw dataset, performs a train-test split, and stores the resulting CSV files in the `artifacts` directory.
2.  **Data Transformation**: Applies a preprocessing pipeline to the training data. This includes handling missing values, encoding categorical features (both nominal and ordinal), and scaling numerical features to prepare the data for modeling.
3.  **Model Training**: Trains multiple classification models (e.g., Logistic Regression, Random Forest, XGBoost) on the transformed data. It evaluates them to identify the best-performing model and saves the corresponding model object (`.pkl` file) in the `artifacts` directory.
4.  **Training Pipeline**: A master script that orchestrates the entire workflow by sequentially running the data ingestion, transformation, and model training components.
5.  **Prediction Pipeline**: A separate pipeline designed for inference. It takes new, unseen data, applies the same preprocessing steps used during training, and uses the saved model to predict churn.

-----

## 📁 Project Structure

The project follows a modular structure to ensure scalability and maintainability.

```
CHURN-PROJECT/
│
├── artifacts/           # Stores outputs like datasets and models
├── logs/                # Stores log files for debugging and monitoring
├── notebooks/           # Jupyter notebooks for experimentation
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   └── 03_modeling.ipynb
│
├── src/                 # Source code for the project
│   ├── __init__.py
│   ├── components/      # Core ML components (modular scripts)
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/        # Manages the end-to-end pipelines
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   │
│   ├── exception.py     # Custom exception handling
│   ├── logger.py        # Logging configuration
│   └── utils.py         # Utility functions (e.g., save/load objects)
│
├── .gitignore           # Files to be ignored by Git
├── README.md            # Project documentation
├── requirements.txt     # Lists all project dependencies
└── setup.py             # Makes the project installable as a package
```

-----

## 🛠️ Technologies Used

  - **Python 3.12.8**
  - **Pandas & NumPy**: For efficient data manipulation and numerical operations.
  - **Scikit-learn**: For data preprocessing, modeling, and evaluation.
  - **Matplotlib & Seaborn**: For data visualization and exploratory analysis.
  - **Dill**: For serializing and saving Python objects (like preprocessor pipelines).
  - **Jupyter Notebook**: For interactive development and EDA.
  - **Custom Logging & Exception Handling**: For robust error management and monitoring.

-----

## ⚙️ Setup and How to Run

Follow these steps to set up and run the project locally.

**1. Clone the repository:**

```bash
git clone https://github.com/Akbarhusain12/churn-project.git
cd CHURN-PROJECT
```

**2. Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

**3. Install the required dependencies:**
The `requirements.txt` file contains all the necessary packages.

```bash
pip install -r requirements.txt
```

**4. Run the complete training pipeline:**
This single command will execute data ingestion, data transformation, and model training in sequence.

```bash
python src/pipeline/train_pipeline.py
```

After successful execution, the trained model and preprocessor object will be saved in the `artifacts` folder.

**5. Run the prediction pipeline (for inference):**
To make predictions on new data, you would use the `predict_pipeline.py`. (Note: This may require setting up an interface like a simple web form or a script that accepts input data).

-----

## ✍️ Author

  - Akbarhusain