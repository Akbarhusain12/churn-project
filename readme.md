# Telco Customer Churn Prediction

This project aims to predict customer churn for a telecom company using machine learning. The workflow includes data ingestion, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## 📁 Project Structure

```
churn-project/
│
├── data/                # Raw and processed data
│
├── notebooks/
│   └── 01_eda.ipynb     # Exploratory Data Analysis notebook
│
├── src/
│   ├── components/
│   │   └── data_ingestion.py   # Data ingestion pipeline
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging utility
│   └── utils.py                # Utility functions (save/load objects, model evaluation)
│
├── artifacts/           # Model artifacts (ignored by git)
│
├── .gitignore
└── README.md
```

---

## 🚀 Features & Workflow

- **Data Ingestion:**  
  Loads raw data, splits into train/test sets, and saves them in the `artifacts/` folder.

- **EDA & Visualization:**  
  Performed in Jupyter Notebook (`notebooks/01_eda.ipynb`) using pandas, matplotlib, and seaborn.

- **Preprocessing & Feature Engineering:**  
  - Handling missing values  
  - Dropping redundant columns  
  - Encoding categorical variables (Label, Ordinal, One-Hot)  
  - Feature selection and engineering

- **Modeling:**  
  - Trains multiple models: Logistic Regression, Random Forest, XGBoost, etc.
  - Hyperparameter tuning with `GridSearchCV`
  - Model evaluation using accuracy, confusion matrix, and classification report

- **Utilities:**  
  - `src/utils.py` provides functions to save Python objects (using `dill`) and evaluate models.
  - Custom exception and logging for robust error handling.

---

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas, numpy** – Data manipulation
- **matplotlib, seaborn** – Visualization
- **scikit-learn** – Machine learning models, preprocessing, evaluation, and hyperparameter tuning
- **xgboost** – Gradient boosting model
- **dill** – Object serialization
- **Jupyter Notebook** – EDA and prototyping
- **Custom Logging & Exception Handling**

---

## ⚙️ How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Akbarhusain12/churn-project.git
    cd churn-project
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run data ingestion:**
    ```bash
    python -m src.components.data_ingestion
    ```

4. **Explore and run EDA:**
    - Open `notebooks/01_eda.ipynb` in Jupyter Notebook or VS Code.

---

## 📦 Notes

- The `artifacts/` folder is ignored by git (see `.gitignore`).
- All logs and errors are handled using custom logger and exception modules.
- Update paths as needed for your environment.

---

## ✍️ Author

- Akbarhusain

---