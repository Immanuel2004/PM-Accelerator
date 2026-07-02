# ML-Classification

ML-Classification is an end-to-end machine learning project for **classification tasks**. It provides a complete pipeline from data ingestion to model deployment, following a modular and production-ready project structure. The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training with hyperparameter tuning, experiment tracking using MLflow and DagsHub, and prediction pipelines.

## Features

* **Data Ingestion:** Load datasets from local files or remote sources.
* **Data Preprocessing:** Handle missing values, encode categorical features, scale numerical features, and clean data.
* **Exploratory Data Analysis (EDA):** Analyze feature distributions, correlations, class balance, and detect outliers.
* **Feature Engineering & Transformation:** Build preprocessing pipelines and transform features for optimal model performance.
* **Model Training:** Train and compare multiple classification algorithms with optional hyperparameter tuning using GridSearchCV.
* **Model Evaluation:** Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC (for binary classification).
* **Experiment Tracking:** Track experiments, parameters, metrics, and trained models using MLflow integrated with DagsHub.
* **Prediction Pipeline:** Perform end-to-end predictions on unseen data using the trained model.
* **Model Persistence:** Save the best-performing model for inference.
* **Logging & Exception Handling:** Comprehensive logging and custom exception handling for easier debugging and maintenance.

## Project Structure

```text
Data Science/
│
├── artifacts/
├── notebooks/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── setup.py
└── README.md
```

## Models Included

The project supports multiple machine learning classification algorithms, including:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Extra Trees Classifier
* Gradient Boosting Classifier
* AdaBoost Classifier
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* XGBoost Classifier
* CatBoost Classifier

## Evaluation Metrics

The trained models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC (Binary Classification)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Immanuel2004/ML-Classification.git
cd ML-Classification
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* CatBoost
* MLflow
* DagsHub
* Joblib

## Future Improvements

* FastAPI deployment
* Docker containerization
* CI/CD with GitHub Actions
* Model monitoring
* Automated retraining pipeline
* Cloud deployment (AWS/Azure/GCP)
