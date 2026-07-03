# Weather Trend Forecasting using Machine Learning

> **PM Accelerator – Technical Assessment**
> **Author:** Immanuel G

---

## Project Overview

This project analyzes historical weather observations from around the world to identify weather patterns and build machine learning models capable of forecasting temperature. Multiple regression models were trained and evaluated to determine the most accurate forecasting approach.

The project demonstrates the complete Data Science lifecycle:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning Model Development
* Model Evaluation
* Ensemble Learning
* Anomaly Detection
* Feature Importance Analysis
* Climate & Environmental Insights

---

## Dataset

**Dataset:** Global Weather Repository

**Source:** Kaggle

The dataset contains over **150,000 weather observations** collected from cities worldwide with more than **40 weather-related features**, including:

* Temperature
* Humidityp
* Pressure
* Wind Speed
* Visibility
* UV Index
* Air Quality Metrics
* Geographic Coordinates
* Sunrise/Sunset Information
* Weather Conditions

---

## Repository Structure

```text
DataScience/
│
├── notebooks/
│   ├── Experiments.ipynb
│   └── __init__.py
│
├── main.py
├── README.md
├── .gitignore
├── requirements.txt
└── LICENSE
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Scikit-learn
* XGBoost
* SHAP
* Jupyter Notebook

---

## Data Preprocessing

The preprocessing pipeline included:

* Removing duplicate records
* Handling missing values
* Converting timestamps into datetime format
* Extracting time-based features:
  * Year
  * Month
  * Day
  * Hour
  * Day of Week
  * Week of Year
  * Season
* Encoding categorical variables
* Scaling numerical features
* Preparing the dataset for machine learning

---

## Exploratory Data Analysis

The EDA focused on understanding weather patterns and feature relationships through:

* Statistical summaries
* Missing value analysis
* Correlation heatmaps
* Temperature distributions
* Humidity and precipitation analysis
* Air quality analysis
* Geographic weather visualization
* Monthly and yearly climate trends

---

## Machine Learning Models

The following models were implemented and compared:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Ensemble Model (Random Forest + XGBoost)

---

## Model Evaluation

The models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### Performance Comparison

| Model             |      MAE |     RMSE |        R² |
| ----------------- | -------: | -------: | --------: |
| **XGBoost**       | **2.20** | **3.07** | **0.898** |
| Ensemble          |     2.22 |     3.11 |     0.896 |
| Random Forest     |     2.51 |     3.54 |     0.865 |
| Linear Regression |     5.67 |     7.05 |     0.464 |

### Best Performing Model

**XGBoost** achieved the highest performance, explaining approximately **90% of the variance** in temperature while producing the lowest prediction error.

---

## Advanced Analysis

### Feature Importance

Feature importance analysis was performed using XGBoost to identify the most influential variables affecting temperature prediction.

### Anomaly Detection

Isolation Forest was used to detect unusual weather observations.

Results:

* Normal observations: **147,837**
* Anomalies detected: **3,018**

These anomalies may represent extreme weather conditions or rare environmental events.

### Climate Analysis

Climate trends were analyzed by examining:

* Monthly average temperatures
* Seasonal variations
* Yearly weather trends

### Environmental Analysis

Air quality metrics such as PM2.5, PM10, Carbon Monoxide, Ozone, and Nitrogen Dioxide were analyzed to understand their relationships with weather variables.

---

## Key Insights

* Weather variables exhibit strong nonlinear relationships with temperature.
* Tree-based ensemble models significantly outperform traditional linear regression.
* XGBoost provided the highest forecasting accuracy.
* Ensemble learning produced competitive performance but did not surpass XGBoost.
* Isolation Forest successfully identified approximately **2%** of observations as anomalies.
* Air quality indicators showed meaningful relationships with temperature and humidity.
* Geographic and seasonal factors strongly influence weather patterns.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Immanuel2004/PM-Accelerator.git
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/Experiments.ipynb
```

Run all notebook cells sequentially to reproduce the analyses and model results.

---

## Results

* Successfully analyzed more than **150,000** global weather observations.
* Built and compared four forecasting models.
* Achieved **R² = 0.898** using XGBoost.
* Detected anomalous weather observations using Isolation Forest.
* Generated visualizations to understand global weather and climate trends.

---

## Future Improvements

* Hyperparameter optimization using Optuna.
* Incorporate deep learning models such as LSTM.
* Integrate live weather APIs for real-time forecasting.
* Develop an interactive Streamlit dashboard.
* Expand forecasting to individual cities and regional climate analysis.

---

## Demo Video

Add your project demo video link here after uploading.

Example:

```
https://youtu.be/your-demo-video
```

---

## Author

**Immanuel G**

Data Science | Machine Learning | Generative AI | Agentic AI

**GitHub**

[github.com/Immanuel2004](https://in.linkedin.com/in/immanuelg01?utm_source=chatgpt.com)

**LinkedIn**

[linkedin.com/in/immanuelg01](https://in.linkedin.com/in/immanuelg01?utm_source=chatgpt.com)

---

## License

This repository was developed as part of the **PM Accelerator Technical Assessment** and is intended for educational and portfolio purposes.
