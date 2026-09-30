# Cars

## 1. Project Overview

This project implements an AI-based used car price prediction system using machine learning.

The system analyzes used car information such as vehicle characteristics, usage details, fuel type, brand, model, year, and location to estimate the market price of a used car.

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model training
- Model evaluation
- Price prediction
- REST API-based prediction
- FastAPI
- Interactive Swagger API documentation
- Streamlit application
- Automated API testing
- Trained model saving and loading

---

## 2. Problem Statement

The used car market contains a large number of vehicles with different prices based on factors such as manufacturing year, kilometers driven, brand, fuel type, model, and location.

Determining a suitable market price manually can be difficult because multiple factors influence the resale value of a vehicle.

The objective of this project is to develop a machine learning system that can analyze used car information and predict an estimated market price.

The system provides a complete machine learning pipeline from data preprocessing and model training to real-time prediction through FastAPI and Streamlit.

---

## 3. Dataset

### Indian Used Car Dataset

The project uses an Indian used-car dataset containing information about vehicles available in the used-car market.

The dataset contains vehicle and market-related information that can be used to predict used car prices.

The dataset includes information such as:

- Car brand
- Car model
- Manufacturing year
- Kilometers driven
- Fuel type
- Transmission information
- City/location
- Other vehicle-related attributes
- Selling/resale price

The raw dataset is stored in:

```text
DATA/RAW/cars.csv
```

The dataset was obtained from Kaggle.

Dataset source:

https://www.kaggle.com/sanjeetsinghnaik/used-car-information

---

## 4. Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

### API

- FastAPI
- Uvicorn
- Pydantic

### Application

- Streamlit

### Testing

- Pytest
- HTTPX
- FastAPI TestClient

### Development Environment

- Jupyter Notebook
- Anaconda

### Model Storage

- Joblib

---

## 5. Machine Learning Workflow

The overall workflow is:

```text
Raw Used Car Data
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Data Preprocessing
        ↓
Train/Test Split
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Best Model Selection
        ↓
Model Saving
        ↓
Price Prediction
        ↓
FastAPI / Streamlit
```

---

## 6. Project Structure

```text
Cars/
│
├── API/
│   └── main.py
│
├── APP/
│   └── streamlit_app.py
│
├── DATA/
│   ├── RAW/
│   │   └── cars.csv
│   │
│   └── PROCESSED/
│       ├── cars_clean.csv
│       └── cars_features.csv
│
├── MODELS/
│   ├── car_price_model.joblib
│   ├── gradient_boosting.joblib
│   ├── linear_regression.joblib
│   └── random_forest.joblib
│
├── NOTEBOOKS/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_prediction_testing.ipynb
│
├── SCREENSHOTS/
│
├── SRC/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── TESTS/
│   └── test_api.py
│
├── requirements.txt
├── README.md
├── Dockerfile
└── .gitignore
```

---

## 7. Data Preprocessing

The data preprocessing pipeline prepares the raw used-car dataset for machine learning.

The preprocessing process includes:

- Loading the raw dataset
- Standardizing column names
- Cleaning string values
- Standardizing missing-value representations
- Removing duplicate records
- Removing invalid values
- Handling missing numerical values
- Handling missing categorical values
- Saving the cleaned dataset

The cleaned dataset is stored in:

```text
DATA/PROCESSED/cars_clean.csv
```

The preprocessing implementation is available in:

```text
SRC/preprocessing.py
```

The preprocessing workflow is also demonstrated in:

```text
NOTEBOOKS/01_data_cleaning.ipynb
```

---

## 8. Exploratory Data Analysis

Exploratory Data Analysis (EDA) is performed to understand the structure and characteristics of the used-car dataset.

The analysis includes:

- Dataset shape
- Dataset information
- Statistical summary
- Missing-value analysis
- Duplicate-value analysis
- Price distribution
- Brand analysis
- Fuel-type analysis
- Year analysis
- Kilometers-driven analysis
- Relationship between vehicle attributes and price
- Correlation analysis
- Visualizations

The EDA workflow is available in:

```text
NOTEBOOKS/02_eda.ipynb
```

---

## 9. Feature Engineering

Feature engineering is performed to create useful features for the machine learning models.

The feature engineering process includes:

- Creating vehicle age from the manufacturing year
- Converting numerical features into appropriate numerical formats
- Preparing categorical features
- Handling missing feature values
- Preparing the final feature dataset
- Removing target-derived information from the prediction features

The engineered dataset is stored in:

```text
DATA/PROCESSED/cars_features.csv
```

The feature engineering implementation is available in:

```text
SRC/feature_engineering.py
```

The notebook implementation is available in:

```text
NOTEBOOKS/03_feature_engineering.ipynb
```

---

## 10. Model Training

The project trains multiple machine learning regression models for used-car price prediction.

The models include:

### Linear Regression

Linear Regression is used as a baseline regression model for predicting the continuous car price target.

### Random Forest Regressor

Random Forest is an ensemble learning algorithm that combines multiple decision trees to model complex relationships between vehicle features and price.

### Gradient Boosting Regressor

Gradient Boosting builds an ensemble of models sequentially to improve prediction performance.

The training process includes:

1. Loading the feature dataset
2. Identifying the target price column
3. Separating input features and target
4. Splitting the data into training and testing sets
5. Identifying numerical and categorical features
6. Applying preprocessing
7. Training multiple regression models
8. Evaluating the models
9. Selecting the best-performing model
10. Saving the trained models

The training implementation is available in:

```text
SRC/train.py
```

The training notebook is available in:

```text
NOTEBOOKS/04_model_training.ipynb
```

---

## 11. Model Evaluation

The trained regression models are evaluated using standard regression metrics.

The evaluation includes:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

The evaluation process compares the actual car prices with the predicted prices.

Additional evaluation visualizations include:

- Actual vs predicted prices
- Prediction errors
- Error distribution

The evaluation implementation is available in:

```text
SRC/evaluate.py
```

The evaluation notebook is available in:

```text
NOTEBOOKS/05_model_evaluation.ipynb
```

---

## 12. Model Saving

After model training, the trained models are saved using Joblib.

The main production model is stored as:

```text
MODELS/car_price_model.joblib
```

Additional trained models are stored in:

```text
MODELS/gradient_boosting.joblib
MODELS/linear_regression.joblib
MODELS/random_forest.joblib
```

These saved models can be loaded later without retraining the machine learning models.

The prediction system loads the trained production model when making new predictions.

---

## 13. Prediction System

The prediction system accepts used-car information and returns an estimated market price.

The prediction workflow is:

```text
Car Information
       ↓
Input Validation
       ↓
Feature Preparation
       ↓
Saved ML Model
       ↓
Price Prediction
       ↓
Predicted Price
```

The prediction functionality is implemented in:

```text
SRC/predict.py
```

The `CarPricePredictor` class is responsible for:

- Loading the trained model
- Loading the feature information
- Preparing input data
- Passing the input to the trained model
- Returning the predicted price

---

## 14. FastAPI

A REST API is provided using FastAPI for real-time used-car price prediction.

The API implementation is located in:

```text
API/main.py
```

The API provides the following endpoints:

### Root Endpoint

```text
GET /
```

Provides basic information about the API.

### Health Endpoint

```text
GET /health
```

Checks whether the API is running and whether the trained machine learning model has been loaded successfully.

### Prediction Endpoint

```text
POST /predict
```

Accepts car features and returns the predicted market price.

---

## 15. Swagger API Documentation

FastAPI automatically provides interactive API documentation.

After starting the API, Swagger documentation can be accessed at:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface allows users to:

- View available API endpoints
- Understand API request formats
- Test the health endpoint
- Enter car features
- Test the prediction endpoint
- View prediction responses
- Check API validation errors

---

## 16. Streamlit Application

A Streamlit application provides a user-friendly interface for Cars.

The application is implemented in:

```text
APP/streamlit_app.py
```

The application contains the following sections:

### Dashboard

Displays:

- Total number of cars
- Average price
- Median price
- Number of features
- Dataset preview
- Price information

### Explore Cars

Allows users to explore the dataset using filters such as:

- City
- Brand
- Fuel type
- Other categorical attributes
- Price range

Cities displayed in the application, such as Mumbai, Thane, Hyderabad, Bengaluru, and others, come directly from the dataset.

### Compare Cars

Allows users to select and compare available cars using the information contained in the dataset.

### Price Prediction

Allows users to enter vehicle information and obtain an estimated used-car market price through the FastAPI prediction service.

---

## 17. API and Streamlit Architecture

The application architecture is:

```text
                    User
                     │
                     ▼
               Streamlit App
                     │
                  HTTP POST
                     │
                     ▼
               FastAPI Server
                     │
                     ▼
              Prediction Module
                     │
                     ▼
             Trained ML Model
                     │
                     ▼
             Predicted Car Price
                     │
                     ▼
               Streamlit Result
```

The Streamlit application communicates with FastAPI through:

```text
http://127.0.0.1:8000
```

---

## 18. API Testing

Automated API tests are implemented using Pytest.

The test file is:

```text
TESTS/test_api.py
```

The tests cover:

- Root endpoint
- Health endpoint
- Prediction endpoint
- Missing feature validation
- Invalid endpoint handling

The tests can be executed using:

```bash
pytest -v
```

---

## 19. Installation

Clone the repository:

```bash
git clone https://github.com/ramcodes123-debug/Cars.git
```

Navigate to the project directory:

```bash
cd Cars
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 20. Running the Project

### Step 1: Run the Tests

From the project root:

```bash
pytest -v
```

---

### Step 2: Start FastAPI

From the project root:

```bash
uvicorn API.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

### Step 3: Start Streamlit

Open a second terminal in the project root and run:

```bash
streamlit run APP/streamlit_app.py
```

The Streamlit application will open in the browser.

The default address is:

```text
http://localhost:8501
```

---

## 21. API Request Example

The prediction endpoint accepts the car features required by the trained model.

Example request format:

```json
{
    "features": {
        "year": 2019,
        "km_driven": 50000,
        "fuel": "Diesel"
    }
}
```

The actual request must contain the feature names required by the trained model.

---

## 22. API Response Example

The prediction endpoint returns the estimated car price.

Example response:

```json
{
    "predicted_price": 650000.0,
    "currency": "INR"
}
```

The actual predicted value depends on the vehicle information supplied to the model.

---

## 23. Model Prediction Example

The system can process vehicle information such as:

```text
Year
Kilometers Driven
Brand
Model
Fuel Type
Transmission
City
Other Vehicle Features
```

The information is passed through the trained preprocessing pipeline and machine learning model to generate an estimated market price.

---

## 24. Project Deliverables

The project contains the following major deliverables:

- Source code
- Jupyter notebooks
- Processed datasets
- Trained machine learning models
- FastAPI REST API
- Swagger API documentation
- Streamlit application
- Automated API tests
- Requirements file
- Dockerfile
- README documentation

---

## 25. Future Improvements

Possible future improvements include:

- Hyperparameter optimization
- Testing additional regression algorithms
- Advanced feature engineering
- Model explainability
- Improved Streamlit visualizations
- Cloud deployment
- API authentication
- Model monitoring
- Automated model retraining
- Real-time market data integration
- Improved price recommendation features

---

## 26. Author

**Name:** SITARAM

**GitHub:** https://github.com/ramcodes123-debug

**Project Repository:** https://github.com/ramcodes123-debug/Cars.git

---

## 27. License

This project is developed for educational and project submission purposes.