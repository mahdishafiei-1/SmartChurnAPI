# SmartChurnAPI

A production-ready, dynamic Machine Learning pipeline and REST API designed to predict telecom customer churn using the IBM Telco dataset. Built with scalability, clean software architecture, and strict anti-data-leakage principles.

---

## Key Features

* **Anti-Data-Leakage Design:** Preprocessing, scaling, and categorical encoding are fully decoupled from raw data preparation and encapsulated inside an isolated, robust Scikit-Learn Pipeline.
* **Dynamic FastAPI Inference:** The API doesn't rely on hardcoded feature columns. It accepts a dynamic JSON payload, and the underlying pipeline automatically handles alignment, missing data, and transformations.
* **Bulk & Single Inference:** Offers dedicated endpoints for predicting a single customer's status or uploading bulk files (`.csv`, `.xlsx`).
* **Automated Hyperparameter Tuning:** Model exploration via `GridSearchCV` measuring ROC-AUC optimization across multiple estimators.

---

## Tech Stack

* **Core:** Python 3.10+
* **ML Framework:** Scikit-Learn, Pandas, Joblib
* **API Development:** FastAPI, Uvicorn, Pydantic

---

## Project Structure
```
├── data/
│   ├── raw/          # Original IBM Telco Dataset
│   └── processed/    # Cleaned textual data
├── models/           # Serialized Scikit-Learn Pipelines (.pkl)
├── reports/          # Evaluation metrics and classification reports
└── src/
    ├── api/
    │   └── main.py   # FastAPI application & endpoints
    ├── clean_data.py # Structured preprocessing script
    ├── models.py     # GridSearch model configurations
    ├── train.py      # Training pipeline execution
    └── evaluate.py   # Evaluation & validation logic
```

## Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```
pip install -r requirements.txt
```
### 2. Run the Pipeline
First, clean the raw data into standard text formats, then execute the training loop to save the optimized operational pipeline:
```
python src/clean_data.py
python src/train.py
```
### 3. Launch the API Server
Start the Uvicorn deployment server:
```
uvicorn src.main:app --reload
```
Open your browser and navigate to http://127.0.0.1:8000/docs to test the dynamic Swagger UI documentation.

---

## API Endpoints

### POST /predict/single
Accepts a flexible payload structure containing raw customer data.

Example Request Body:
```
{
  "data": {
    "Latitude": 33.96,
    "Longitude": -118.27,
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2,
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Mailed check",
    "Monthly Charges": 53.85,
    "Total Charges": 108.15
  }
}
```

### POST /predict/file
Accepts an uploaded .csv or .xlsx batch file, processes all records concurrently through the pipeline, and appends the final predictions.
