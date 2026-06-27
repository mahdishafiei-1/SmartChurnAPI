import os
import joblib
import pandas as pd
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Extra, Field

app = FastAPI(title="Smart Churn API")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
MODEL_NAME = "GradientBoostingClassifier.pkl"
model_path = os.path.join(project_root, "models", MODEL_NAME)

model = joblib.load(model_path)

class CustomerData(BaseModel):
    data: Dict[str, Any]

@app.post("/predict/single")
def predict_single(customer: CustomerData):
    try:
        input_data = customer.data
        data = pd.DataFrame([input_data])
        
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0][1]
        
        return {
            "Churn_Prediction": int(prediction),
            "Churn_Probability": float(probability),
            "Status": "Will Churn" if prediction == 1 else "Will Stay"
        }
    except Exception as e:
        raise HTTPException(detail=f"Error: {e}", status_code=400)

@app.post("/predict/file")
def predict_file(file: UploadFile = File(...)):

    try:
        if file.filename.endswith(".csv"):
            data = pd.read_csv(file.file)
        elif file.filename.endswith((".xls", ".xlsx")):
            data = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="format have to CSV or XLS or XLSX.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ERROR: {e}")
    try:
        columns_to_drop = ["Churn Value", "Churn Score", "CLTV"]
        data = data.drop(columns_to_drop, axis=1, errors="ignore")

        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]
        data["Churn_Prediction"] = prediction
        data["Churn_Probability"] = probability
        data["Churn_Status"] = data["Churn_Prediction"].map({1: "Will Churn", 0: "Will Stay"})

        output_data = data.to_dict(orient="records")
        
        return {
            "status": "success",
            "data": output_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ERROR: {e}")
    

