import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Extra, Field

app = FastAPI(title="Smart Churn API")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
MODEL_NAME = "GradientBoostingClassifier.pkl"
model_path = os.path.join(project_root, "models", MODEL_NAME)

model = joblib.load(model_path)

class CustomerData(BaseModel):
    Latitude: float
    Longitude: float
    Senior_Citizen: int = Field(alias="Senior Citizen")
    Partner: int
    Dependents: int
    Tenure_Months: int = Field(alias="Tenure Months")
    Multiple_Lines: int = Field(alias="Multiple Lines")
    Online_Security: int = Field(alias="Online Security")
    Online_Backup: int = Field(alias="Online Backup")
    Device_Protection: int = Field(alias="Device Protection")
    Tech_Support: int = Field(alias="Tech Support")
    Streaming_TV: int = Field(alias="Streaming TV")
    Streaming_Movies: int = Field(alias="Streaming Movies")
    Contract: int
    Paperless_Billing: int = Field(alias="Paperless Billing")
    Monthly_Charges: float = Field(alias="Monthly Charges")
    Total_Charges: float = Field(alias="Total Charges")
    Payment_Method_Bank_transfer_automatic: int = Field(alias="Payment Method_Bank transfer (automatic)")
    Payment_Method_Credit_card_automatic: int = Field(alias="Payment Method_Credit card (automatic)")
    Payment_Method_Electronic_check: int = Field(alias="Payment Method_Electronic check")
    Payment_Method_Mailed_check: int = Field(alias="Payment Method_Mailed check")
    Internet_Service_DSL: int = Field(alias="Internet Service_DSL")
    Internet_Service_Fiber_optic: int = Field(alias="Internet Service_Fiber optic")
    Internet_Service_No: int = Field(alias="Internet Service_No")

    class Config:
        populate_by_name = True
        extra = Extra.allow

@app.post("/predict/single")
def predict_single(customer: CustomerData):
    try:
        data = pd.DataFrame([customer.dict(by_alias=True)])
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
    

