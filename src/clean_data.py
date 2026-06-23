import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "data", "raw", "Telco_customer_churn.csv")
cleaned_file_path = os.path.join(project_root, "data", "processed", "Cleaned_Telco_customer_churn.csv")

binary_columns = [
    "Partner", "Senior Citizen", "Dependents", "Paperless Billing", 
    "Tech Support", "Online Backup", "Streaming Movies", 
    "Streaming TV", "Multiple Lines", "Online Security", 
    "Contract", "Device Protection"
]

le = LabelEncoder()

def transform(columns):
    for col in columns:
        df[col] = le.fit_transform(df[col])

try:
    df = pd.read_csv(file_path)

    columns_to_drop = ["Unnamed: 0", "CustomerID", "Count", "Country", "State", "Zip Code", "Lat Long", "Churn Label", "City", "Gender", "Phone Service", "Churn Reason"]
    df = df.drop(columns_to_drop, axis=1, errors="ignore")

    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)


    df["Tech Support"] = df["Tech Support"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Online Backup"] = df["Online Backup"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Streaming Movies"] = df["Streaming Movies"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Streaming TV"] = df["Streaming TV"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Multiple Lines"] = df["Multiple Lines"].map({"No phone service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Online Backup"] = df["Online Backup"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Online Security"] = df["Online Security"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})
    df["Device Protection"] = df["Device Protection"].map({"No internet service" : "No", "No" : "No", "Yes" : "Yes"})

    transform(binary_columns)
    
    df = pd.get_dummies(df, columns=["Payment Method", "Internet Service"], dtype=int)
    
    df.to_csv(cleaned_file_path, index=False)

except Exception as e:
    print(f"ERROR: {e}")

