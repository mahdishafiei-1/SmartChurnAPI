import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "data", "processed", "Cleaned_Telco_customer_churn.csv")
engineered_file_path = os.path.join(project_root, "data", "processed", "Engineered_Telco_customer_churn.csv")

def add_engineered_features(input_data):
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    else:
        df = input_data.copy()

    service_lst = ["Online Security", "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"]
    security_score_lst = ["Online Security", "Online Backup", "Device Protection", "Tech Support"]
    entertainment_score_lst = ["Streaming TV", "Streaming Movies"]

    df["Expected Monthly"] = df["Total Charges"] / df["Tenure Months"].replace(0, 1)

    df["Total Services"] = df[service_lst].apply(lambda x: sum(x == "Yes"), axis=1)
    df["Security Score"] = df[security_score_lst].apply(lambda x: sum(x == "Yes"), axis=1)
    df["Entertainment Score"] = df[entertainment_score_lst].apply(lambda x: sum(x == "Yes"), axis=1)

    return df

feature_engineer_transformer = FunctionTransformer(add_engineered_features, validate=False)

if __name__ == "__main__":
    df_engineered = add_engineered_features(file_path)
    df_engineered.to_csv(engineered_file_path, index=False)
