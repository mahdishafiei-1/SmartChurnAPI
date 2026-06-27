import os
import pandas as pd
from models import get_param_grids
from sklearn.pipeline import Pipeline
from evaluate import evaluate_model_performance
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "data", "processed", "Cleaned_Telco_customer_churn.csv")

def main():
    try:
        df = pd.read_csv(file_path)
        
        x = df.drop(["Churn Value", "Churn Score", "CLTV"], axis=1, errors="ignore")
        y = df["Churn Value"]

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y)

        categorical_cols = x.select_dtypes(exclude=['number']).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
            ],
            remainder='passthrough'
        )

        models_dict = get_param_grids()
        
        best_overall_score = 0.0
        best_overall_name = ""
        
        for model_name, config in models_dict.items():
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', config["estimator"])
            ])
            param_grid = config["param_grid"]

            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                n_jobs=-1,
                cv=5
            )
            grid_search.fit(x_train, y_train)
            
            current_score = grid_search.best_score_
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_name = model_name
            
            evaluate_model_performance(
                best_estimator=grid_search.best_estimator_,
                model_name=model_name,
                x_test=x_test,
                y_test=y_test,
                best_params=grid_search.best_params_,
                best_score=grid_search.best_score_
            )
            
            models_dir = os.path.join(project_root, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            model_file_path = os.path.join(models_dir, f"{model_name}.pkl")
            joblib.dump(grid_search.best_estimator_, model_file_path)
            print(f"{model_name} Saved.\n")

        print(f"BEST MODEL IS: {best_overall_name} with ROC-AUC: {best_overall_score:.4f}")

    except Exception as e:
        print(f"ERROR {e}")


if __name__ == "__main__":
    main()

