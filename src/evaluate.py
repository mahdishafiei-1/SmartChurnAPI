import os
import json
from sklearn.metrics import roc_auc_score, classification_report

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

def evaluate_model_performance(best_estimator, model_name: str, x_test, y_test, best_params, best_score):
    y_pred = best_estimator.predict(x_test)

    roc_auc = roc_auc_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    performance_data = {
        "evaluation_results": {
            "roc_auc": roc_auc,
            "report": report_dict
        },
        "best_parameters": best_params,
        "best_score": best_score
    }
    
    report_path = os.path.join(project_root, "reports", f"{model_name}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(performance_data, f, indent=4, ensure_ascii=False)

