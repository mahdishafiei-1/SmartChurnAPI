from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def get_param_grids():
    models = {}

    models["RandomForestClassifier"] = {
        "estimator" : RandomForestClassifier(random_state=42),
        "param_grid" : {
        "model__n_estimators" : [50, 100, 200],
        "model__max_features" : ["sqrt", "log2", None],
        "model__min_samples_leaf" : [1, 2, 5, 10]
        }
    }


    models["GradientBoostingClassifier"] = {
        "estimator" : GradientBoostingClassifier(random_state=42),
        "param_grid" : {
            "model__n_estimators": [50, 100, 200, 400],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__subsample": [0.8, 1.0]
        }
    }


    models["LogisticRegression"] = {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
        "model__penalty": ["l1", "l2", "elasticnet", None],
        "model__C" : [0.01, 0.1, 1, 10, 100],
        "model__solver": ["lbfgs"]
        }
    }


    models["KNeighborsClassifier"] = {
        "estimator" : KNeighborsClassifier(),
        "param_grid" : {
        "model__n_neighbors" : [3, 5, 7, 10],
        "model__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "model__weights": ["uniform", "distance"]
        }
    }


    models["DecisionTreeClassifier"] = {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "model__max_depth": [None, 5, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 5, 10, 20],
            "model__criterion": ["gini", "entropy"],
        }
    }

    return models