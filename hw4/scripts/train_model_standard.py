import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import json

def main():
    dt = pd.read_csv("/tmp/fe_standard_data.csv")

    dt['num'] = (dt['num'] == 0).astype(int)

    # Assuming 'dt' is your DataFrame and 'num' is the target variable
    X = dt.drop('num', axis=1)  # Features
    y = dt['num']  # Target

    # Split the data with 90-10 ratio and stratify on 'y' to maintain ratio of classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    # Set up the hyperparameter grid for logistic regression
    log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear']}
    log_reg_grid = GridSearchCV(LogisticRegression(), log_reg_params, cv=5, scoring='accuracy')
    log_reg_grid.fit(X_train, y_train)

    # Set up the hyperparameter grid for random forest
    rf_params = {'n_estimators': [10, 50, 100, 200], 'max_features': ['auto', 'sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]}
    rf_grid = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)

    # Check best parameters and scores
    best_log_reg = {"best_params": log_reg_grid.best_params_, "best_score": log_reg_grid.best_score_}
    best_rf = {"best_params": rf_grid.best_params_, "best_score": rf_grid.best_score_}

    with open("/tmp/best_log_reg.json", "w") as f:
        json.dump(best_log_reg, f)

    with open("/tmp/best_rf.json", "w") as f:
        json.dump(best_rf, f)

    # Compare performance metrics and select the final model
    final_model = log_reg_grid if log_reg_grid.best_score_ > rf_grid.best_score_ else rf_grid
    print("Selected model type:", type(final_model.estimator).__name__)

if __name__ == "__main__":
    main()
