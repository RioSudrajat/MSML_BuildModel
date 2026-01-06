import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import os

# === 1. Setup DagsHub & MLflow ===
print("Initializing DagsHub & MLflow...")
dagshub.init(repo_owner='RioSudrajat', repo_name='MSML', mlflow=True)
mlflow.set_experiment("Bank Marketing Tuning")

# === 2. Load Data ===
print("Loading data...")
if os.path.exists('train.csv'):
    train_path = 'train.csv'
    test_path = 'test.csv'
else:
    train_path = '../Membangun_model/train.csv'
    test_path = '../Membangun_model/test.csv' # Fallback for test

# Ensure test path is handled similar to modelling.py logic if it was just 'test.csv'
if not os.path.exists(train_path) and os.path.exists('train.csv'):
    train_path = 'train.csv'
    test_path = 'test.csv'

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path) # Load Test Data

df_train = pd.read_csv(train_path)
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']
X_test = df_test.drop('y', axis=1) # Test features
y_test = df_test['y'] # Test target

# === 3. Hyperparameter Tuning ===
print("Starting RandomizedSearchCV...")

with mlflow.start_run(run_name="RandomizedSearch_RF"):
    # Define Parameter Grid
    # Define Parameter Grid (Aligned with Notebook)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Define Model & Scorer
    rf = RandomForestClassifier(random_state=42)
    f1_scorer = make_scorer(f1_score)
    
    # Run Random Search (Aligned with Notebook)
    # n_iter=20 is a reasonable default to balance speed/search space if not specified, 
    # but notebook said "Fitting 3 folds for each of 20 candidates", so n_iter=20.
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, 
                                       n_iter=20, cv=3, scoring=f1_scorer, 
                                       verbose=2, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    # Get Best Results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"Best Params: {best_params}")
    print(f"Best CV F1 Score: {best_score:.4f}")
    
    # Log Best Params & Metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1_score", best_score)
    
    # Log Best Model
    mlflow.sklearn.log_model(random_search.best_estimator_, "best_model_tuned")
    
    # Evaluate on Test Set
    print("Evaluating best model on test set...")
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate Metrics Manually
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall: {rec:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    # Log Metrics Manually (Using standard names to align with DagsHub columns)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    print("Tuning complete. Best model and metrics logged to DagsHub.")
