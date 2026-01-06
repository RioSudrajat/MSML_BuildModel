import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
import os

# === 1. Setup DagsHub & MLflow ===
print("Initializing DagsHub & MLflow...")
dagshub.init(repo_owner='RioSudrajat', repo_name='MSML', mlflow=True)
mlflow.set_experiment("Bank Marketing Tuning")

# === 2. Load Data ===
print("Loading data...")
if os.path.exists('train.csv'):
    train_path = 'train.csv'
else:
    train_path = '../Membangun_model/train.csv'

df_train = pd.read_csv(train_path)
X_train = df_train.drop('y', axis=1)
y_train = df_train['y']

# === 3. Hyperparameter Tuning ===
print("Starting GridSearchCV...")

with mlflow.start_run(run_name="GridSearch_RF"):
    # Define Parameter Grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Define Model & Scorer
    rf = RandomForestClassifier(random_state=42)
    f1_scorer = make_scorer(f1_score)
    
    # Run Grid Search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring=f1_scorer, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get Best Results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best Params: {best_params}")
    print(f"Best CV F1 Score: {best_score:.4f}")
    
    # Log Best Params & Metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_f1_score", best_score)
    
    # Log Best Model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model_tuned")
    
    print("Tuning complete. Best model logged to DagsHub.")
