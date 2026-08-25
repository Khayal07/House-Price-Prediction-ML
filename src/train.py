import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from data_prep import load_and_prepare_data

def train_and_save_model():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data("../data/data.csv")
    
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    
    print("Setting up GridSearchCV for XGBoost...")
    xgb = XGBRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [4, 5, 6],
        'subsample': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='r2',
        cv=5,
        verbose=1,
        n_jobs=-1  
    )
    
    print(f"Training and Tuning on {len(X_train)} rows... (This might take a minute)")
    grid_search.fit(X_train, y_train_log)
    
    best_model = grid_search.best_estimator_
    print(f"\nBest Parameters Found: {grid_search.best_params_}")
    
    print("\nEvaluating best model...")
    predictions_log = best_model.predict(X_test)
    
    predictions = np.expm1(predictions_log)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"--- FINAL RESULTS ---")
    print(f"MAE: ${mae:,.2f}")
    print(f"R2 Score: {r2:.4f}\n")
    
    print("Saving best model to model.pkl...")
    joblib.dump(best_model, "model.pkl")
    print("Done! Model is optimized and ready.")

if __name__ == "__main__":
    train_and_save_model()