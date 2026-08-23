import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_prep import load_and_prepare_data

def train_and_save_model():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data("../data/data.csv")
    
    print(f"Training Random Forest model on {len(X_train)} rows...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"\n--- RESULTS ---")
    print(f"MAE: ${mae:,.2f}")
    print(f"R2 Score: {r2:.4f}\n")
    
    print("Saving model to src/model.pkl...")
    joblib.dump(model, "model.pkl")
    print("Done! Model is ready for production.")

if __name__ == "__main__":
    train_and_save_model()