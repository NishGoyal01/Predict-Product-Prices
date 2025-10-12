import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

def train_model(data_path, target_col):
    # Load processed data
    df = pd.read_csv(data_path)
    
    # Features & target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Convert any categorical features to numeric
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train
    print("�� Training model...")
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    dump(model, "models/baseline_model.pkl")
    print("💾 Model saved to models/baseline_model.pkl")

if __name__ == "__main__":
    train_model("data/processed/train.csv", "price")

