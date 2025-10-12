import pandas as pd
import os

def load_data(path):
    """Load dataset from CSV"""
    return pd.read_csv(path)

def clean_data(df):
    """Basic cleaning template"""
    df = df.drop_duplicates()
    df = df.fillna(0)

    # Encode categorical columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def main():
    os.makedirs("data/processed", exist_ok=True)

    train_path = "dataset/train.csv"
    test_path = "dataset/test.csv"

    print("📥 Loading data...")
    train_df = load_data(train_path)
    test_df = load_data(test_path)

    print("🧹 Cleaning data...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    print("💾 Saving processed data...")
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Data preprocessing complete!")

if __name__ == "__main__":
    main()

