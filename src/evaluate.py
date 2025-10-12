import pandas as pd
from joblib import load

def predict(model_path, test_data_path, output_path="submissions/submission.csv"):
    model = load(model_path)
    test = pd.read_csv(test_data_path)
    preds = model.predict(test)
    pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
    print("✅ Submission saved at", output_path)

if __name__ == "__main__":
    predict("models/baseline_model.pkl", "data/processed/test.csv")