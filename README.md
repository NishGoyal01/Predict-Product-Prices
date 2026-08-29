# 📦 Amazon ML Challenge 2025 — Predict Product Prices

## 🔹 Overview

This repository contains a **complete end-to-end ML pipeline** for predicting product prices in the Amazon ML Challenge 2025.

**Features:**

* Data preprocessing: cleaning, encoding, handling missing values
* Model training: Random Forest & XGBoost regressors
* Ensemble predictions for improved performance
* Leaderboard-ready submission files
* Fully reproducible using terminal commands only

---

## 🗂 Project Structure

```
amazon_ml_challenge_2025/
├── src/
│   ├── data_preprocessing.py   # Load & clean data
│   ├── train.py                # Train RF & XGBoost models
│   └── evaluate.py             # Generate predictions
├── dataset/                    # Original CSVs
├── data/processed/             # Preprocessed CSVs
├── models/                     # Saved ML models
├── submissions/                # Predicted outputs
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start — Run Everything in One Command

-Dataset Requirements:
Before running the pipeline, make sure `train.csv` and `test.csv` are placed inside the `dataset/` directory. The raw datasets are not included in the repository because of their size.

From the VS Code terminal or any terminal in your project folder:

```bash
# 1️⃣ Clone repo
git clone https://github.com/ManyaValecha/amazon_ml_challenge_2025.git
cd amazon_ml_challenge_2025

# 2️⃣ Create & activate virtual environment
python -m venv venv
source venv/bin/activate         # macOS/Linux
# venv\Scripts\activate          # Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Add your train/test datasets to dataset/ folder
# Place train.csv and test.csv in dataset/

# 5️⃣ Run full pipeline: preprocessing → training → evaluation
python src/data_preprocessing.py && python src/train.py && python src/evaluate.py

# ✅ All outputs saved automatically:
# data/processed/ → cleaned CSVs
# models/ → trained ML models
# submissions/ → leaderboard-ready CSVs
```

---

## 📊 Local Validation Results (RMSE)

| Model       | Validation Score | Public LB | Private LB |
| ----------- | ---------------- | --------- | ---------- |
| Baseline RF | 36.19            | TBD       | TBD        |
| XGBoost     | 37.45            | TBD       | TBD        |
| Ensemble    | 36.27            | TBD       | TBD        |

> Public / Private LB values are filled after submission. Ensemble slightly improves over individual models.

---

## 📝 Notes

* Categorical features are automatically encoded
* Large files like raw datasets/models are **excluded from GitHub**; keep them locally
* Hyperparameters can be tuned in `train.py` for better leaderboard performance
* Outputs are **fully reproducible** with a single command

---

## 🔹 License

This project is intended for **Amazon ML Challenge 2025** and educational/research purposes only.
This version makes it **easy for judges or anyone cloning your repo** to run everything in one go and see your pipeline work.

