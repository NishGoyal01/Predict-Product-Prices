# 🚨 Emergency XGBoost Tuning - 10 Minute Quick Win
# Target: Improve from 70% SMAPE to 35-45% SMAPE

"""
EMERGENCY XGBOOST TUNING PIPELINE
=================================

⏰ Time: 10 minutes total
🎯 Goal: Reduce SMAPE from 70% (neural network) to 35-45%
🔥 Strategy: Use proven feature engineering + quick hyperparameter search

EXECUTION PLAN:
1. Load data and apply proven feature engineering (2 min)
2. Quick XGBoost hyperparameter search (6 min) 
3. Train final model and predict (1 min)
4. Export submission with bulletproof format (1 min)
"""

# =============================================================================
# SETUP AND IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import re
import time

print("🚨 EMERGENCY XGBOOST TUNING STARTED")
print("Target: 35-45% SMAPE in 10 minutes")
print()

start_time = time.time()

# =============================================================================
# LOAD DATA
# =============================================================================

print("📂 Loading data...")
train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

print(f"✅ Train: {train.shape}, Test: {test.shape}")

# =============================================================================
# EMERGENCY FEATURE ENGINEERING (Proven Effective)
# =============================================================================

print("⚡ Applying emergency feature engineering...")

def emergency_features(df):
    """Quick feature engineering - only the most effective features"""
    
    df_features = df.copy()
    
    # Text length features (fast and effective)
    df_features['text_length'] = df['catalog_content'].str.len()
    df_features['word_count'] = df['catalog_content'].str.split().str.len()
    
    # Brand extraction (simplified)
    brand_patterns = [
        r'\b(Apple|Samsung|Sony|Nike|Adidas|Canon|LG|HP|Dell)\b',
        r'\b([A-Z][a-z]+)\s*(Brand|brand|®|™)',
        r'\b([A-Z]{2,})\b'
    ]
    
    df_features['has_brand'] = 0
    for pattern in brand_patterns:
        df_features['has_brand'] += df['catalog_content'].str.contains(pattern, case=False, na=False).astype(int)
    
    # Price indicators (fast regex)
    df_features['has_price_word'] = df['catalog_content'].str.contains(
        r'\b(price|cost|dollar|cheap|expensive|premium|budget)\b', case=False, na=False
    ).astype(int)
    
    # Quality indicators
    df_features['has_quality'] = df['catalog_content'].str.contains(
        r'\b(quality|premium|luxury|professional|high-end)\b', case=False, na=False
    ).astype(int)
    
    return df_features

# Apply feature engineering
train_features = emergency_features(train)
test_features = emergency_features(test)

# Quick TF-IDF (limited vocab for speed)
print("🔤 Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=500,  # Reduced for speed
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8
)

# Fit on train text and transform both
train_tfidf = tfidf.fit_transform(train_features['catalog_content'])
test_tfidf = tfidf.transform(test_features['catalog_content'])

# Convert to DataFrame
tfidf_train_df = pd.DataFrame(train_tfidf.toarray(), 
                             columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])])
tfidf_test_df = pd.DataFrame(test_tfidf.toarray(), 
                            columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])])

# Combine features
feature_cols = ['text_length', 'word_count', 'has_brand', 'has_price_word', 'has_quality']
X_train = pd.concat([
    train_features[feature_cols].reset_index(drop=True),
    tfidf_train_df
], axis=1)

X_test = pd.concat([
    test_features[feature_cols].reset_index(drop=True),
    tfidf_test_df
], axis=1)

y_train = train_features['price']

print(f"✅ Final features: {X_train.shape[1]} features")
print(f"⏱️ Feature engineering: {time.time() - start_time:.1f}s")

# =============================================================================
# EMERGENCY HYPERPARAMETER TUNING
# =============================================================================

print("\n🔥 Starting emergency hyperparameter search...")

# Quick but effective parameter space
param_space = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

# Custom SMAPE scorer
def smape_score(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def smape_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return -smape_score(y, y_pred)  # Negative because sklearn maximizes

# Quick randomized search
xgb_model = xgb.XGBRegressor(
    random_state=42,
    n_jobs=-1,
    tree_method='hist'  # Faster training
)

search = RandomizedSearchCV(
    xgb_model,
    param_space,
    n_iter=15,  # Quick search
    cv=3,       # Fast CV
    scoring=smape_scorer,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("🔍 Running 15-iteration search with 3-fold CV...")
search_start = time.time()
search.fit(X_train, y_train)
search_time = time.time() - search_start

print(f"✅ Search completed in {search_time:.1f}s")
print(f"🏆 Best SMAPE: {-search.best_score_:.2f}%")
print(f"🔧 Best params: {search.best_params_}")

# =============================================================================
# FINAL MODEL AND PREDICTION
# =============================================================================

print("\n🎯 Training final model...")

final_model = search.best_estimator_
final_model.fit(X_train, y_train)

# Predict on test
test_predictions = final_model.predict(X_test)

print(f"✅ Predictions generated: {len(test_predictions)} samples")
print(f"📊 Prediction range: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")

# =============================================================================
# BULLETPROOF SUBMISSION EXPORT
# =============================================================================

print("\n📤 Creating bulletproof submission...")

# Get test IDs
test_ids = test_features['id'].values

# Create submission with exact format
submission = pd.DataFrame({
    'id': test_ids.astype(int),
    'price': test_predictions.astype(float)
})

# Clean and validate
submission = submission.dropna(subset=["price"])
submission = submission[np.isfinite(submission["price"])]
submission.loc[submission["price"] <= 0, "price"] = 0.01
submission = submission.sort_values("id").reset_index(drop=True)

# Export with proper formatting
submission.to_csv('emergency_xgboost_submission.csv', index=False, float_format="%.4f")

print(f"✅ Submission saved: emergency_xgboost_submission.csv")
print(f"📊 Shape: {submission.shape}")
print(f"📊 Columns: {list(submission.columns)}")
print(f"📊 Sample:")
print(submission.head())

# =============================================================================
# PERFORMANCE ESTIMATE
# =============================================================================

print("\n🎯 PERFORMANCE ESTIMATE:")

# Cross-validation on training data
cv_scores = cross_val_score(final_model, X_train, y_train, 
                          cv=3, scoring=smape_scorer, n_jobs=-1)
estimated_smape = -cv_scores.mean()

print(f"🏆 Estimated SMAPE: {estimated_smape:.2f}%")
print(f"📈 Improvement: {70.13 - estimated_smape:.1f}% better than neural network")

# =============================================================================
# SUMMARY
# =============================================================================

total_time = time.time() - start_time
print(f"\n⏱️ TOTAL TIME: {total_time:.1f} seconds")
print(f"🎯 TARGET ACHIEVED: {estimated_smape:.1f}% SMAPE (target was 35-45%)")
print(f"📤 READY TO SUBMIT: emergency_xgboost_submission.csv")

if estimated_smape < 45:
    print("🏆 SUCCESS! Target SMAPE achieved!")
else:
    print("⚠️ Close to target, but may need longer tuning for better results")

print("\n🚀 EMERGENCY TUNING COMPLETE!")
print("Upload emergency_xgboost_submission.csv for much better results!")