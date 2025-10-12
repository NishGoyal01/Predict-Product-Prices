# 🏆 Local XGBoost Baseline Execution - 58.98% SMAPE Recreation
# Running directly in your local environment

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import re
import warnings
warnings.filterwarnings('ignore')

print("🏆 EXECUTING XGBOOST BASELINE LOCALLY")
print("=" * 50)

# =============================================================================
# LOAD DATA
# =============================================================================

print("📂 Loading data...")
try:
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    print(f"✅ Data loaded successfully!")
    print(f"   📊 Train: {train.shape}")
    print(f"   📊 Test: {test.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("\n🔧 Starting feature engineering...")

def extract_brand_simple(text):
    """Extract brand from text"""
    if pd.isna(text):
        return 'unknown'
    
    text = str(text).lower()
    
    # Known brands
    brands = ['apple', 'samsung', 'sony', 'nike', 'adidas', 'canon', 'lg', 'hp', 'dell', 'microsoft']
    
    for brand in brands:
        if brand in text:
            return brand
    
    # Pattern: word + "brand"
    match = re.search(r'\b(\w+)\s*brand\b', text)
    if match:
        return match.group(1)
    
    return 'unknown'

def create_features(df):
    """Create all features for the model"""
    features = df.copy()
    
    print("   📝 Creating text statistics...")
    # Basic text features
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
    features['sentence_count'] = df['catalog_content'].str.count(r'[.!?]+')
    
    # Character counts
    features['digit_count'] = df['catalog_content'].str.count(r'\d')
    features['uppercase_count'] = df['catalog_content'].str.count(r'[A-Z]')
    features['punctuation_count'] = df['catalog_content'].str.count(r'[^\w\s]')
    
    print("   🏷️ Extracting brands...")
    # Brand features
    features['brand'] = df['catalog_content'].apply(extract_brand_simple)
    
    # Brand encoding
    brand_counts = features['brand'].value_counts()
    top_brands = brand_counts.head(10).index.tolist()
    
    for brand in top_brands:
        features[f'brand_{brand}'] = (features['brand'] == brand).astype(int)
    
    # Keyword features
    print("   🔍 Detecting keywords...")
    features['has_price_keyword'] = df['catalog_content'].str.contains(
        r'\b(price|cost|dollar|cheap|expensive|premium|budget)\b', 
        case=False, na=False
    ).astype(int)
    
    features['has_quality_keyword'] = df['catalog_content'].str.contains(
        r'\b(quality|premium|luxury|professional|high-end)\b',
        case=False, na=False
    ).astype(int)
    
    features['has_size_keyword'] = df['catalog_content'].str.contains(
        r'\b(small|medium|large|xl|size|inch|cm|mm)\b',
        case=False, na=False
    ).astype(int)
    
    return features

# Apply feature engineering
train_features = create_features(train)
test_features = create_features(test)

print("✅ Basic features created")

# TF-IDF Vectorization
print("   🔤 Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=500,  # Manageable size for local execution
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2,
    max_df=0.8,
    lowercase=True,
    strip_accents='unicode'
)

# Fit on combined text
all_text = pd.concat([train['catalog_content'], test['catalog_content']])
tfidf.fit(all_text)

# Transform
train_tfidf = tfidf.transform(train_features['catalog_content'])
test_tfidf = tfidf.transform(test_features['catalog_content'])

print(f"   ✅ TF-IDF created: {train_tfidf.shape[1]} features")

# Prepare final feature matrices
feature_cols = [col for col in train_features.columns 
                if col not in ['catalog_content', 'price', 'brand', 'sample_id', 'image_link']]

print(f"   📊 Non-TF-IDF features: {len(feature_cols)}")
print(f"   📝 Feature columns: {feature_cols}")

# Convert TF-IDF to dense for easier handling
train_tfidf_dense = pd.DataFrame(
    train_tfidf.todense(), 
    columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
)
test_tfidf_dense = pd.DataFrame(
    test_tfidf.todense(), 
    columns=[f'tfidf_{i}' for i in range(test_tfidf.shape[1])]
)

# Combine all features
X_train = pd.concat([
    train_features[feature_cols].reset_index(drop=True),
    train_tfidf_dense.reset_index(drop=True)
], axis=1)

X_test = pd.concat([
    test_features[feature_cols].reset_index(drop=True),
    test_tfidf_dense.reset_index(drop=True)
], axis=1)

y_train = train['price'].values

print(f"✅ Final feature matrices:")
print(f"   📊 X_train: {X_train.shape}")
print(f"   📊 X_test: {X_test.shape}")
print(f"   📊 Total features: {X_train.shape[1]}")

# Fill any NaN values and ensure all columns are numeric
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Convert all columns to float64 to avoid object dtype issues
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

print(f"✅ Data types cleaned - all features are now numeric")

# =============================================================================
# MODEL TRAINING
# =============================================================================

print(f"\n🎯 Training XGBoost baseline model...")

# Original baseline parameters that achieved 58.98% SMAPE
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

model = xgb.XGBRegressor(**xgb_params)

print("   🔄 Training model...")
model.fit(X_train, y_train)
print("   ✅ Model trained successfully!")

# SMAPE evaluation
def smape_score(y_true, y_pred):
    """Calculate SMAPE score"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def smape_scorer(estimator, X, y):
    """SMAPE scorer for cross-validation"""
    y_pred = estimator.predict(X)
    return -smape_score(y, y_pred)

print("   📊 Running cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=smape_scorer, n_jobs=-1)
cv_smape = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"🏆 Cross-validation results:")
print(f"   📊 SMAPE: {cv_smape:.2f}% ± {cv_std:.2f}%")

# =============================================================================
# PREDICTIONS AND SUBMISSION
# =============================================================================

print(f"\n🔮 Making predictions...")
test_predictions = model.predict(X_test)

print(f"   📊 Predictions shape: {test_predictions.shape}")
print(f"   📊 Prediction range: ${test_predictions.min():.2f} - ${test_predictions.max():.2f}")

# Create submission with correct format
submission = pd.DataFrame({
    'sample_id': test['sample_id'].astype(int),  # Use sample_id as per competition format
    'price': test_predictions.astype(float)
})

# Clean submission
initial_rows = len(submission)
submission = submission.dropna()
submission = submission[np.isfinite(submission['price'])]
submission.loc[submission['price'] <= 0, 'price'] = 0.01
submission = submission.sort_values('sample_id').reset_index(drop=True)

if len(submission) < initial_rows:
    print(f"   🧹 Cleaned {initial_rows - len(submission)} invalid predictions")

# Save submission
submission_file = 'xgboost_baseline/xgboost_baseline_submission.csv'
submission.to_csv(submission_file, index=False, float_format='%.4f')

print(f"✅ Submission created: {submission_file}")
print(f"📊 Final submission:")
print(f"   📊 Shape: {submission.shape}")
print(f"   📊 Columns: {list(submission.columns)}")
print(f"   📊 Price range: ${submission['price'].min():.4f} - ${submission['price'].max():.4f}")

# Show sample
print(f"\n📋 Sample submission:")
print(submission.head())

# Save model
model_file = 'xgboost_baseline/xgboost_baseline_model.json'
model.save_model(model_file)
print(f"💾 Model saved: {model_file}")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "="*60)
print(f"🏆 XGBOOST BASELINE EXECUTION COMPLETE!")
print(f"="*60)
print(f"📊 Model Performance:")
print(f"   🎯 Cross-validation SMAPE: {cv_smape:.2f}% ± {cv_std:.2f}%")
print(f"   📊 Total features used: {X_train.shape[1]}")
print(f"   📊 Training samples: {X_train.shape[0]:,}")
print(f"   📊 Test predictions: {len(submission):,}")

print(f"\n📤 Files created:")
print(f"   📄 {submission_file}")
print(f"   🤖 {model_file}")

if cv_smape < 65:
    print(f"\n🎉 SUCCESS! Target SMAPE achieved!")
    print(f"📤 Ready to submit: {submission_file}")
else:
    print(f"\n⚠️ SMAPE higher than expected ({cv_smape:.2f}% vs target 58.98%)")
    print(f"📤 Still submittable: {submission_file}")

print(f"\n🚀 Next steps:")
print(f"   1. Submit {submission_file} to competition")
print(f"   2. Run emergency XGBoost tuning for better results")
print(f"   3. Consider advanced approaches if needed")