# 🏆 XGBoost Baseline Model - 58.98% SMAPE
# Original baseline model that achieved the best results

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import re
import sys
import os

# Add src to path for imports
sys.path.append('../src')

print("🚀 XGBoost Baseline Model - Recreating 58.98% SMAPE Results")
print("=" * 60)

# =============================================================================
# FEATURE ENGINEERING (From Proven Pipeline)
# =============================================================================

def extract_brand(text):
    """Extract brand from catalog content"""
    if pd.isna(text):
        return 'unknown'
    
    text = str(text).lower()
    
    # Brand patterns (from EDA insights)
    brand_patterns = [
        r'\b(apple|samsung|sony|nike|adidas|canon|lg|hp|dell|microsoft|google)\b',
        r'\b([a-z]+)\s*brand\b',
        r'\b([a-z]+)®\b',
        r'\b([a-z]+)™\b'
    ]
    
    for pattern in brand_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    
    return 'unknown'

def extract_ipq_features(text):
    """Extract Item-Per-Quantity features"""
    if pd.isna(text):
        return {'ipq_count': 0, 'ipq_avg_length': 0, 'ipq_variety': 0}
    
    text = str(text).lower()
    
    # IPQ patterns
    ipq_patterns = [
        r'\b(\d+)\s*(piece|pieces|pc|pcs|item|items|unit|units|pack|packs)\b',
        r'\b(\d+)\s*(kg|g|lb|oz|ml|l|liter|litre)\b',
        r'\b(\d+)\s*x\s*(\d+)\b'
    ]
    
    ipq_matches = []
    for pattern in ipq_patterns:
        matches = re.findall(pattern, text)
        ipq_matches.extend(matches)
    
    if not ipq_matches:
        return {'ipq_count': 0, 'ipq_avg_length': 0, 'ipq_variety': 0}
    
    return {
        'ipq_count': len(ipq_matches),
        'ipq_avg_length': np.mean([len(str(match)) for match in ipq_matches]),
        'ipq_variety': len(set(ipq_matches))
    }

def create_text_features(df):
    """Create text-based features"""
    features = df.copy()
    
    # Basic text statistics
    features['text_length'] = df['catalog_content'].str.len()
    features['word_count'] = df['catalog_content'].str.split().str.len()
    features['avg_word_length'] = features['text_length'] / features['word_count']
    features['sentence_count'] = df['catalog_content'].str.count(r'[.!?]+')
    
    # Special character counts
    features['digit_count'] = df['catalog_content'].str.count(r'\d')
    features['uppercase_count'] = df['catalog_content'].str.count(r'[A-Z]')
    features['punctuation_count'] = df['catalog_content'].str.count(r'[^\w\s]')
    
    # Price indicators
    features['has_price_keyword'] = df['catalog_content'].str.contains(
        r'\b(price|cost|dollar|cheap|expensive|premium|budget|affordable)\b', 
        case=False, na=False
    ).astype(int)
    
    # Quality indicators
    features['has_quality_keyword'] = df['catalog_content'].str.contains(
        r'\b(quality|premium|luxury|professional|high-end|deluxe)\b',
        case=False, na=False
    ).astype(int)
    
    return features

def engineer_features(train_df, test_df):
    """Complete feature engineering pipeline"""
    
    print("🔧 Starting feature engineering...")
    
    # Create text features
    train_features = create_text_features(train_df)
    test_features = create_text_features(test_df)
    
    # Extract brands
    print("   📝 Extracting brands...")
    train_features['brand'] = train_df['catalog_content'].apply(extract_brand)
    test_features['brand'] = test_df['catalog_content'].apply(extract_brand)
    
    # Brand encoding (top brands only)
    brand_counts = train_features['brand'].value_counts()
    top_brands = brand_counts.head(15).index.tolist()
    
    brand_encoder = LabelEncoder()
    all_brands = list(set(train_features['brand'].tolist() + test_features['brand'].tolist()))
    brand_encoder.fit(all_brands)
    
    train_features['brand_encoded'] = brand_encoder.transform(train_features['brand'])
    test_features['brand_encoded'] = brand_encoder.transform(test_features['brand'])
    
    # Create brand dummies for top brands
    for brand in top_brands:
        train_features[f'brand_{brand}'] = (train_features['brand'] == brand).astype(int)
        test_features[f'brand_{brand}'] = (test_features['brand'] == brand).astype(int)
    
    # Extract IPQ features
    print("   📊 Extracting IPQ features...")
    ipq_train = train_df['catalog_content'].apply(extract_ipq_features)
    ipq_test = test_df['catalog_content'].apply(extract_ipq_features)
    
    # Convert IPQ to DataFrame
    ipq_train_df = pd.DataFrame(ipq_train.tolist())
    ipq_test_df = pd.DataFrame(ipq_test.tolist())
    
    # Add IPQ features
    for col in ipq_train_df.columns:
        train_features[col] = ipq_train_df[col]
        test_features[col] = ipq_test_df[col]
    
    # TF-IDF Vectorization
    print("   🔤 Creating TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=800,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    # Fit on combined text to ensure consistent vocabulary
    all_text = pd.concat([train_df['catalog_content'], test_df['catalog_content']])
    tfidf.fit(all_text)
    
    # Transform train and test
    train_tfidf = tfidf.transform(train_features['catalog_content'])
    test_tfidf = tfidf.transform(test_features['catalog_content'])
    
    # Convert to DataFrame
    tfidf_cols = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    train_tfidf_df = pd.DataFrame(train_tfidf.todense(), columns=tfidf_cols, index=train_features.index)
    test_tfidf_df = pd.DataFrame(test_tfidf.todense(), columns=tfidf_cols, index=test_features.index)
    
    # Combine all features
    feature_cols = [col for col in train_features.columns if col not in ['catalog_content', 'price', 'brand']]
    
    X_train = pd.concat([
        train_features[feature_cols].reset_index(drop=True),
        train_tfidf_df.reset_index(drop=True)
    ], axis=1)
    
    X_test = pd.concat([
        test_features[feature_cols].reset_index(drop=True),
        test_tfidf_df.reset_index(drop=True)
    ], axis=1)
    
    print(f"✅ Feature engineering complete!")
    print(f"   📊 Training features: {X_train.shape}")
    print(f"   📊 Test features: {X_test.shape}")
    
    return X_train, X_test, feature_cols + tfidf_cols

# =============================================================================
# MODEL TRAINING (Original Parameters)
# =============================================================================

def train_xgboost_baseline(X_train, y_train):
    """Train XGBoost with original baseline parameters"""
    
    print("🎯 Training XGBoost baseline model...")
    
    # Original parameters that achieved 58.98% SMAPE
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
        'objective': 'reg:squarederror'
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    
    # Custom SMAPE scoring
    def smape_scorer(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    
    # Cross-validation score
    def smape_cv_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return -smape_scorer(y, y_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=smape_cv_scorer, n_jobs=-1)
    cv_smape = -cv_scores.mean()
    
    print(f"✅ Model trained successfully!")
    print(f"🏆 Cross-validation SMAPE: {cv_smape:.2f}%")
    
    return model, cv_smape

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def create_xgboost_baseline_submission():
    """Complete pipeline to recreate 58.98% SMAPE submission"""
    
    # Load data
    print("📂 Loading data...")
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    
    print(f"✅ Data loaded: Train {train.shape}, Test {test.shape}")
    
    # Feature engineering
    X_train, X_test, feature_names = engineer_features(train, test)
    y_train = train['price']
    
    # Train model
    model, cv_smape = train_xgboost_baseline(X_train, y_train)
    
    # Make predictions
    print("🔮 Making predictions...")
    test_predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test['id'].astype(int),
        'price': test_predictions.astype(float)
    })
    
    # Clean submission
    submission = submission.dropna()
    submission.loc[submission['price'] <= 0, 'price'] = 0.01
    submission = submission.sort_values('sample_id').reset_index(drop=True)
    
    # Save submission
    submission.to_csv('xgboost_baseline_submission.csv', index=False, float_format='%.4f')
    
    print(f"✅ Submission created: xgboost_baseline_submission.csv")
    print(f"📊 Shape: {submission.shape}")
    print(f"📊 Estimated SMAPE: {cv_smape:.2f}%")
    print(f"📊 Price range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
    
    # Save model
    model.save_model('xgboost_baseline_model.json')
    print(f"💾 Model saved: xgboost_baseline_model.json")
    
    return submission, model, cv_smape

if __name__ == "__main__":
    submission, model, smape = create_xgboost_baseline_submission()
    
    print(f"\n🎯 XGBOOST BASELINE RECREATION COMPLETE!")
    print(f"🏆 Expected SMAPE: {smape:.2f}%")
    print(f"📤 Ready to submit: xgboost_baseline_submission.csv")
    print(f"💾 Model file: xgboost_baseline_model.json")