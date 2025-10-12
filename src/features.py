"""
Feature Engineering Pipeline for Amazon ML Challenge 2025
========================================================

This module implements the feature engineering pipeline based on EDA insights:
- TF-IDF vectorization for text content
- Brand extraction and encoding
- IPQ normalization and binning
- Text statistics computation
- Price transformation utilities

Key Insights from EDA:
- Text length correlation with price: 0.167
- Brand variance in pricing: High (CV: 0.59)
- Price skewness: 13.60 (needs log transformation)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import csr_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import extract_brand, extract_ipq, calculate_smape

class FeatureEngineer:
    """
    Comprehensive feature engineering pipeline for Amazon ML Challenge
    """
    
    def __init__(self, 
                 tfidf_max_features=2000,
                 top_brands_count=50,
                 ipq_bins=10,
                 min_brand_count=5):
        """
        Initialize feature engineering pipeline
        
        Args:
            tfidf_max_features (int): Maximum features for TF-IDF vectorization
            top_brands_count (int): Number of top brands to encode separately
            ipq_bins (int): Number of bins for IPQ discretization
            min_brand_count (int): Minimum count for brand to be considered
        """
        self.tfidf_max_features = tfidf_max_features
        self.top_brands_count = top_brands_count
        self.ipq_bins = ipq_bins
        self.min_brand_count = min_brand_count
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.brand_encoder = None
        self.top_brands = set()  # Initialize as empty set
        self.ipq_bins_edges = np.array([])  # Initialize as empty array
        self.text_scaler = None
        
        # Feature names for tracking
        self.feature_names = []
        self.is_fitted = False
        
    def preprocess_text(self, text_series):
        """
        Clean and preprocess text content
        
        Args:
            text_series (pd.Series): Series of text content
            
        Returns:
            pd.Series: Cleaned text content
        """
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove special characters but keep alphanumeric and spaces
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        return text_series.apply(clean_text)
    
    def extract_text_features(self, text_series):
        """
        Extract basic text statistics
        
        Args:
            text_series (pd.Series): Series of text content
            
        Returns:
            pd.DataFrame: Text feature dataframe
        """
        features = pd.DataFrame()
        
        # Basic length features
        features['text_length'] = text_series.str.len()
        features['word_count'] = text_series.str.split().str.len()
        features['char_density'] = features['text_length'] / (features['word_count'] + 1)
        
        # Advanced text features
        features['avg_word_length'] = features['text_length'] / (features['word_count'] + 1)
        features['digit_count'] = text_series.str.count(r'\d')
        features['upper_count'] = text_series.str.count(r'[A-Z]')
        features['punctuation_count'] = text_series.str.count(r'[^\w\s]')
        
        # Ratio features
        features['digit_ratio'] = features['digit_count'] / (features['text_length'] + 1)
        features['upper_ratio'] = features['upper_count'] / (features['text_length'] + 1)
        features['punct_ratio'] = features['punctuation_count'] / (features['text_length'] + 1)
        
        # Fill any NaN values
        features = features.fillna(0)
        
        return features
    
    def process_brands(self, text_series, fit=True):
        """
        Extract and encode brand information
        
        Args:
            text_series (pd.Series): Series of text content
            fit (bool): Whether to fit the encoder
            
        Returns:
            pd.DataFrame: Brand features
        """
        # Extract brands
        brands = text_series.apply(extract_brand)
        
        if fit:
            # Get brand counts and identify top brands
            brand_counts = brands.value_counts()
            self.top_brands = set(brand_counts.head(self.top_brands_count).index)
            
            # Filter brands by minimum count
            valid_brands = set(brand_counts[brand_counts >= self.min_brand_count].index)
            self.top_brands = self.top_brands.intersection(valid_brands)
            
            print(f"📊 Identified {len(self.top_brands)} top brands for encoding")
        
        # Create brand features
        brand_features = pd.DataFrame()
        
        # Binary features for top brands
        for brand in self.top_brands:
            brand_features[f'brand_{brand.lower().replace(" ", "_")}'] = (brands == brand).astype(int)
        
        # Brand frequency (how common is this brand)
        brand_counts_map = brands.value_counts().to_dict()
        brand_features['brand_frequency'] = brands.map(brand_counts_map).fillna(1)
        
        # Brand length
        brand_features['brand_length'] = brands.str.len().fillna(0)
        
        # Is brand extracted (binary)
        brand_features['has_brand'] = (brands.notna() & (brands != "")).astype(int)
        
        return brand_features
    
    def process_ipq(self, text_series, fit=True):
        """
        Extract and process Item Pack Quantity information
        
        Args:
            text_series (pd.Series): Series of text content
            fit (bool): Whether to fit the binning
            
        Returns:
            pd.DataFrame: IPQ features
        """
        # Extract IPQ values
        ipqs = text_series.apply(extract_ipq)
        
        # Handle extreme outliers (based on EDA: normal range 0-96)
        ipqs_clipped = np.clip(ipqs, 0, 100)
        
        if fit:
            # Create bins for IPQ discretization
            self.ipq_bins_edges = np.percentile(ipqs_clipped[ipqs_clipped > 0], 
                                              np.linspace(0, 100, self.ipq_bins + 1))
            print(f"📦 IPQ bin edges: {self.ipq_bins_edges}")
        
        ipq_features = pd.DataFrame()
        
        # Original IPQ (log transformed for extreme values)
        ipq_features['ipq_raw'] = ipqs
        ipq_features['ipq_log'] = np.log1p(ipqs)
        ipq_features['ipq_clipped'] = ipqs_clipped
        
        # Binned IPQ
        if len(self.ipq_bins_edges) > 1:
            ipq_features['ipq_binned'] = pd.cut(ipqs_clipped, 
                                              bins=self.ipq_bins_edges.tolist(), 
                                              labels=False, 
                                              duplicates='drop').fillna(0)
        else:
            ipq_features['ipq_binned'] = 0
        
        # IPQ indicators
        ipq_features['has_ipq'] = (ipqs > 0).astype(int)
        ipq_features['is_single_item'] = (ipqs == 1).astype(int)
        ipq_features['is_bulk'] = (ipqs > 10).astype(int)
        
        return ipq_features
    
    def create_tfidf_features(self, text_series, fit=True):
        """
        Create TF-IDF features from text content
        
        Args:
            text_series (pd.Series): Series of text content
            fit (bool): Whether to fit the vectorizer
            
        Returns:
            pd.DataFrame: TF-IDF features
        """
        # Preprocess text
        clean_text = self.preprocess_text(text_series)
        
        if fit:
            # Initialize and fit TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Only words starting with letter
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(clean_text)
            print(f"🔤 TF-IDF matrix shape: {tfidf_matrix.shape}")
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call fit_transform first.")
            tfidf_matrix = self.tfidf_vectorizer.transform(clean_text)
        
        # Convert to DataFrame - handle sparse matrix
        try:
            tfidf_array = tfidf_matrix.toarray()
        except:
            tfidf_array = np.array(tfidf_matrix)
            
        tfidf_features = pd.DataFrame(
            tfidf_array,
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=text_series.index
        )
        
        return tfidf_features
    
    def fit_transform(self, df):
        """
        Fit the feature engineering pipeline and transform data
        
        Args:
            df (pd.DataFrame): Input dataframe with 'catalog_content' column
            
        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        print("🔧 Fitting and transforming features...")
        
        # Extract text content
        text_content = df['catalog_content'].fillna("")
        
        # 1. Text statistics features
        print("📊 Extracting text statistics...")
        text_features = self.extract_text_features(text_content)
        
        # 2. Brand features
        print("🏷️ Processing brand features...")
        brand_features = self.process_brands(text_content, fit=True)
        
        # 3. IPQ features
        print("📦 Processing IPQ features...")
        ipq_features = self.process_ipq(text_content, fit=True)
        
        # 4. TF-IDF features
        print("🔤 Creating TF-IDF features...")
        tfidf_features = self.create_tfidf_features(text_content, fit=True)
        
        # Combine all features
        feature_matrix = pd.concat([
            text_features,
            brand_features,
            ipq_features,
            tfidf_features
        ], axis=1)
        
        # Scale text statistics features
        text_feature_cols = text_features.columns
        self.text_scaler = StandardScaler()
        if len(text_feature_cols) > 0:
            feature_matrix[text_feature_cols] = self.text_scaler.fit_transform(
                feature_matrix[text_feature_cols]
            )
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        self.is_fitted = True
        
        print(f"✅ Feature engineering complete!")
        print(f"📊 Final feature matrix shape: {feature_matrix.shape}")
        print(f"🔢 Feature breakdown:")
        print(f"   - Text statistics: {len(text_features.columns)}")
        print(f"   - Brand features: {len(brand_features.columns)}")
        print(f"   - IPQ features: {len(ipq_features.columns)}")
        print(f"   - TF-IDF features: {len(tfidf_features.columns)}")
        
        return feature_matrix
    
    def transform(self, df):
        """
        Transform new data using fitted pipeline
        
        Args:
            df (pd.DataFrame): Input dataframe with 'catalog_content' column
            
        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Use fit_transform() first.")
        
        print("🔄 Transforming features...")
        
        # Extract text content
        text_content = df['catalog_content'].fillna("")
        
        # 1. Text statistics features
        text_features = self.extract_text_features(text_content)
        
        # 2. Brand features
        brand_features = self.process_brands(text_content, fit=False)
        
        # 3. IPQ features
        ipq_features = self.process_ipq(text_content, fit=False)
        
        # 4. TF-IDF features
        tfidf_features = self.create_tfidf_features(text_content, fit=False)
        
        # Combine all features
        feature_matrix = pd.concat([
            text_features,
            brand_features,
            ipq_features,
            tfidf_features
        ], axis=1)
        
        # Scale text statistics features
        text_feature_cols = [col for col in text_features.columns if col in feature_matrix.columns]
        if len(text_feature_cols) > 0 and self.text_scaler is not None:
            feature_matrix[text_feature_cols] = self.text_scaler.transform(
                feature_matrix[text_feature_cols]
            )
        
        # Ensure same features as training
        for col in self.feature_names:
            if col not in feature_matrix.columns:
                feature_matrix[col] = 0
        
        feature_matrix = feature_matrix[self.feature_names]
        
        print(f"✅ Transform complete! Shape: {feature_matrix.shape}")
        
        return feature_matrix

def prepare_target(prices, transform_type='log'):
    """
    Transform target variable for better model performance
    
    Args:
        prices (pd.Series): Price values
        transform_type (str): Type of transformation ('log', 'sqrt', 'none')
        
    Returns:
        pd.Series: Transformed prices
    """
    if transform_type == 'log':
        return np.log1p(prices)  # log(1 + price) to handle zero prices
    elif transform_type == 'sqrt':
        return np.sqrt(prices)
    else:
        return prices

def inverse_transform_target(transformed_prices, transform_type='log'):
    """
    Inverse transform predictions back to original scale
    
    Args:
        transformed_prices (pd.Series or np.array): Transformed price predictions
        transform_type (str): Type of transformation used
        
    Returns:
        np.array: Prices in original scale
    """
    if transform_type == 'log':
        return np.expm1(transformed_prices)  # exp(price) - 1
    elif transform_type == 'sqrt':
        return np.square(transformed_prices)
    else:
        return transformed_prices

# Quick feature engineering utility for rapid prototyping
def quick_features(df, max_features=1000):
    """
    Quick feature extraction for rapid baseline development
    
    Args:
        df (pd.DataFrame): Input dataframe
        max_features (int): Maximum TF-IDF features
        
    Returns:
        tuple: (feature_matrix, feature_names)
    """
    engineer = FeatureEngineer(tfidf_max_features=max_features)
    features = engineer.fit_transform(df)
    return features, engineer.feature_names

if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("🧪 Testing Feature Engineering Pipeline...")
    
    # Load sample data
    train_df = pd.read_csv('../dataset/train.csv')
    sample_df = train_df.sample(1000, random_state=42)
    
    # Test feature engineering
    engineer = FeatureEngineer(tfidf_max_features=500)
    features = engineer.fit_transform(sample_df)
    
    print(f"\n📊 Sample feature engineering results:")
    print(f"Input shape: {sample_df.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature names sample: {engineer.feature_names[:10]}")
    
    # Test target transformation
    original_prices = sample_df['price']
    log_prices = prepare_target(original_prices, 'log')
    recovered_prices = inverse_transform_target(log_prices, 'log')
    
    mse = np.mean((original_prices - recovered_prices) ** 2)
    print(f"\n🎯 Target transformation test - MSE: {mse:.6f}")
    
    print("\n✅ Feature engineering pipeline test complete!")