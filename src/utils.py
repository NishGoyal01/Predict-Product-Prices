import re
import os
import pandas as pd
import multiprocessing
from time import time as timer
from tqdm import tqdm
import numpy as np
from pathlib import Path
from functools import partial
import requests
import urllib.request
from PIL import Image

def download_image(image_link, savefolder):
    """Download single image from URL - Official implementation"""
    if(isinstance(image_link, str)):
        filename = Path(image_link).name
        image_save_path = os.path.join(savefolder, filename)
        if(not os.path.exists(image_save_path)):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)    
            except Exception as ex:
                print('Warning: Not able to download - {}\n{}'.format(image_link, ex))
        else:
            return
    return

def download_images(image_links, download_folder):
    """Download images using multiprocessing - Official implementation"""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    results = []
    download_image_partial = partial(download_image, savefolder=download_folder)
    with multiprocessing.Pool(100) as pool:
        for result in tqdm(pool.imap(download_image_partial, image_links), total=len(image_links)):
            results.append(result)
        pool.close()
        pool.join()

def load_dataset(file_path):
    """Load dataset from CSV"""
    return pd.read_csv(file_path)

def extract_ipq(catalog_content):
    """
    Extract Item Pack Quantity from catalog content
    Look for patterns like "Pack of X", "X-pack", etc.
    """
    if pd.isna(catalog_content):
        return 1
    
    # Common patterns for pack quantities
    patterns = [
        r'pack of (\d+)',
        r'(\d+)-pack',
        r'(\d+) pack',
        r'quantity[:\s]*(\d+)',
        r'count[:\s]*(\d+)',
        r'value:\s*(\d+\.?\d*)',  # Extract numerical values
        r'(\d+\.?\d*)\s*(ounce|oz|fl oz|count|lb|pound)',  # With units
    ]
    
    text = str(catalog_content).lower()
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    
    return 1  # Default to 1 if no pack info found

def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    
    # Avoid division by zero
    mask = denominator != 0
    smape = np.zeros_like(denominator)
    smape[mask] = diff[mask] / denominator[mask]
    
    return np.mean(smape) * 100  # Return as percentage

def preprocess_text(text):
    """Basic text preprocessing for catalog content"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Extract key information
    # Brand extraction (simple heuristic)
    brand_patterns = [
        r'item name:\s*([^,\n]+)',
        r'brand:\s*([^,\n]+)',
        r'^([A-Z][a-zA-Z]+)',  # First capitalized word
    ]
    
    return text.strip()

def extract_brand(catalog_content):
    """Extract brand name from catalog content"""
    if pd.isna(catalog_content):
        return "unknown"
    
    text = str(catalog_content)
    
    # Look for "Item Name:" pattern first
    brand_match = re.search(r'Item Name:\s*([^,\n]+)', text)
    if brand_match:
        brand = brand_match.group(1).strip()
        # Take first word as brand
        return brand.split()[0] if brand.split() else "unknown"
    
    return "unknown"

def extract_numerical_features(catalog_content):
    """Extract numerical features from catalog content"""
    if pd.isna(catalog_content):
        return {}
    
    text = str(catalog_content).lower()
    features = {}
    
    # Extract value and unit
    value_pattern = r'value:\s*(\d+\.?\d*)'
    unit_pattern = r'unit:\s*(\w+)'
    
    value_match = re.search(value_pattern, text)
    unit_match = re.search(unit_pattern, text)
    
    features['value'] = float(value_match.group(1)) if value_match else 0.0
    features['unit'] = unit_match.group(1) if unit_match else 'unknown'
    
    # Count bullet points (feature richness indicator)
    features['bullet_points'] = len(re.findall(r'bullet point \d+:', text))
    
    # Text length features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    
    return features

def validate_submission_format(submission_df, expected_samples):
    """Validate submission format matches requirements"""
    required_columns = ['sample_id', 'price']
    
    # Check columns
    if not all(col in submission_df.columns for col in required_columns):
        return False, f"Missing required columns. Expected: {required_columns}"
    
    # Check number of samples
    if len(submission_df) != expected_samples:
        return False, f"Expected {expected_samples} samples, got {len(submission_df)}"
    
    # Check for positive prices
    if (submission_df['price'] <= 0).any():
        return False, "All prices must be positive"
    
    # Check for missing values
    if submission_df[required_columns].isnull().any().any():
        return False, "Submission contains missing values"
    
    return True, "Submission format is valid"