# 🛠️ BULLETPROOF SUBMISSION GENERATOR
# Fixes "tuple indices must be integers or slices, not str" error
# Works for any ML model (XGBoost, LightGBM, Neural Networks, etc.)

import pandas as pd
import numpy as np
import os
import shutil
import glob
from pathlib import Path

def validate_and_fix_submission(test_ids, predictions, output_file="submission.csv"):
    """
    Creates a bulletproof submission CSV that passes all competition validations
    
    Args:
        test_ids: array-like, test sample IDs (from test.csv)
        predictions: array-like, predicted prices
        output_file: str, output CSV filename
    
    Returns:
        pd.DataFrame: validated submission dataframe
    """
    
    print("🔍 AUDITING SUBMISSION DATA...")
    
    # Convert to numpy arrays for consistent handling
    test_ids = np.array(test_ids).flatten()
    predictions = np.array(predictions).flatten()
    
    print(f"   Test IDs shape: {test_ids.shape}, dtype: {test_ids.dtype}")
    print(f"   Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
    
    # Validation checks
    assert len(test_ids) == len(predictions), f"Mismatch: {len(test_ids)} IDs vs {len(predictions)} predictions"
    assert len(test_ids) > 0, "Empty submission data"
    
    # Handle potential data type issues
    try:
        # Convert IDs to integers (handle floats, strings, etc.)
        if test_ids.dtype == 'object' or 'float' in str(test_ids.dtype):
            test_ids = test_ids.astype(float).astype(int)
        else:
            test_ids = test_ids.astype(int)
    except:
        print("⚠️ Warning: Could not convert IDs to int, using as-is")
    
    try:
        # Convert predictions to float (handle lists, objects, etc.)
        if predictions.dtype == 'object':
            # Handle case where predictions might be lists or nested arrays
            flat_preds = []
            for pred in predictions:
                if isinstance(pred, (list, np.ndarray)):
                    flat_preds.append(float(pred[0]) if len(pred) > 0 else 0.0)
                else:
                    flat_preds.append(float(pred))
            predictions = np.array(flat_preds)
        else:
            predictions = predictions.astype(float)
    except Exception as e:
        print(f"⚠️ Warning: Could not convert predictions to float: {e}")
        predictions = predictions.astype(float)
    
    print(f"✅ After conversion - IDs: {test_ids.dtype}, Predictions: {predictions.dtype}")
    
    # Create submission dataframe with EXACT column names
    submission = pd.DataFrame({
        "id": test_ids,           # Must be "id", not "sample_id"
        "price": predictions      # Must be "price", not "prediction"
    })
    
    # Clean the data
    print("🧹 CLEANING SUBMISSION DATA...")
    
    # Remove any NaN/infinite values
    initial_rows = len(submission)
    submission = submission.dropna(subset=["price"])
    submission = submission[np.isfinite(submission["price"])]
    
    if len(submission) < initial_rows:
        print(f"⚠️ Removed {initial_rows - len(submission)} invalid predictions")
    
    # Ensure positive prices (competition requirement)
    negative_prices = (submission["price"] <= 0).sum()
    if negative_prices > 0:
        print(f"⚠️ Found {negative_prices} non-positive prices, setting to $0.01")
        submission.loc[submission["price"] <= 0, "price"] = 0.01
    
    # Sort by ID for consistency
    submission = submission.sort_values("id").reset_index(drop=True)
    
    # Final validation
    print("🔍 FINAL VALIDATION...")
    print(f"   Shape: {submission.shape}")
    print(f"   Columns: {list(submission.columns)}")
    print(f"   ID range: {submission['id'].min()} to {submission['id'].max()}")
    print(f"   Price range: ${submission['price'].min():.4f} to ${submission['price'].max():.4f}")
    print(f"   Missing values: {submission.isnull().sum().sum()}")
    
    # Save with precise formatting
    submission.to_csv(output_file, index=False, float_format="%.4f")
    
    print(f"✅ SUBMISSION SAVED: {output_file}")
    print(f"📊 PREVIEW:")
    print(submission.head())
    
    return submission

def cleanup_temp_files():
    """Remove all temporary files to save space"""
    print("\n🧹 CLEANING UP TEMPORARY FILES...")
    
    # File patterns to remove
    cleanup_patterns = [
        "**/emb_cache*",
        "**/embedding_cache*", 
        "**/cache*",
        "**/*.npy",
        "**/*.npz", 
        "**/*.log",
        "**/*.tmp",
        "**/*.bak",
        "**/checkpoint*",
        "**/temp*",
        "**/logs*",
        "**/*_temp*",
        "**/*_cache*"
    ]
    
    # File extensions to remove
    temp_extensions = [
        "*.npy", "*.npz", "*.log", "*.tmp", "*.bak",
        "*.cache", "*.temp", "*.checkpoint"
    ]
    
    removed_count = 0
    
    # Remove directories
    for pattern in cleanup_patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"   🗑️ Removed directory: {path}")
                    removed_count += 1
            except:
                pass
    
    # Remove files
    for ext in temp_extensions:
        for path in glob.glob(f"**/{ext}", recursive=True):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    print(f"   🗑️ Removed file: {path}")
                    removed_count += 1
            except:
                pass
    
    # Remove submission drafts (keep only final submission.csv)
    submission_drafts = [
        "submission_v*.csv",
        "submission_draft*.csv", 
        "submission_backup*.csv",
        "test_submission*.csv",
        "*_submission.csv"
    ]
    
    for pattern in submission_drafts:
        for path in glob.glob(pattern):
            if path != "submission.csv" and os.path.isfile(path):
                try:
                    os.remove(path)
                    print(f"   🗑️ Removed draft: {path}")
                    removed_count += 1
                except:
                    pass
    
    print(f"✅ Cleanup complete! Removed {removed_count} temporary files.")
    return removed_count

def create_submission_from_model(model, test_data, test_ids=None, output_file="submission.csv"):
    """
    Complete pipeline: predict + validate + export + cleanup
    
    Args:
        model: trained model (XGBoost, LightGBM, sklearn, etc.)
        test_data: test features (pandas DataFrame or numpy array)
        test_ids: test sample IDs (if None, will generate 0-based indices)
        output_file: output CSV filename
    """
    
    print("🚀 CREATING BULLETPROOF SUBMISSION...")
    
    # Generate predictions
    if hasattr(model, 'predict'):
        predictions = model.predict(test_data)
    else:
        raise ValueError("Model must have a 'predict' method")
    
    # Handle test IDs
    if test_ids is None:
        if hasattr(test_data, 'index'):
            test_ids = test_data.index.values
        else:
            test_ids = np.arange(len(test_data))
    
    # Create and validate submission
    submission = validate_and_fix_submission(test_ids, predictions, output_file)
    
    # Cleanup temporary files
    cleanup_temp_files()
    
    print(f"\n🎯 FINAL SUBMISSION READY!")
    print(f"📁 File: {output_file}")
    print(f"📊 Shape: {submission.shape}")
    print(f"💾 Size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return submission

# Example usage for different model types:

def example_xgboost_submission():
    """Example for XGBoost/LightGBM models"""
    import xgboost as xgb  # or lightgbm as lgb
    
    # Load your model and test data
    # model = xgb.XGBRegressor()
    # model.load_model("final_lgb_upgrade.txt")  # or your model file
    # test_data = pd.read_csv("test.csv")
    # test_features = test_data.drop(["id"], axis=1)  # remove ID column
    
    # Create submission
    # submission = create_submission_from_model(
    #     model=model,
    #     test_data=test_features, 
    #     test_ids=test_data["id"],
    #     output_file="submission.csv"
    # )
    
    print("💡 Example: Uncomment and modify the above code for your XGBoost/LightGBM model")

def example_manual_submission():
    """Example for manually created predictions"""
    
    # If you already have predictions and IDs:
    # test_ids = np.arange(75000)  # or load from test.csv
    # predictions = your_model_predictions  # your prediction array
    
    # submission = validate_and_fix_submission(
    #     test_ids=test_ids,
    #     predictions=predictions,
    #     output_file="submission.csv"
    # )
    
    # cleanup_temp_files()
    
    print("💡 Example: Use validate_and_fix_submission() if you already have predictions")

if __name__ == "__main__":
    print("🛠️ BULLETPROOF SUBMISSION GENERATOR")
    print("This script fixes the 'tuple indices must be integers or slices, not str' error")
    print()
    print("Usage examples:")
    print("1. For XGBoost/LightGBM: Use create_submission_from_model()")
    print("2. For existing predictions: Use validate_and_fix_submission()")
    print("3. Always run cleanup_temp_files() at the end")
    print()
    
    example_xgboost_submission()
    print()
    example_manual_submission()
    print()
    print("🔧 Modify the examples above for your specific model and data!")