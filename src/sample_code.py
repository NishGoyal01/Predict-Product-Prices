import pandas as pd
import numpy as np
from utils import load_dataset, extract_ipq, calculate_smape

def generate_sample_output():
    """
    Generate sample output in the required format
    This is a dummy implementation for demonstration
    """
    # Load test data (replace with actual test.csv path)
    # test_df = load_dataset('dataset/test.csv')
    
    # For demo, create sample data
    sample_ids = [f"TEST_{i:05d}" for i in range(1, 101)]
    
    # Generate dummy predictions (replace with actual model predictions)
    np.random.seed(42)
    predictions = np.random.uniform(10, 500, len(sample_ids))
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Ensure positive prices
    output_df['price'] = output_df['price'].clip(lower=0.01)
    
    # Save to CSV
    output_df.to_csv('submissions/sample_output.csv', index=False)
    print("✅ Sample output saved to submissions/sample_output.csv")
    
    return output_df

if __name__ == "__main__":
    # Generate sample output
    output = generate_sample_output()
    print(f"Generated {len(output)} predictions")
    print(f"Price range: ${output['price'].min():.2f} - ${output['price'].max():.2f}")
    print(f"Mean price: ${output['price'].mean():.2f}")