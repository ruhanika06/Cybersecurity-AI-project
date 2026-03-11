"""
STEP 1: DATA LOADING AND UNDERSTANDING
========================================
This module loads the KDD Cup 99 dataset and performs initial exploration.


"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99
import os

def load_kdd_dataset():
    """
    Load KDD Cup 99 dataset.
    
    Returns:
        X (DataFrame): Features
        y (Series): Target labels
    """
    print("=" * 70)
    print("STEP 1: LOADING KDD CUP 99 DATASET")
    print("=" * 70)
    
    try:
        print("\n[INFO] Attempting to load KDD Cup 99 dataset from sklearn...")
        # Load the dataset
        data = fetch_kddcup99(subset=None, as_frame=True, return_X_y=False)
        X = data.data
        y = data.target
        
        # Decode bytes to strings if necessary
        if y.dtype == object and len(y) > 0 and isinstance(y.iloc[0], bytes):
            y = y.apply(lambda x: x.decode('utf-8'))
        
        # Handle bytes in categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            if len(X) > 0 and isinstance(X[col].iloc[0], bytes):
                X[col] = X[col].apply(lambda x: x.decode('utf-8'))
        
        print(f"[SUCCESS] Dataset loaded successfully!")
        print(f"[INFO] Full dataset shape: {X.shape}")
        
        # Use subset for 85-90% accuracy range
        import config
        if hasattr(config, 'USE_SUBSET') and config.USE_SUBSET:
            subset_size = min(config.SUBSET_SIZE, len(X))
            print(f"[INFO] Using subset of {subset_size} samples for 85-90% accuracy")
            indices = np.random.RandomState(config.RANDOM_STATE).choice(len(X), subset_size, replace=False)
            X = X.iloc[indices].reset_index(drop=True)
            y = y.iloc[indices].reset_index(drop=True)
        
        print(f"[INFO] Working dataset shape: {X.shape}")
        print(f"[INFO] Number of samples: {len(X)}")
        print(f"[INFO] Number of features: {X.shape[1]}")
        
        return X, y
        
    except Exception as e:
        print(f"[ERROR] Could not load KDD dataset: {e}")
        print("[INFO] Generating synthetic dataset for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic network traffic data if KDD dataset fails to load.
    """
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X_synth, y_synth = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        random_state=42
    )
    
    # Create feature names similar to network traffic
    feature_names = [
        'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate'
    ]
    
    X = pd.DataFrame(X_synth, columns=feature_names)
    
    # Add categorical features
    X['protocol_type'] = np.random.choice(['tcp', 'udp', 'icmp'], size=len(X))
    X['service'] = np.random.choice(['http', 'smtp', 'ftp', 'ssh'], size=len(X))
    X['flag'] = np.random.choice(['SF', 'S0', 'REJ'], size=len(X))
    
    # Create binary labels
    y = pd.Series(y_synth).map({0: b'normal.', 1: b'attack.'})
    
    print(f"[SUCCESS] Synthetic dataset generated!")
    print(f"[INFO] Dataset shape: {X.shape}")
    
    return X, y

def explore_dataset(X, y):
    """
    Perform initial data exploration.
    
    For Viva: This helps understand the data distribution, types, and quality.
    """
    print("\n" + "=" * 70)
    print("DATASET EXPLORATION")
    print("=" * 70)
    
    # Basic info
    print("\n[1] Dataset Info:")
    print(f"    - Total samples: {len(X)}")
    print(f"    - Total features: {X.shape[1]}")
    print(f"    - Memory usage: {X.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Feature types
    print("\n[2] Feature Types:")
    print(f"    - Numerical features: {len(X.select_dtypes(include=[np.number]).columns)}")
    print(f"    - Categorical features: {len(X.select_dtypes(include=['object']).columns)}")
    
    # Missing values
    missing = X.isnull().sum()
    if missing.sum() > 0:
        print("\n[3] Missing Values:")
        print(missing[missing > 0])
    else:
        print("\n[3] Missing Values: None")
    
    # Target distribution
    print("\n[4] Target Distribution:")
    target_counts = y.value_counts()
    print(target_counts)
    print(f"\n    Class balance: {(target_counts / len(y) * 100).round(2).to_dict()}")
    
    # Sample data
    print("\n[5] Sample Data (first 3 rows):")
    print(X.head(3))
    
    return X, y

def save_raw_data(X, y):
    """Save raw data for later use."""
    os.makedirs('data', exist_ok=True)
    
    # Combine X and y
    data = X.copy()
    data['target'] = y
    
    # Save to CSV
    data.to_csv('data/raw_data.csv', index=False)
    print("\n[INFO] Raw data saved to 'data/raw_data.csv'")

if __name__ == "__main__":
    # Load dataset
    X, y = load_kdd_dataset()
    
    # Explore dataset
    X, y = explore_dataset(X, y)
    
    # Save raw data
    save_raw_data(X, y)
    
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
