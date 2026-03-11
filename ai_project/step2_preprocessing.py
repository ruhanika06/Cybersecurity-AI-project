"""
STEP 2: DATA PREPROCESSING
===========================
This module handles missing values, encodes categorical features, and normalizes data.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import config

def load_raw_data():
    """Load the raw data saved from Step 1."""
    print("=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    
    print("\n[INFO] Loading raw data...")
    data = pd.read_csv('data/raw_data.csv')
    
    X = data.drop('target', axis=1)
    y = data['target']
    
    print(f"[SUCCESS] Data loaded: {X.shape}")
    return X, y

def handle_missing_values(X):
    """
    Handle missing values using appropriate strategies.
    
    For Viva: We use median for numerical (robust to outliers) 
    and mode for categorical features.
    """
    print("\n[1] Handling Missing Values...")
    
    missing_before = X.isnull().sum().sum()
    
    if missing_before > 0:
        # Numerical columns: fill with median
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        print(f"    - Filled {missing_before} missing values")
    else:
        print("    - No missing values found")
    
    return X

def encode_categorical_features(X, y):
    """
    Encode categorical features using Label Encoding.
    
    For Viva: Label Encoding converts categories to numbers (0, 1, 2...).
    Alternative: One-Hot Encoding (creates binary columns for each category).
    """
    print("\n[2] Encoding Categorical Features...")
    
    # Encode target variable
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)
    
    # Convert to binary: normal vs attack
    # In KDD, 'normal.' is one class, everything else is attack
    y_binary = np.where(y == 'normal.', 0, 1)
    
    print(f"    - Target classes: {label_encoder_y.classes_}")
    print(f"    - Binary encoding: 0=Normal, 1=Attack")
    
    # Encode categorical features in X
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"    - Encoded '{col}': {len(le.classes_)} unique values")
    
    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/feature_encoders.pkl')
    joblib.dump(label_encoder_y, 'models/target_encoder.pkl')
    
    return X, y_binary, encoders, label_encoder_y

def normalize_features(X_train, X_test):
    """
    Normalize features using StandardScaler.
    
    For Viva: Normalization ensures all features have mean=0, std=1.
    This prevents features with large values from dominating the model.
    """
    print("\n[3] Normalizing Features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"    - Features normalized using StandardScaler")
    print(f"    - Mean: ~0, Std: ~1")
    
    return X_train_scaled, X_test_scaled, scaler

def split_data(X, y):
    """
    Split data into training and testing sets.
    
    For Viva: We use 80-20 split with stratification to maintain class balance.
    """
    print("\n[4] Splitting Data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"    - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"    - Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"    - Stratification: Class balance maintained")
    
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test):
    """Save preprocessed data."""
    print("\n[5] Saving Preprocessed Data...")
    
    # Save training data
    train_data = X_train.copy()
    train_data['target'] = y_train
    train_data.to_csv('data/train_data.csv', index=False)
    
    # Save testing data
    test_data = X_test.copy()
    test_data['target'] = y_test
    test_data.to_csv('data/test_data.csv', index=False)
    
    print("    - Saved to 'data/train_data.csv' and 'data/test_data.csv'")

if __name__ == "__main__":
    # Load raw data
    X, y = load_raw_data()
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Encode categorical features
    X, y, encoders, label_encoder = encode_categorical_features(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Save preprocessed data
    save_preprocessed_data(X_train_scaled, X_test_scaled, y_train, y_test)
    
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
