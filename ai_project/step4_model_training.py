"""
STEP 4: MODEL TRAINING
=======================
This module trains multiple supervised learning models.


"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import time
import os
import config

def load_data_for_training():
    """Load preprocessed data and selected features."""
    print("=" * 70)
    print("STEP 4: MODEL TRAINING")
    print("=" * 70)
    
    print("\n[INFO] Loading data...")
    
    # Load train and test data
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')
    
    # Load selected features
    selected_features = joblib.load('models/selected_features.pkl')
    
    # Extract features and target
    X_train = train_data[selected_features]
    y_train = train_data['target']
    X_test = test_data[selected_features]
    y_test = test_data['target']
    
    print(f"[SUCCESS] Data loaded")
    print(f"    - Training samples: {len(X_train)}")
    print(f"    - Testing samples: {len(X_test)}")
    print(f"    - Features: {len(selected_features)}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression model.
    
    For Viva: Logistic Regression is a linear classifier that predicts
    probability using the sigmoid function. Good baseline model.
    """
    print("\n[1] Training Logistic Regression...")
    
    model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"    - Training completed in {training_time:.2f} seconds")
    
    # Save model
    joblib.dump(model, 'models/logistic_regression.pkl')
    print(f"    - Model saved to 'models/logistic_regression.pkl'")
    
    return model, training_time

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    
    For Viva: Random Forest uses bagging - creates multiple decision trees
    on random subsets of data and features, then averages predictions.
    Reduces overfitting compared to single decision tree.
    """
    print("\n[2] Training Random Forest...")
    
    model = RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"    - Training completed in {training_time:.2f} seconds")
    print(f"    - Number of trees: {model.n_estimators}")
    
    # Save model
    joblib.dump(model, 'models/random_forest.pkl')
    print(f"    - Model saved to 'models/random_forest.pkl'")
    
    return model, training_time

def train_svm(X_train, y_train):
    """
    Train Support Vector Machine model.
    
    For Viva: SVM finds the hyperplane that maximizes the margin between
    classes. RBF kernel allows non-linear decision boundaries.
    """
    print("\n[3] Training Support Vector Machine...")
    
    # Use a subset for SVM if dataset is too large (SVM is slow)
    if len(X_train) > 10000:
        print(f"    - Using subset of 10000 samples for faster training")
        indices = np.random.choice(len(X_train), 10000, replace=False)
        X_train_subset = X_train.iloc[indices]
        y_train_subset = y_train.iloc[indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train
    
    model = SVC(**config.SVM_PARAMS, probability=True)
    
    start_time = time.time()
    model.fit(X_train_subset, y_train_subset)
    training_time = time.time() - start_time
    
    print(f"    - Training completed in {training_time:.2f} seconds")
    print(f"    - Kernel: {model.kernel}")
    
    # Save model
    joblib.dump(model, 'models/svm.pkl')
    print(f"    - Model saved to 'models/svm.pkl'")
    
    return model, training_time


def save_training_summary(training_times):
    """Save training summary."""
    summary = pd.DataFrame({
        'Model': list(training_times.keys()),
        'Training Time (seconds)': list(training_times.values())
    })
    
    os.makedirs('results', exist_ok=True)
    summary.to_csv('results/training_summary.csv', index=False)
    print(f"\n[INFO] Training summary saved to 'results/training_summary.csv'")

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data_for_training()
    
    # Train all models
    training_times = {}
    
    lr_model, lr_time = train_logistic_regression(X_train, y_train)
    training_times['Logistic Regression'] = lr_time
    
    rf_model, rf_time = train_random_forest(X_train, y_train)
    training_times['Random Forest'] = rf_time
    
    svm_model, svm_time = train_svm(X_train, y_train)
    training_times['SVM'] = svm_time
    
    # Save summary
    save_training_summary(training_times)
    
    print("\n" + "=" * 70)
    print("STEP 4 COMPLETED SUCCESSFULLY!")
    print("All models trained and saved!")
    print("=" * 70)
