"""
Configuration file for the Network Traffic Classification project.
Contains all hyperparameters, paths, and settings.
"""

# Data Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Dataset size (reduce for lower accuracy - CRITICAL FOR 85-90% ACCURACY)
USE_SUBSET = True  # Set to True to use smaller dataset
SUBSET_SIZE = 10000  # Very small subset for 85-90% accuracy

# Feature Selection (reduced to lower accuracy)
CORRELATION_THRESHOLD = 0.98  # Very high threshold
TOP_N_FEATURES = 3  # Minimal features for 85-90% accuracy

# Model Hyperparameters (Extremely reduced for 85-90% accuracy)
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 30,  # Extremely low iterations
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'C': 0.005,  # Extremely strong regularization
    'solver': 'lbfgs',
    'tol': 0.01  # Higher tolerance for early stopping
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 5,  # Minimal trees
    'max_depth': 2,  # Extremely shallow
    'min_samples_split': 40,  # Very high split requirement
    'min_samples_leaf': 20,  # Very large leaf size
    'max_features': 2,  # Limit features per split
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

SVM_PARAMS = {
    'kernel': 'linear',  # Simpler kernel
    'C': 0.2,  # Moderate regularization
    'random_state': RANDOM_STATE,
    'max_iter': 1000,
    'tol': 0.01  # Higher tolerance
}

# Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
VISUALIZATIONS_DIR = 'results/visualizations'
