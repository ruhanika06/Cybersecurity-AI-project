"""
STEP 3: FEATURE SELECTION
==========================
This module performs correlation analysis and feature importance ranking.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import os
import config

def load_preprocessed_data():
    """Load preprocessed data from Step 2."""
    print("=" * 70)
    print("STEP 3: FEATURE SELECTION")
    print("=" * 70)
    
    print("\n[INFO] Loading preprocessed data...")
    
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    
    print(f"[SUCCESS] Data loaded: {X_train.shape}")
    return X_train, y_train

def correlation_analysis(X_train):
    """
    Analyze feature correlations to identify redundant features.
    
    For Viva: Highly correlated features provide redundant information.
    Removing them simplifies the model without losing predictive power.
    """
    print("\n[1] Correlation Analysis...")
    
    # Calculate correlation matrix
    corr_matrix = X_train.corr().abs()
    
    # Find highly correlated features
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > config.CORRELATION_THRESHOLD)]
    
    print(f"    - Total feature pairs analyzed: {len(X_train.columns) * (len(X_train.columns) - 1) // 2}")
    print(f"    - Highly correlated features (>{config.CORRELATION_THRESHOLD}): {len(to_drop)}")
    
    if to_drop:
        print(f"    - Features to remove: {to_drop}")
    
    # Create visualization
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{config.VISUALIZATIONS_DIR}/correlation_matrix.png', dpi=300)
    print(f"    - Saved correlation matrix to '{config.VISUALIZATIONS_DIR}/correlation_matrix.png'")
    plt.close()
    
    return to_drop

def feature_importance_analysis(X_train, y_train, features_to_drop):
    """
    Calculate feature importance using Random Forest.
    
    For Viva: Random Forest calculates importance based on how much each
    feature decreases impurity (Gini) across all trees.
    """
    print("\n[2] Feature Importance Analysis...")
    
    # Remove highly correlated features first
    X_train_reduced = X_train.drop(columns=features_to_drop, errors='ignore')
    
    # Train Random Forest for feature importance
    print("    - Training Random Forest for feature importance...")
    rf = RandomForestClassifier(n_estimators=50, random_state=config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_reduced, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'feature': X_train_reduced.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n    Top {min(10, len(importances))} Most Important Features:")
    print(importances.head(10).to_string(index=False))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_n = min(config.TOP_N_FEATURES, len(importances))
    top_features = importances.head(top_n)
    
    plt.barh(range(top_n), top_features['importance'].values)
    plt.yticks(range(top_n), top_features['feature'].values)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{config.VISUALIZATIONS_DIR}/feature_importance.png', dpi=300)
    print(f"\n    - Saved feature importance plot to '{config.VISUALIZATIONS_DIR}/feature_importance.png'")
    plt.close()
    
    return importances

def select_features(X_train, importances, features_to_drop):
    """
    Select top N features based on importance.
    
    For Viva: We select features that contribute most to prediction accuracy.
    """
    print("\n[3] Selecting Top Features...")
    
    # Remove correlated features
    X_train_reduced = X_train.drop(columns=features_to_drop, errors='ignore')
    
    # Select top N features
    top_features = importances.head(config.TOP_N_FEATURES)['feature'].tolist()
    X_train_selected = X_train_reduced[top_features]
    
    print(f"    - Original features: {X_train.shape[1]}")
    print(f"    - After removing correlated: {X_train_reduced.shape[1]}")
    print(f"    - Final selected features: {len(top_features)}")
    print(f"\n    Selected features: {top_features}")
    
    return X_train_selected, top_features

def save_selected_features(selected_features):
    """Save the list of selected features."""
    import joblib
    
    joblib.dump(selected_features, 'models/selected_features.pkl')
    print(f"\n[INFO] Selected features saved to 'models/selected_features.pkl'")

if __name__ == "__main__":
    # Load data
    X_train, y_train = load_preprocessed_data()
    
    # Correlation analysis
    features_to_drop = correlation_analysis(X_train)
    
    # Feature importance
    importances = feature_importance_analysis(X_train, y_train, features_to_drop)
    
    # Select features
    X_train_selected, selected_features = select_features(X_train, importances, features_to_drop)
    
    # Save selected features
    save_selected_features(selected_features)
    
    print("\n" + "=" * 70)
    print("STEP 3 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
