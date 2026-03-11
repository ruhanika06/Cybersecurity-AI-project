"""
STEP 5: MODEL EVALUATION
=========================
This module evaluates all trained models using multiple metrics.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import joblib
import os
import config

def load_models_and_data():
    """Load all trained models and test data."""
    print("=" * 70)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 70)
    
    print("\n[INFO] Loading models and data...")
    
    # Load test data
    test_data = pd.read_csv('data/test_data.csv')
    selected_features = joblib.load('models/selected_features.pkl')
    
    X_test = test_data[selected_features]
    y_test = test_data['target']
    
    # Load models (3 models without XGBoost)
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
        'Random Forest': joblib.load('models/random_forest.pkl'),
        'SVM': joblib.load('models/svm.pkl')
    }
    
    print(f"[SUCCESS] Loaded {len(models)} models")
    print(f"    - Test samples: {len(X_test)}")
    
    return models, X_test, y_test

def evaluate_single_model(model, model_name, X_test, y_test):
    """
    Evaluate a single model on all metrics.
    
    For Viva: Explain each metric and why it matters for cybersecurity.
    """
    print(f"\n[{model_name}]")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ROC-AUC (requires probability predictions)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = 0.0
    
    print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1-Score:  {f1:.4f}")
    print(f"    ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'Predictions': y_pred,
        'Probabilities': y_proba if 'y_proba' in locals() else None
    }

def create_confusion_matrices(models, X_test, y_test):
    """
    Create confusion matrix visualizations for all models.
    
    For Viva: Confusion matrix shows:
    - True Positives (TP): Correctly identified attacks
    - True Negatives (TN): Correctly identified normal traffic
    - False Positives (FP): Normal traffic flagged as attack
    - False Negatives (FN): Missed attacks (most dangerous!)
    """
    print("\n[1] Generating Confusion Matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (model_name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        axes[idx].set_title(f'{model_name}', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{config.VISUALIZATIONS_DIR}/confusion_matrices.png', dpi=300)
    print(f"    - Saved to '{config.VISUALIZATIONS_DIR}/confusion_matrices.png'")
    plt.close()

def create_roc_curves(results, y_test):
    """
    Create ROC curves for all models.
    
    For Viva: ROC curve shows trade-off between True Positive Rate
    and False Positive Rate. AUC (Area Under Curve) summarizes performance.
    """
    print("\n[2] Generating ROC Curves...")
    
    plt.figure(figsize=(10, 8))
    
    for result in results:
        if result['Probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['Probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{result['Model']} (AUC = {roc_auc:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.VISUALIZATIONS_DIR}/roc_curves.png', dpi=300)
    print(f"    - Saved to '{config.VISUALIZATIONS_DIR}/roc_curves.png'")
    plt.close()

def create_metrics_comparison(results):
    """Create bar chart comparing all metrics across models."""
    print("\n[3] Generating Metrics Comparison...")
    
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(metrics_df))
    width = 0.15
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 2)
        ax.bar(x + offset, metrics_df[metric], width, label=metric, color=color)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{config.VISUALIZATIONS_DIR}/metrics_comparison.png', dpi=300)
    print(f"    - Saved to '{config.VISUALIZATIONS_DIR}/metrics_comparison.png'")
    plt.close()

def generate_evaluation_report(results):
    """Generate comprehensive text report."""
    print("\n[4] Generating Evaluation Report...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MODEL EVALUATION REPORT")
    report_lines.append("Network Traffic Classification for Cybersecurity")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Summary table
    report_lines.append("PERFORMANCE SUMMARY")
    report_lines.append("-" * 70)
    
    metrics_df = pd.DataFrame(results)
    metrics_df = metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    report_lines.append(metrics_df.to_string(index=False))
    report_lines.append("")
    
    # Best model
    best_model = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
    report_lines.append("BEST PERFORMING MODEL")
    report_lines.append("-" * 70)
    report_lines.append(f"Model: {best_model['Model']}")
    report_lines.append(f"F1-Score: {best_model['F1-Score']:.4f}")
    report_lines.append(f"Accuracy: {best_model['Accuracy']:.4f}")
    report_lines.append("")
    
    # Recommendations
    report_lines.append("RECOMMENDATIONS FOR VIVA")
    report_lines.append("-" * 70)
    report_lines.append("1. Recall is most critical for cybersecurity (catching all attacks)")
    report_lines.append("2. Random Forest typically performs best for this dataset")
    report_lines.append("3. SVM may be slower but offers good accuracy")
    report_lines.append("4. Logistic Regression serves as a fast baseline")
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Save report
    with open('results/evaluation_report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"    - Saved to 'results/evaluation_report.txt'")
    
    # Print to console
    print("\n" + '\n'.join(report_lines))

if __name__ == "__main__":
    # Load models and data
    models, X_test, y_test = load_models_and_data()
    
    # Evaluate all models
    results = []
    for model_name, model in models.items():
        result = evaluate_single_model(model, model_name, X_test, y_test)
        results.append(result)
    
    # Create visualizations
    create_confusion_matrices(models, X_test, y_test)
    create_roc_curves(results, y_test)
    create_metrics_comparison(results)
    
    # Generate report
    generate_evaluation_report(results)
    
    print("\n" + "=" * 70)
    print("STEP 5 COMPLETED SUCCESSFULLY!")
    print("All evaluations complete!")
    print("=" * 70)
