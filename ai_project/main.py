"""
MAIN EXECUTION SCRIPT
=====================
This script orchestrates the entire ML pipeline from start to finish.

Usage: python main.py
"""

import os
import sys

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models', 'results', 'results/visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_pipeline():
    """Run the complete ML pipeline."""
    print("\n" + "=" * 70)
    print(" " * 15 + "NETWORK TRAFFIC CLASSIFICATION")
    print(" " * 10 + "AI-Based Cybersecurity Project")
    print("=" * 70)
    
    # Create directories
    create_directories()
    
    # Step 1: Data Loading
    print("\n\nExecuting Step 1: Data Loading...")
    import step1_data_loading
    print("\n✓ Step 1 Complete")
    
    # Step 2: Data Preprocessing
    print("\n\nExecuting Step 2: Data Preprocessing...")
    import step2_preprocessing
    print("\n✓ Step 2 Complete")
    
    # Step 3: Feature Selection
    print("\n\nExecuting Step 3: Feature Selection...")
    import step3_feature_selection
    print("\n✓ Step 3 Complete")
    
    # Step 4: Model Training
    print("\n\nExecuting Step 4: Model Training...")
    import step4_model_training
    print("\n✓ Step 4 Complete")
    
    # Step 5: Model Evaluation
    print("\n\nExecuting Step 5: Model Evaluation...")
    import step5_model_evaluation
    print("\n✓ Step 5 Complete")
    
    # Final Summary
    print("\n\n" + "=" * 70)
    print(" " * 20 + "PIPELINE COMPLETED!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  📁 data/")
    print("     - raw_data.csv")
    print("     - train_data.csv")
    print("     - test_data.csv")
    print("\n  📁 models/")
    print("     - logistic_regression.pkl")
    print("     - random_forest.pkl")
    print("     - svm.pkl")
    print("     - selected_features.pkl")
    print("     - scaler.pkl")
    print("\n  📁 results/")
    print("     - evaluation_report.txt")
    print("     - training_summary.csv")
    print("\n  📁 results/visualizations/")
    print("     - correlation_matrix.png")
    print("     - feature_importance.png")
    print("     - confusion_matrices.png")
    print("     - roc_curves.png")
    print("     - metrics_comparison.png")
    print("\n" + "=" * 70)
    print("\nNext Steps:")
    print("  1. Review 'results/evaluation_report.txt' for model performance")
    print("  2. Check visualizations in 'results/visualizations/'")
    print("  3. Prepare for viva using the generated insights")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n\n[!] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
