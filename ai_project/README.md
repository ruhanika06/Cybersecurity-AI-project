# AI Network Traffic Classifier

## 📁 Project Files

This folder contains all the core Python files and documentation for the AI-based Network Traffic Classification project.

### 🐍 Python Files

- **`main.py`** - Main execution script (run this!)
- **`config.py`** - Configuration and hyperparameters
- **`step1_data_loading.py`** - Data loading and exploration
- **`step2_preprocessing.py`** - Data preprocessing
- **`step3_feature_selection.py`** - Feature selection
- **`step4_model_training.py`** - Model training (3 models)
- **`step5_model_evaluation.py`** - Model evaluation
- **`requirements.txt`** - Python dependencies

### 📚 Documentation

- **`README.md`** - This file
- **`QUICK_START.md`** - Quick start guide (if available)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Project
```bash
python main.py
```

---

## 📊 What This Project Does

1. **Loads** KDD Cup 99 dataset (network traffic data)
2. **Preprocesses** data (handles missing values, encoding, scaling)
3. **Selects** best features using correlation and importance analysis
4. **Trains** 3 machine learning models:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine (SVM)
5. **Evaluates** models using multiple metrics
6. **Generates** visualizations and reports

---

## 📈 Expected Results

- **Accuracy**: 85-90%
- **Best Model**: Random Forest
- **Output**: Models, visualizations, and evaluation report

---

---

**Ready to run!** Just execute `python main.py` 🎉
