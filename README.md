# 🏦 Bank Term Deposit Prediction (Kaggle Playground S5E8)

This repository contains my solution for the **Kaggle Playground Series - Season 5, Episode 8** competition.  
The task is a **binary classification problem**: predict whether a client will subscribe to a term deposit.


---

## 📦 Libraries Used
- **Data Handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Preprocessing**: LabelEncoder, RobustScaler, category_encoders
- **Models**: RandomForestClassifier, XGBoost
- **Evaluation**: roc_auc_score, classification_report, confusion_matrix

---

## 📊 Dataset Overview
- **Train set**: 750,000 rows, 18 columns  
- **Test set**: 250,000 rows, 17 columns  
- **Target variable**: `y` (0 = No, 1 = Yes)  

**Target distribution (imbalanced):**
- `0` → 87.9%  
- `1` → 12.1%  

---

## 🔍 Exploratory Data Analysis (EDA)
- No missing values or duplicates.  
- Negative balances are expected in banking datasets (overdrafts).  
- **Duration** can leak the target → dropped from training & test sets.  

**Correlation Heatmap Findings:**
- No numeric feature strongly correlated with `y`.  
- `pdays` and `previous` moderately correlated (0.56).  
- Suggests categorical features & nonlinear models are more important.  

---

## ⚙️ Preprocessing
- **Encoding**:  
  - Frequency Encoding → `job`, `month`, `poutcome`  
  - Label Encoding → `default`, `housing`, `loan`  
  - One-Hot Encoding → `marital`, `education`, `contact`  

- **Scaling**:  
  - Applied **RobustScaler** to reduce the effect of outliers.  

- **Column Alignment**: Ensured consistent feature sets across train/val/test.  

---

## 🧠 Models
### 1. Random Forest
- **AUC (Validation)**: ~0.962  
- **Classification Report**:  
  - Precision (Class 1): ~0.77  
  - Recall (Class 1): ~0.63  
  - F1-score (Class 1): ~0.69  
  - Accuracy: ~0.93  

### 2. XGBoost (Final Model)
- **AUC (Validation)**: ~0.967  
- **Classification Report**:  
  - Precision (Class 1): ~0.77  
  - Recall (Class 1): ~0.67  
  - F1-score (Class 1): ~0.72  
  - Accuracy: ~0.94  

---

## 📈 Results
- Both models performed well, but **XGBoost achieved higher AUC and better recall** on the minority class (`y=1`).  
- **Final Model**: XGBoost  
- **ROC Curve**: XGBoost outperformed Random Forest.  
- **Confusion Matrix**: Shows balanced precision-recall tradeoff.  

---

## 🚀 Submission
Used **XGBoost probabilities** for final Kaggle submission.  

**Submission file format:**
```csv
id,y
750000,0.05
750001,0.12
750002,0.30
...
```
---

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearch / Optuna).
- Try SMOTE / class weights for imbalance.
- Feature engineering on categorical variables.
- Ensemble of XGBoost + LightGBM + CatBoost.

---
