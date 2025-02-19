# DHL2-Task03-Disease-Diagnosis-Prediction-Diabetes-

![GitHub](https://img.shields.io/badge/Python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![GitHub](https://img.shields.io/badge/Model-Gradient_Boosting-green)
![GitHub](https://img.shields.io/badge/Dataset-PIMA_Diabetes-red)

A machine learning pipeline for diabetes prediction with model interpretation using SHAP values.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Model Training](#model-training)
- [SHAP Interpretation](#shap-interpretation)
- [Results](#results)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview
This project predicts diabetes risk using medical data. It includes:
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model comparison (GBM, SVM, Neural Network)
- SHAP-based model interpretation
- Clinical insights for healthcare professionals

---

## Dataset
**Source**: [PIMA Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database)  
**Features**:  
| Feature | Description | 
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Diabetes likelihood score |
| Age | Age (years) |

**Target Variable**: `Outcome` (1 = Diabetic, 0 = Non-Diabetic)

---

## Code Structure
disease-diagnosis/
├── data/
│   ├── diabetes.csv           # Raw dataset
│   └── processed/            # Processed data
├── models/                    # Saved models
├── notebooks/
│   └── Diabetes_Prediction.ipynb  # Main analysis notebook
├── scripts/
│   ├── preprocess.py         # Data cleaning script
│   └── train_model.py        # Model training script
├── requirements.txt          # Dependency list
└── README.md                 # Project documentation

## Model Training
### Algorithms Implemented
```python
# Gradient Boosting
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Support Vector Machine
SVC(
    kernel='rbf',
    C=1.0,
    probability=True,
    random_state=42
)

# Neural Network
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

```
## Model Training
### Algorithms Implemented
```python
# Gradient Boosting
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Support Vector Machine
SVC(
    kernel='rbf',
    C=1.0,
    probability=True,
    random_state=42
)

# Neural Network
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
```
## Training Process
```python
# Example training snippet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,
    random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model.fit(X_train_scaled, y_train)
```
## SHAP Interpretation
![SHAP Interpretation](assets/model.png)
```python
import shap

# Generate SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_scaled)

# Visualize
shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)
```
## Results
### Performance Metrics
**Results**
Performance Metrics
| Model	Accuracy	| F1-Score	| AUC-ROC |
| Gradient Boosting	| 78.3% | 	0.72	| 0.82 |
| SVM	| 76.1%	| 0.70	| 0.80 |
| Neural Network	| 74.5%	| 0.68	| 0.78 |

# Key Insights
## Critical Risk Factors:
**1** Glucose levels (>140 mg/dL increase risk by 4×).

**2** BMI (>30 correlates with 3.2× higher risk).

**3** Age (Patients >35 years have 2.8× higher risk).
## Prevention Strategies:
**1**Regular glucose monitoring for high-BMI patients

**2** Weight management programs

**3** Annual screenings for patients over 35

## Model Trustworthiness:
SHAP values confirm clinically recognized risk factors

Consistent with medical literature on diabetes
