# Heart Health Risk Assessment System

![Python](https://img.shields.io/badge/Python-3.9+-yellow)
![Machine Learning](https://img.shields.io/badge/ML-Logistic%20Regression-blue)

## Project Overview
The **Heart Health Risk Assessment System** is a professional healthcare analytics application designed to predict the risk of cardiovascular disease based on clinical parameters. This project focuses on leveraging supervised machine learning to provide clinicians with early-warning risk assessments, enabling data-driven patient care.

### Core Objectives
- **Automated Screening:** Rapidly assess patient risk using standard clinical metrics.
- **Explainable AI:** Provide clear insights into key risk factors contributing to the assessment.
- **Clinical Efficiency:** Streamline the diagnostic workflow through a high-performance interactive interface.

---

## System Architecture & ML Pipeline
The system implements a structured medical data pipeline to ensure accuracy and reproducibility.

### 1. Data Processing
- **Input Handling:** Captures 13 critical clinical features through a validated UI.
- **Normalization:** Employs a **StandardScaler** to ensure all features (e.g., Blood Pressure, Cholesterol) are processed on a uniform scale, preventing bias toward high-magnitude features.

### 2. Model Logic
- **Algorithm:** **Logistic Regression** was selected for its balance of high accuracy and clinical interpretability.
- **Inference Process:** 
  1. The pre-trained model loads serialized weights from `logistic_model.pkl`.
  2. The input vector is transformed using the fitted scaler.
  3. The model calculates the risk probability using the sigmoid function, outputting a risk score and categorical classification (HIGH/LOW).

---

## Input-Output Specifications

### Clinical Inputs
| Category | Parameters |
| :--- | :--- |
| **Demographics** | Age, Biological Sex |
| **Vitals** | Resting BP (mm Hg), Serum Cholesterol (mg/dl), Max Heart Rate |
| **Clinical Data** | Chest Pain Type, Fasting Blood Sugar, Resting ECG |
| **Advanced Metrics** | ST Depression (Oldpeak), ST Slope, Major Vessels (0-3), Thalassemia |

### System Outputs
- **Risk Category:** Determinate classification (HIGH or LOW Risk).
- **Confidence Score:** Percentage-based risk probability (e.g., 85.4% Risk).
- **Contributing Factors:** Automated flagging of parameters exceeding medical thresholds (e.g., Cholesterol > 200).

---

## Technical Stack
- **Language:** Python 3.9+
- **ML Engine:** Scikit-learn, NumPy, Pandas
- **Frontend/Dashboard:** Streamlit
- **Model Serialization:** Joblib
- **Deployment Platform:** Streamlit Community Cloud (Publicly Hosted)

---

## Project Structure
```bash
heart-risk-predictor/
├── app.py                 
├── logistic_model.pkl    
├── scaler.pkl            
├── requirements.txt      
└── README.md             
```

---

> [!WARNING]
> **Medical Disclaimer:** This system is for educational and academic purposes only. It does not replace professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for clinical evaluation.

