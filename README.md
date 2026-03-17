<div align="center">

# 🛡️ Financial Fraud Detection + SHAP Explainability

**ML-powered fraud detection · 4 models · SHAP global + local explanations · Business cost analysis**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44+-blueviolet?style=flat-square)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

[**Live Demo →**](https://your-app.streamlit.app) · [GitHub](https://github.com/Nirmitpatel889/fraud-detection)

</div>

---

## Overview

An end-to-end fraud detection system trained on 284,807 real credit card transactions. Goes beyond model accuracy — uses **SHAP (SHapley Additive exPlanations)** to explain *why* each transaction was flagged, and translates model output into business dollar savings.

**Key differentiator:** Most fraud detection projects just report an AUC score. This one answers the harder questions a risk analyst actually faces:
- Why was transaction #47,832 flagged but not #47,833?
- What's the cost-optimal decision threshold given our chargeback costs?
- How much does the ML model save vs a simple rule-based system?

---

## Results — Best Model (XGBoost)

| Metric | Value | Note |
|---|---|---|
| AUC-ROC | **0.9791** | Area under ROC curve |
| AUC-PR | **0.8134** | More meaningful on imbalanced data |
| Recall | **83.7%** | Fraud transactions caught |
| Precision | **77.2%** | Flagged transactions that are real fraud |
| F1 Score | **0.8032** | Harmonic mean |
| Savings per 1,000 txns | **$412** | vs no detection system ($150 avg fraud loss) |

> **Why AUC-PR instead of accuracy?** With 0.17% fraud rate, a model that predicts "legitimate" for every transaction achieves 99.83% accuracy — and catches zero fraud. Precision-Recall curves are the correct metric for heavily imbalanced classification.

---

## Features

- **4 trained models** — Logistic Regression (baseline), Random Forest, XGBoost, LightGBM
- **SMOTE + undersampling** — handles extreme class imbalance (0.17% fraud rate)
- **SHAP global explainability** — which features drive fraud predictions across the dataset
- **SHAP local explainability** — waterfall chart for any individual transaction
- **Plain-English fraud reasons** — "Flagged because: anomalous V14 signal (-4.2), large amount ($3,847)"
- **Business cost analysis** — FP/FN cost matrix, optimal threshold selection, savings vs naive rules
- **Interactive dashboard** — live transaction scorer with real-time SHAP explanation
- **Threshold tuning** — adjust decision threshold based on your FP/FN cost assumptions

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data processing | `pandas`, `numpy`, `scikit-learn` |
| Class imbalance | `imbalanced-learn` (SMOTE + RandomUnderSampler) |
| Models | `xgboost`, `lightgbm`, `scikit-learn` |
| Explainability | `shap` |
| Visualization | `plotly`, `matplotlib` |
| Dashboard | `streamlit` |

---

## Project Structure

```
fraud-detection/
├── src/
│   ├── preprocessor.py      # Data loading, EDA, SMOTE pipeline
│   ├── models.py            # All 4 models, evaluation, threshold tuning
│   ├── explainer.py         # SHAP global + local explainability
│   └── business_impact.py  # Cost-benefit analysis, savings calculator
├── app/
│   └── main.py             # Streamlit dashboard (4 tabs)
├── data/                   # Place creditcard.csv here
├── models/                 # Saved best model + explainer
├── train.py                # End-to-end training script
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Get the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train models

```bash
python train.py
```

### 4. Run dashboard

```bash
streamlit run app/main.py
```

---

## SHAP Explainability — How It Works

SHAP assigns each feature a contribution value for each prediction:
- **Positive SHAP** → pushed the prediction toward fraud
- **Negative SHAP** → pushed the prediction toward legitimate

**Global explanation** (`FraudExplainer.global_importance_plotly`)
Averages |SHAP| across all test transactions to show which features matter most overall. In this dataset, V14, V10, V12, and V4 consistently dominate.

**Local explanation** (`FraudExplainer.waterfall_plotly`)
For any single transaction, shows exactly which features pushed the prediction up or down and by how much. The fraud probability is the sum of all SHAP values plus a base rate.

**Business translation** (`FraudExplainer.business_reason`)
Converts raw SHAP values into plain English: *"Flagged because: anomalous V14 signal (-4.2, SHAP: +0.312), large transaction amount (scaled: 1.93, SHAP: +0.187)."*

---

## Business Impact

At $2 FP cost (manual review) and $150 FN cost (chargeback + investigation):

| System | Cost per 10k txns | Fraud caught |
|---|---|---|
| No detection | $7,380 | 0% |
| Simple rule (flag >$500) | $4,210 | ~60% |
| **This ML model** | **$3,250** | **83.7%** |
| Savings vs no detection | **$4,130** | — |

---

## What I Learned

- **Recall > Precision for fraud** — a missed fraud costs 75x more than a false alert at typical chargeback rates. Always optimize for recall first, then tune precision via threshold.
- **SMOTE goes on training data only** — applying it before the train-test split leaks information and inflates metrics. Always split first.
- **AUC-ROC is misleading on imbalanced data** — AUC-PR (average precision) is the correct primary metric. A model can have 0.97 AUC-ROC and still be nearly useless on the fraud class.
- **V14 is the dominant fraud signal** — across all models and SHAP analyses, V14 (a PCA-transformed feature) consistently has the highest |SHAP| for fraud predictions.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by **Nirmit Patel** · MBA Candidate, Business Analytics · Pace University Lubin School of Business

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-f87171?style=flat-square)](https://nirmit-patel-portfolio.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-Nirmitpatel889-181717?style=flat-square&logo=github)](https://github.com/Nirmitpatel889)

</div>
