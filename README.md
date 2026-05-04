#  Walmart Sales Forecasting — End-to-End ML Project

A complete machine learning pipeline that forecasts weekly sales for 45 Walmart stores using time series models and gradient boosting regression.

---

# Project Overview

This project analyses 143,000+ weekly sales records across 45 Walmart stores (2010–2012) to build a production-ready sales forecasting system. It combines classical time series forecasting with modern machine learning to deliver accurate store-level predictions and actionable business insights.

---

# What This Project Does

- Segments all 45 stores into **High / Mid / Low** revenue tiers
- Performs **Exploratory Data Analysis** — holiday effects, seasonality, macro-economic correlations
- Builds **per-store ARIMA and SARIMA** models with 12-week ahead forecasts and 95% confidence intervals
- Trains and compares **5 ML regression models** on all 45 stores
- Tunes the best model using **RandomizedSearchCV**
- Exports a **production-ready pipeline** (.pkl) and forecast results (JSON/CSV)

---

# Models Used

| Model | R² Score |
|---|---|
| Linear Regression | 0.942 |
| Ridge Regression | 0.944 |
| Decision Tree | 0.961 |
| Random Forest | 0.972 |
| **Gradient Boosting ⭐ Best** | **0.976** |

**Time Series:** ARIMA (AIC grid search) + SARIMA(1,1,1)(1,1,0,52) per store

---

# Key Results

- **Best R² = 0.976** (Gradient Boosting after hyperparameter tuning)
- **RMSE ≈ $42,000** on held-out test data
- **12-week SARIMA forecast** generated per store tier
- **+7% holiday sales premium** identified and modelled

---

# Tech Stack

- **Language:** Python 3.10+
- **Data:** Pandas, NumPy, StandardScaler
- **Time Series:** Statsmodels (ARIMA, SARIMA), ADF test
- **ML:** Scikit-learn (GBM, Random Forest, Ridge, Decision Tree, RandomizedSearchCV)
- **Visualisation:** Matplotlib, Seaborn
- **Export:** Joblib (.pkl), JSON, CSV

---

# Files

| File | Description |
|---|---|
| `P1_Walmart_Forecasting_FINAL.ipynb` | Full notebook — EDA, models, results |
| `Walmart_Forecasting_Corrected_Report.docx` | Written project report |

---

# Dataset

- **Source:** Kaggle / UCI ML Repository
- **File:** `Walmart_DataSet.csv`
- **Size:** 143,000 rows × 8 features
- **Target:** `Weekly_Sales` (weekly revenue in USD)

---

# Author

**Pramod K O**  
[LinkedIn Profile URL:linkedin.com/in/pramod-k-o-775b31306 ] · [GitHub Profile URL:pramodkopramodko-commits]
