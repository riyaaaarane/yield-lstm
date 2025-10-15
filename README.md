# 🌾 Crop Yield Prediction using Bidirectional LSTM & Machine Learning

This repository presents a **data-driven analysis of agricultural productivity** across five villages in Karnataka—**Dhawaleshwar, Kulali, Mahalingpur, Munyal, and Teradal**—over a six-year period (2007–2012).  
The study leverages **climate data, vegetation indices, and crop yield data** to forecast agricultural output using both **traditional ML models** and **deep learning (Bidirectional LSTM)**.

---

## 📌 Project Overview
Agricultural productivity is influenced by **rainfall, temperature, and vegetation dynamics**.  
This project explores:

- **Statistical & feature engineering approaches:** lag features, rolling means.
- **Machine learning models:** Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, KNN, XGBoost.
- **Deep learning:** Bidirectional LSTM for sequential data.
- **Model evaluation:** Compare performance across models using R², MAE, and RMSE.

---

## 🗂️ Data Sources
- **Google Earth Engine (GEE):**
  - Precipitation (mm)
  - Temperature (°C)
  - Leaf Area Index (LAI)
  - Fraction of Absorbed Photosynthetically Active Radiation (FAPAR)
- **Crop Yield Data:** Village-level datasets (2007–2012)

---

## ⚙️ Methodology

### 1. Stationarity & ADF Test
- Performed **Augmented Dickey-Fuller (ADF) test** on all key variables: precipitation, temperature, LAI, FAPAR, and yield.
- Checked for **stationarity** in time-series data.
- If non-stationary, **differencing** or transformations were applied.
- This ensures better performance for models sensitive to temporal trends (especially LSTM).

### 2. Data Preprocessing
- Handle missing values.
- Generate **lag features** (`1, 2, 7` days) for short-term dependencies.
- Generate **rolling mean features** (`7, 30` days) for climate variables.
- One-hot encode categorical villages.
- Scale features using `MinMaxScaler`.
- Prepare sequences for LSTM with a **90-day lookback**.

### 3. Baseline Machine Learning Models
- **Linear Regression:** R² ≈ 0.994  
- **Ridge Regression:** R² ≈ 0.993  
- **Lasso Regression:** R² ≈ 0.994  
- **Elastic Net Regression:** R² ≈ 0.984  
- **Random Forest Regressor:** R² ≈ 0.847  
- **K-Nearest Neighbors:** R² ≈ 0.172  
- **XGBoost Regressor:** R² ≈ 0.493  

> Linear models performed exceptionally well due to strong correlations and low noise in village-level yield data.

### 4. Bidirectional LSTM Model
- **Architecture**:
  - BiLSTM (128 units, return sequences)
  - Dropout 0.2
  - BiLSTM (256 units)
  - Dropout 0.2
  - Dense layer (1 neuron for yield)
- Optimizer: Adam (lr=0.004, clipnorm=1.0)
- Loss: Mean Squared Error
- Early stopping: patience=30 epochs

### 4. Training
- Epochs: 100
- Batch size: 32
- Validation split: 20%

### 5. Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

---

> The Bidirectional LSTM captures **temporal patterns** more effectively, while linear models are strong baselines for small, structured datasets.





pip install numpy pandas matplotlib scikit-learn statsmodels tensorflow
