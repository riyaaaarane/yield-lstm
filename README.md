# 🌾 Crop Yield Prediction using LSTM and Time-Series Decomposition

This repository presents a **data-driven analysis of agricultural productivity** across five villages in Karnataka—**Dhawaleshwar, Kulali, Mahalingpur, Munyal, and Teradal**—over a six-year period (2007–2012).  
The study leverages **climate data and vegetation indices** to model and forecast crop yields, providing insights into how environmental variability affects agricultural output.

---

## 📌 Project Overview
Agricultural productivity is shaped by complex interactions between **rainfall, temperature, and vegetation dynamics**.  
This project integrates **statistical time-series analysis** with **deep learning (LSTM)** to:
- Analyze climatic variables and vegetation indices.
- Perform seasonal decomposition to identify trend, seasonality, and residual effects.
- Forecast crop yields using **Long Short-Term Memory (LSTM)** networks.
- Evaluate model performance using standard regression metrics.

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
1. **Data Preprocessing**
   - Missing values handled and temporal alignment performed.
   - Features normalized using `MinMaxScaler`.
   - Data reshaped into supervised format:  
     X ∈ R^(N × T × F),  y ∈ R^(N × 1)  
     where:
     - \(N\) = number of samples  
     - \(T\) = lookback period  
     - \(F\) = number of features  

2. **Statistical Analysis**
   - Seasonal decomposition applied to precipitation, temperature, LAI, FAPAR, and yield.

3. **Model Architecture**
   - Two stacked **LSTM layers** (50 units each).
   - One **Dense layer** (1 neuron) for yield prediction.
   - Total trainable parameters: **31,451**.

4. **Training & Optimization**
   - Optimizer: **Adam**  
   - Loss: **Mean Squared Error (MSE)**  
   - Training: 10 epochs, batch size = 32  
   - Validation split = 20%  

---

## 📊 Performance Metrics
The model is evaluated using the following metrics:

**1. Mean Squared Error (MSE):**  
MSE = (1/n) * Σ (yᵢ - ŷᵢ)²  

**2. Root Mean Squared Error (RMSE):**  
RMSE = √[ (1/n) * Σ (yᵢ - ŷᵢ)² ]  

**3. Mean Absolute Error (MAE):**  
MAE = (1/n) * Σ |yᵢ - ŷᵢ|  

**4. Coefficient of Determination (R²):**  
R² = 1 - [ Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)² ]  

**5. Mean Absolute Percentage Error (MAPE):**  
MAPE = (100/n) * Σ |(yᵢ - ŷᵢ) / yᵢ|  

Where:  
- yᵢ = actual value  
- ŷᵢ = predicted value  
- ȳ = mean of actual values  
- n = number of samples  


---

## 📈 Results
- Seasonal patterns strongly influence yield variations.
- Rainfall distribution and vegetation indices (LAI, FAPAR) are key predictors.
- LSTM outperforms traditional statistical baselines in yield forecasting.

---


## 📦 Requirements
Install the dependencies before running the code:

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels tensorflow
