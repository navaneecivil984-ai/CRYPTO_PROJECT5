# 🚀 CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning

CryptoCast is an interactive, multi-horizon financial forecasting dashboard built using Python, Streamlit, and TensorFlow. The system leverages advanced deep learning sequence architectures (CNN, LSTM, RNN, and Transformers) trained on historical data to predict future Bitcoin (BTC) values across multiple strategic time horizons.

---

## 📌 Problem Statement & Business Use Cases
Bitcoin prices exhibit massive volatility driven by high-frequency market mechanics, making traditional linear and statistical metrics unreliable for time-series projections. 

This system designs, evaluates, and provisions sequential deep learning networks capable of extracting non-linear temporal behaviors from multi-variable feature vectors. It addresses core requirements across multiple modern domains:
* **Algorithmic Trading Platforms:** Generates alpha signals across short-term structures.
* **Risk Management Systems:** Provides forward volatility signals to preserve capital matrices.
* **Investment Analytics Dashboards:** Delivers automated predictive insights for asset allocation.

---

## 📊 Dataset Structure
The models consume historical data matrices spanning daily trading metrics with 6 fundamental input feature channels:

| Column Name | Description |
| :--- | :--- |
| `Date` | Trading day index (DD-MM-YYYY) |
| `Price` | Closing value in USD (**Target Feature Vector**) |
| `Open` | Daily market introduction value |
| `High` | Daily ceiling threshold valuation |
| `Low` | Daily floor threshold valuation |
| `Vol.` | Trade volume total (standardized scaling applied) |
| `Change %` | Day-over-day price delta percentage value |

---

## 🏆 Model Benchmarking Performance
The models were configured, optimized, and evaluated over a fixed validation matrix block. Performance tracking clearly shows **1D-CNN** emerging as the optimal performer:

| Model Architecture | MAE (Normalized) | RMSE (Normalized) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **🥇 CNN (1D)** | **0.023971** | **0.034115** | **4.94%** |
| 🥈 LSTM | 0.083501 | 0.114252 | 14.70% |
| 🥉 RNN | 0.140156 | 0.188045 | 24.59% |
| ❌ Transformer | 0.321661 | 0.392678 | 59.97% |

### ⚖️ Architectural Insights:
* **1D-CNN:** Achieved superior temporal feature extraction. Its localized pooling kernels successfully mapped complex transactional interactions within a 60-day historical window.
* **LSTM:** Captured broader price directionality well, but exhibited trailing offset gaps near sudden volatility peaks.
* **RNN:** Suffered from classic backpropagation gradient degradation over extended step arrays.
* **Transformer:** High validation variance indicates a need for heavier hyperparameter tuning to manage abrupt cryptocurrency market shifts smoothly.

---

## 🛠️ Project Architecture & Layout
The dashboard transitions away from cluttered subpage routes to an optimized, unified runtime canvas layout:

```text
├── Bitcoin Historical Data.csv   # Cleaned training timeline dataset
├── model_comparison.csv          # Evaluation benchmarking data array
├── cnn_model.h5                  # Pre-trained CNN weights matrix 
├── lstm_model.h5                 # Pre-trained LSTM weights matrix
├── rnn_model.h5                  # Pre-trained RNN weights matrix
├── transformer_model.keras       # Native Keras v3 Transformer graph schema
├── project5crypto.ipynb          # Model training & optimization notebook
└── app.py                        # Streamlit web interface engine script
