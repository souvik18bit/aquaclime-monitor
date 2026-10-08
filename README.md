# 🐟 PondIQ — Fish Pond Health AI System

> **An affordable real-time water monitoring system that alerts fish farmers before low oxygen levels cause fish deaths.**

---

## 📌 Overview

PondIQ is a complete end-to-end machine learning pipeline for aquaculture water quality monitoring. It predicts dangerous dissolved oxygen (DO) drops, detects sensor anomalies, and decides whether to activate pond aerators — all in real time.

---

## 🧠 ML Models

| Model | Algorithm | Purpose | Key Metric |
|---|---|---|---|
| **DO Forecaster** | LSTM (TensorFlow) | Predict next DO value | MAE / RMSE (mg/L) |
| **Anomaly Detector** | Isolation Forest | Flag unusual readings | Precision / Recall |
| **Aerator Classifier** | Random Forest + XGBoost | Activate aerator? ON/OFF | F1-Score / AUC |

---

## 📁 Project Structure

```
PondIQ/
├── data/
│   └── pond_sensor_data.csv       # Generated synthetic dataset
├── models/
│   ├── lstm_forecaster.py         # LSTM DO forecasting model
│   ├── anomaly_detector.py        # Isolation Forest anomaly detection
│   └── aerator_classifier.py      # RF + XGB aerator classifier
├── api/
│   └── app.py                     # FastAPI REST API
├── dashboard/
│   └── dashboard.py               # Streamlit dashboard
├── utils/
│   └── data_generator.py          # Synthetic data generator
├── train_all.py                   # Master training script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
cd PondIQ
python train_all.py
# Skip LSTM if no GPU / TensorFlow:
python train_all.py --skip-lstm
```

### 3. Launch the Dashboard
```bash
streamlit run dashboard/dashboard.py
```

### 4. Start the API
```bash
cd api
uvicorn app:app --reload
# Swagger UI → http://localhost:8000/docs
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/latest` | Latest sensor reading |
| `POST` | `/predict/do` | LSTM DO forecast (needs ≥16 readings) |
| `POST` | `/predict/anomaly` | Anomaly detection (needs ≥8 readings) |
| `POST` | `/predict/aerator` | Aerator ON/OFF decision (needs ≥16 readings) |

### Example Request
```json
POST /predict/aerator
{
  "readings": [
    {"dissolved_oxygen": 4.2, "temperature": 29.1, "ph": 7.3,
     "turbidity": 22.5, "hour": 3, "month": 7},
    ...16 rows total
  ]
}
```

### Example Response
```json
{
  "aerator_on": true,
  "confidence": 0.89,
  "label": "🟢 AERATOR ON — Activate Now!",
  "timestamp": "2025-06-01T03:15:00"
}
```

---

## 📊 Sensor Features

| Feature | Unit | Normal Range | Crisis Threshold |
|---|---|---|---|
| Dissolved Oxygen (DO) | mg/L | 6–10 | < 4.0 (critical) |
| Temperature | °C | 24–32 | > 34 |
| pH | — | 7.0–8.5 | < 6.5 or > 9.0 |
| Turbidity | NTU | 10–40 | > 70 |

---

## 📈 Key Metrics (as per Lean Canvas)

| Metric | Target |
|---|---|
| Fish Mortality Reduction | > 70% fewer crisis events missed |
| DO Forecast MAE | < 0.5 mg/L |
| Alert Response Time | < 5 seconds |
| Aerator Activation Accuracy | > 90% F1-Score |

---

## 🗺️ Roadmap

- [ ] Real sensor integration (ESP32 / Raspberry Pi via MQTT)
- [ ] Multi-pond support
- [ ] WhatsApp / SMS alert integration (Twilio)
- [ ] Mobile app (React Native)
- [ ] Federated learning across farms

---

## 👥 Target Users

- Fish Farmers / Aquaculture Farms
- Fisheries Departments
- Marine & Environmental NGOs
- Water Treatment & Management Companies

---

## 📜 License
MIT — Free to use, modify, and deploy.

---

*Built as part of PondIQ — "Fitbit + Smart Automation for Aquaculture"*
