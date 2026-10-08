"""
PondIQ — FastAPI REST API
Exposes three endpoints:
  POST /predict/do        → LSTM DO forecast
  POST /predict/anomaly   → Isolation Forest anomaly check
  POST /predict/aerator   → RF/XGB aerator on/off decision
  GET  /health            → Service health check
  GET  /latest            → Most recent sensor reading (demo)
"""

import os
import sys
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(
    title="PondIQ API",
    description="Real-time fish-pond water quality prediction service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    dissolved_oxygen: float = Field(..., ge=0, le=14,  description="mg/L")
    temperature:      float = Field(..., ge=0, le=45,  description="°C")
    ph:               float = Field(..., ge=0, le=14,  description="pH units")
    turbidity:        float = Field(..., ge=0, le=200, description="NTU")
    hour:             int   = Field(..., ge=0, le=23)
    month:            int   = Field(..., ge=1, le=12)


class SensorWindow(BaseModel):
    """A sequence of sensor readings (at least 16 rows for LSTM, 8 for others)."""
    readings: List[SensorReading]


class DOForecastResponse(BaseModel):
    predicted_do:       float
    alert_level:        str
    timestamp:          str


class AnomalyResponse(BaseModel):
    is_anomaly:         bool
    anomaly_score:      float
    label:              str
    timestamp:          str


class AeratorResponse(BaseModel):
    aerator_on:         bool
    confidence:         float
    label:              str
    timestamp:          str


# ── Lazy-load helpers ─────────────────────────────────────────────────────────

def _to_df(window: SensorWindow):
    import pandas as pd
    return pd.DataFrame([r.dict() for r in window.readings])


def _alert_level(do: float) -> str:
    if do < 4.0:
        return "🔴 CRITICAL — Fish mortality risk"
    if do < 5.0:
        return "🟡 WARNING  — Activate aerator"
    if do < 6.0:
        return "🟠 CAUTION  — Monitor closely"
    return "🟢 NORMAL"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check service status."""
    return {"status": "ok", "service": "PondIQ API", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict/do", response_model=DOForecastResponse)
def predict_do(window: SensorWindow):
    """
    Predict the next dissolved-oxygen value.
    Requires at least 16 sensor readings in `readings`.
    """
    if len(window.readings) < 16:
        raise HTTPException(status_code=422,
                            detail="Minimum 16 readings required for LSTM forecast.")
    try:
        from models.lstm_forecaster import predict_next_do
        df = _to_df(window)
        predicted = predict_next_do(df)
        return DOForecastResponse(
            predicted_do=round(predicted, 3),
            alert_level=_alert_level(predicted),
            timestamp=datetime.utcnow().isoformat(),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="LSTM model not trained yet. Run: python models/lstm_forecaster.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/anomaly", response_model=AnomalyResponse)
def predict_anomaly(window: SensorWindow):
    """
    Detect anomalies in the latest sensor reading.
    Requires at least 8 readings for rolling stats.
    """
    if len(window.readings) < 8:
        raise HTTPException(status_code=422,
                            detail="Minimum 8 readings required for anomaly detection.")
    try:
        from models.anomaly_detector import detect_anomaly
        df = _to_df(window)
        result = detect_anomaly(df)
        return AnomalyResponse(
            **result,
            timestamp=datetime.utcnow().isoformat(),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="Anomaly model not trained yet. Run: python models/anomaly_detector.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/aerator", response_model=AeratorResponse)
def predict_aerator(window: SensorWindow):
    """
    Decide whether to activate the aerator.
    Requires at least 16 readings for rolling features.
    """
    if len(window.readings) < 16:
        raise HTTPException(status_code=422,
                            detail="Minimum 16 readings required for aerator prediction.")
    try:
        from models.aerator_classifier import predict_aerator as _predict
        df = _to_df(window)
        result = _predict(df)
        return AeratorResponse(
            **result,
            timestamp=datetime.utcnow().isoformat(),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="Aerator model not trained yet. Run: python models/aerator_classifier.py")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latest")
def latest_reading():
    """Return the latest row of the dataset (demo endpoint)."""
    import pandas as pd
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "pond_sensor_data.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset not found. Run data generator first.")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    last = df.iloc[-1].to_dict()
    last["timestamp"] = str(last["timestamp"])
    return last


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
