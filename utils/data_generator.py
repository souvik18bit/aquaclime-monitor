"""
PondIQ — Synthetic Fish Pond Sensor Data Generator
Generates realistic time-series data for dissolved oxygen, temperature, pH, turbidity.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_pond_data(days: int = 90, freq_minutes: int = 15, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic pond sensor readings.

    Parameters
    ----------
    days          : Number of days to simulate
    freq_minutes  : Sampling interval in minutes
    seed          : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, dissolved_oxygen, temperature, ph, turbidity,
        hour, day_of_week, month, crisis_event, aerator_on
    """
    np.random.seed(seed)

    n = int(days * 24 * 60 / freq_minutes)
    start = datetime(2025, 1, 1)
    timestamps = [start + timedelta(minutes=freq_minutes * i) for i in range(n)]

    hours = np.array([t.hour for t in timestamps])
    day_of_week = np.array([t.weekday() for t in timestamps])
    month = np.array([t.month for t in timestamps])

    # --- Temperature (°C) ---
    # Peaks midday ~30°C, cooler at night ~24°C, seasonal variation
    temp_diurnal = 3.0 * np.sin(2 * np.pi * (hours - 6) / 24)
    temp_seasonal = 2.0 * np.sin(2 * np.pi * (month - 3) / 12)
    temperature = 27.0 + temp_diurnal + temp_seasonal + np.random.normal(0, 0.5, n)
    temperature = np.clip(temperature, 20, 36)

    # --- pH ---
    # Photosynthesis raises pH during day, drops at night
    ph_diurnal = 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)
    ph = 7.5 + ph_diurnal + np.random.normal(0, 0.15, n)
    ph = np.clip(ph, 6.0, 9.5)

    # --- Turbidity (NTU) ---
    turbidity = 20 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.exponential(3, n)
    turbidity = np.clip(turbidity, 5, 80)

    # --- Dissolved Oxygen (mg/L) ---
    # Inversely related to temperature, peaks in morning (algae photosynthesis),
    # drops at night (respiration consumes O2)
    do_diurnal = 2.5 * np.sin(2 * np.pi * (hours - 4) / 24)
    do_temp_effect = -0.2 * (temperature - 27)
    do_base = 7.5 + do_diurnal + do_temp_effect + np.random.normal(0, 0.3, n)

    # Inject realistic crisis events (DO drops < 4 mg/L)
    crisis_indices = []
    i = 0
    while i < n:
        # Crisis more likely at night (hours 0-6) or in hot months
        if np.random.random() < 0.002:
            crisis_len = np.random.randint(4, 20)  # 1–5 hours
            for j in range(i, min(i + crisis_len, n)):
                do_base[j] -= np.random.uniform(3.5, 5.5)
                crisis_indices.append(j)
            i += crisis_len + np.random.randint(50, 200)
        else:
            i += 1

    do = np.clip(do_base, 0.5, 14.0)

    # --- Crisis label (DO < 4 mg/L is dangerous) ---
    crisis_event = (do < 4.0).astype(int)

    # --- Aerator on: activates when DO < 5 mg/L or predicted to drop ---
    aerator_on = (do < 5.0).astype(int)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "dissolved_oxygen": np.round(do, 3),
        "temperature": np.round(temperature, 2),
        "ph": np.round(ph, 2),
        "turbidity": np.round(turbidity, 2),
        "hour": hours,
        "day_of_week": day_of_week,
        "month": month,
        "crisis_event": crisis_event,
        "aerator_on": aerator_on,
    })

    return df


def create_sequences(data: np.ndarray, seq_len: int = 16):
    """
    Slide a window over a 1-D or 2-D array to produce (X, y) pairs
    for time-series forecasting.

    Parameters
    ----------
    data    : shape (n, features) or (n,)
    seq_len : look-back window length

    Returns
    -------
    X : shape (samples, seq_len, features)
    y : shape (samples,) — next DO value
    """
    X, y = [], []
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, 0])  # predict DO (first column)
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = generate_pond_data(days=90)
    out = "PondIQ/data/pond_sensor_data.csv"
    df.to_csv(out, index=False)
    print(f"✅ Dataset saved → {out}")
    print(df.describe())
    print(f"\nCrisis events: {df['crisis_event'].sum()} rows "
          f"({df['crisis_event'].mean()*100:.1f}% of data)")
