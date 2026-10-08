"""
PondIQ — Anomaly Detection Model
Uses Isolation Forest to flag unusual sensor readings (equipment failure,
pollution events, ghost-fishing contamination, etc.)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_pond_data

# ── Config ───────────────────────────────────────────────────────────────────
FEATURES        = ["dissolved_oxygen", "temperature", "ph", "turbidity"]
CONTAMINATION   = 0.04        # expected anomaly fraction (~4 %)
MODEL_DIR       = os.path.join(os.path.dirname(__file__))
MODEL_PATH      = os.path.join(MODEL_DIR, "anomaly_model.pkl")
SCALER_PATH     = os.path.join(MODEL_DIR, "anomaly_scaler.pkl")


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics and rate-of-change features."""
    df = df.copy()
    for col in FEATURES:
        df[f"{col}_roll_mean"] = df[col].rolling(8, min_periods=1).mean()
        df[f"{col}_roll_std"]  = df[col].rolling(8, min_periods=1).std().fillna(0)
        df[f"{col}_diff"]      = df[col].diff().fillna(0)
    return df


def get_all_features(df: pd.DataFrame) -> list[str]:
    base = list(FEATURES)
    extra = []
    for col in FEATURES:
        extra += [f"{col}_roll_mean", f"{col}_roll_std", f"{col}_diff"]
    return base + extra


# ── Train ─────────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame | None = None):
    if df is None:
        csv_path = os.path.join(MODEL_DIR, "..", "data", "pond_sensor_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        else:
            print("📦 Generating synthetic dataset…")
            df = generate_pond_data(days=90)

    df = engineer_features(df)
    feat_cols = get_all_features(df)
    X = df[feat_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Anomaly model saved  → {MODEL_PATH}")
    print(f"💾 Anomaly scaler saved → {SCALER_PATH}")

    # ── Evaluate using crisis_event as proxy ground truth ─────────────────────
    raw_preds = model.predict(X_scaled)          # +1 = normal, -1 = anomaly
    df["anomaly_pred"] = (raw_preds == -1).astype(int)
    df["anomaly_score"] = -model.decision_function(X_scaled)  # higher = more anomalous

    # Use crisis_event as approximate ground truth
    if "crisis_event" in df.columns:
        print("\n📊 Anomaly Detection Report (vs crisis_event labels):")
        print(classification_report(df["crisis_event"], df["anomaly_pred"],
                                    target_names=["Normal", "Anomaly"]))
        _plot_results(df)

    return model, scaler, df


# ── Inference ─────────────────────────────────────────────────────────────────

def detect_anomaly(recent_df: pd.DataFrame) -> dict:
    """
    Detect anomaly in the most recent sensor window.

    Parameters
    ----------
    recent_df : DataFrame with FEATURES columns (at least 8 rows for rolling stats)

    Returns
    -------
    dict with keys: is_anomaly (bool), anomaly_score (float), label (str)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Anomaly model not found. Train first.")

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df_feat = engineer_features(recent_df)
    feat_cols = get_all_features(df_feat)
    last_row = df_feat[feat_cols].iloc[[-1]].values

    X_scaled   = scaler.transform(last_row)
    pred       = model.predict(X_scaled)[0]
    score      = float(-model.decision_function(X_scaled)[0])

    is_anomaly = pred == -1
    severity   = "🔴 CRITICAL" if score > 0.3 else "🟡 WARNING" if is_anomaly else "🟢 NORMAL"

    return {
        "is_anomaly":    bool(is_anomaly),
        "anomaly_score": round(score, 4),
        "label":         severity,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # DO timeline with anomaly flags
    n = min(1000, len(df))
    axes[0].plot(df["dissolved_oxygen"].values[:n], label="DO (mg/L)", alpha=0.8)
    anomaly_idx = df.index[df["anomaly_pred"] == 1].tolist()
    anomaly_idx = [i for i in anomaly_idx if i < n]
    axes[0].scatter(anomaly_idx,
                    df["dissolved_oxygen"].values[anomaly_idx],
                    color="red", s=15, zorder=5, label="Detected Anomaly")
    axes[0].axhline(4.0, color="red",    linestyle=":", alpha=0.6, label="Crisis (4 mg/L)")
    axes[0].axhline(5.0, color="orange", linestyle=":", alpha=0.6, label="Alert (5 mg/L)")
    axes[0].set_title("Dissolved Oxygen with Detected Anomalies")
    axes[0].set_ylabel("DO (mg/L)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Anomaly score distribution
    axes[1].hist(df["anomaly_score"], bins=80, color="steelblue", alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--", label="Decision boundary")
    axes[1].set_title("Anomaly Score Distribution")
    axes[1].set_xlabel("Anomaly Score (higher = more anomalous)")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "..", "dashboard", "anomaly_results.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"📈 Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    train()
