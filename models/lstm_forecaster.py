"""
PondIQ — LSTM Dissolved Oxygen Forecaster
Predicts DO level for the next time-step given the past SEQ_LEN readings.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ── Optional TensorFlow import ──────────────────────────────────────────────
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed — LSTM training disabled. "
          "Run: pip install tensorflow")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_pond_data, create_sequences

# ── Config ───────────────────────────────────────────────────────────────────
SEQ_LEN      = 16          # 16 × 15 min = 4 hours of history
FEATURES     = ["dissolved_oxygen", "temperature", "ph", "turbidity",
                "hour", "month"]
TARGET       = "dissolved_oxygen"
TRAIN_RATIO  = 0.80
EPOCHS       = 50
BATCH_SIZE   = 64
MODEL_DIR    = os.path.join(os.path.dirname(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "lstm_do_model.keras")
SCALER_PATH  = os.path.join(MODEL_DIR, "lstm_scaler.pkl")


# ── Data preparation ─────────────────────────────────────────────────────────

def load_or_generate_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "pond_sensor_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["timestamp"])
    print("📦 Generating synthetic dataset…")
    df = generate_pond_data(days=90)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def prepare_data(df: pd.DataFrame):
    data = df[FEATURES].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    joblib.dump(scaler, SCALER_PATH)
    print(f"💾 Scaler saved → {SCALER_PATH}")

    X, y = create_sequences(data_scaled, SEQ_LEN)

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler


# ── Model definition ──────────────────────────────────────────────────────────

def build_lstm_model(seq_len: int, n_features: int) -> "Sequential":
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame | None = None):
    if not TF_AVAILABLE:
        print("❌ TensorFlow required for training.")
        return

    if df is None:
        df = load_or_generate_data()

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    model = build_lstm_model(SEQ_LEN, len(FEATURES))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test).flatten()

    # Inverse transform: pad to full feature width, extract DO column
    def inverse_do(scaled_vals):
        pad = np.zeros((len(scaled_vals), len(FEATURES)))
        pad[:, 0] = scaled_vals
        return scaler.inverse_transform(pad)[:, 0]

    y_pred = inverse_do(y_pred_scaled)
    y_true = inverse_do(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n📊 Test MAE : {mae:.4f} mg/L")
    print(f"📊 Test RMSE: {rmse:.4f} mg/L")

    _plot_results(history, y_true, y_pred)
    return model, history


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_next_do(recent_df: pd.DataFrame) -> float:
    """
    Predict the next DO value given the last SEQ_LEN rows of sensor readings.

    Parameters
    ----------
    recent_df : DataFrame with at least SEQ_LEN rows and FEATURES columns

    Returns
    -------
    Predicted dissolved oxygen (mg/L)
    """
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow not available.")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found. Train first.")

    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    data   = recent_df[FEATURES].values[-SEQ_LEN:]
    scaled = scaler.transform(data)
    X      = scaled.reshape(1, SEQ_LEN, len(FEATURES))

    pred_scaled = model.predict(X, verbose=0).flatten()[0]

    pad = np.zeros((1, len(FEATURES)))
    pad[0, 0] = pred_scaled
    pred_do = scaler.inverse_transform(pad)[0, 0]
    return float(np.clip(pred_do, 0.0, 14.0))


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_results(history, y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Actual vs Predicted
    n_show = min(500, len(y_true))
    axes[1].plot(y_true[:n_show],  label="Actual DO",    alpha=0.8)
    axes[1].plot(y_pred[:n_show],  label="Predicted DO", alpha=0.8, linestyle="--")
    axes[1].axhline(4.0, color="red",    linestyle=":", label="Crisis threshold (4 mg/L)")
    axes[1].axhline(5.0, color="orange", linestyle=":", label="Alert threshold (5 mg/L)")
    axes[1].set_title("Actual vs Predicted Dissolved Oxygen")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("DO (mg/L)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "..", "dashboard", "lstm_results.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"📈 Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    train()
