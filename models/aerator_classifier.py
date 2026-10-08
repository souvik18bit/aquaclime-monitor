"""
PondIQ — Aerator Activation Classifier
Uses Random Forest + XGBoost (if available) to decide whether the aerator
should be activated based on current & rolling sensor readings.
Output: 0 = OFF | 1 = ON
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, f1_score, roc_auc_score,
                              RocCurveDisplay)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("ℹ️  XGBoost not installed — using GradientBoosting fallback. "
          "Run: pip install xgboost")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_generator import generate_pond_data

# ── Config ───────────────────────────────────────────────────────────────────
RAW_FEATURES = ["dissolved_oxygen", "temperature", "ph", "turbidity", "hour", "month"]
TARGET       = "aerator_on"
MODEL_DIR    = os.path.join(os.path.dirname(__file__))
RF_PATH      = os.path.join(MODEL_DIR, "aerator_rf_model.pkl")
XGB_PATH     = os.path.join(MODEL_DIR, "aerator_xgb_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "aerator_scaler.pkl")


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["dissolved_oxygen", "temperature"]:
        df[f"{col}_roll_mean_8"]  = df[col].rolling(8,  min_periods=1).mean()
        df[f"{col}_roll_mean_16"] = df[col].rolling(16, min_periods=1).mean()
        df[f"{col}_diff"]         = df[col].diff().fillna(0)
        df[f"{col}_diff2"]        = df[f"{col}_diff"].diff().fillna(0)  # acceleration
    df["do_temp_ratio"]    = df["dissolved_oxygen"] / (df["temperature"] + 1e-6)
    df["do_below_5"]       = (df["dissolved_oxygen"] < 5.0).astype(int)
    df["do_below_4"]       = (df["dissolved_oxygen"] < 4.0).astype(int)
    df["night_flag"]       = df["hour"].apply(lambda h: 1 if h < 6 or h > 22 else 0)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ("timestamp", "crisis_event", "aerator_on")]


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
    feat_cols = get_feature_cols(df)
    X = df[feat_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, shuffle=False  # keep temporal order
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    results = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, RF_PATH)
    rf_f1 = f1_score(y_test, rf.predict(X_test_s))
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])
    results["Random Forest"] = {"model": rf, "f1": rf_f1, "auc": rf_auc,
                                 "y_pred": rf.predict(X_test_s)}
    print(f"\n🌲 Random Forest  — F1: {rf_f1:.4f}  AUC: {rf_auc:.4f}")

    # ── XGBoost / GradientBoosting ────────────────────────────────────────────
    if XGB_AVAILABLE:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    else:
        xgb = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )
    xgb.fit(X_train_s, y_train)
    joblib.dump(xgb, XGB_PATH)
    xgb_f1  = f1_score(y_test, xgb.predict(X_test_s))
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_s)[:, 1])
    model_name = "XGBoost" if XGB_AVAILABLE else "GradientBoosting"
    results[model_name] = {"model": xgb, "f1": xgb_f1, "auc": xgb_auc,
                           "y_pred": xgb.predict(X_test_s)}
    print(f"🚀 {model_name} — F1: {xgb_f1:.4f}  AUC: {xgb_auc:.4f}")

    # Best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best = results[best_name]
    print(f"\n🏆 Best model: {best_name}")
    print(classification_report(y_test, best["y_pred"],
                                 target_names=["Aerator OFF", "Aerator ON"]))

    _plot_results(rf, X_test_s, y_test, feat_cols, results)
    return rf, xgb, scaler


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_aerator(recent_df: pd.DataFrame, use_model: str = "rf") -> dict:
    """
    Predict aerator state for the latest row in recent_df.

    Parameters
    ----------
    recent_df : DataFrame with RAW_FEATURES + hour/month columns (≥16 rows recommended)
    use_model : "rf" or "xgb"

    Returns
    -------
    dict with keys: aerator_on (bool), confidence (float), label (str)
    """
    model_path = RF_PATH if use_model == "rf" else XGB_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    model  = joblib.load(model_path)
    scaler = joblib.load(SCALER_PATH)

    df_feat   = engineer_features(recent_df)
    feat_cols = get_feature_cols(df_feat)
    last_row  = df_feat[feat_cols].iloc[[-1]].values

    X_scaled = scaler.transform(last_row)
    proba    = model.predict_proba(X_scaled)[0][1]
    decision = int(proba >= 0.5)

    label = "🟢 AERATOR ON — Activate Now!" if decision else "⚪ AERATOR OFF — Standby"

    return {
        "aerator_on":  bool(decision),
        "confidence":  round(float(proba), 4),
        "label":       label,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(rf, X_test_s, y_test, feat_cols, results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion matrix (RF)
    cm = confusion_matrix(y_test, rf.predict(X_test_s))
    disp = ConfusionMatrixDisplay(cm, display_labels=["OFF", "ON"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix — Random Forest")

    # Feature importance (top 15)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    axes[1].barh([feat_cols[i] for i in top_idx[::-1]],
                  importances[top_idx[::-1]], color="steelblue")
    axes[1].set_title("Top 15 Feature Importances (RF)")
    axes[1].set_xlabel("Importance")
    axes[1].grid(True, axis="x", alpha=0.3)

    # ROC curves
    for name, res in results.items():
        RocCurveDisplay.from_predictions(
            y_test, res["model"].predict_proba(X_test_s)[:, 1],
            name=f"{name} (AUC={res['auc']:.3f})", ax=axes[2]
        )
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set_title("ROC Curve")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "..", "dashboard", "aerator_results.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"📈 Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    train()
