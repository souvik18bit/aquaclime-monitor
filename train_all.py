"""
PondIQ — Master Training Script
Runs the full ML pipeline:
  1. Generate synthetic dataset
  2. Train Anomaly Detector (fast, no GPU needed)
  3. Train Aerator Classifier  (fast, no GPU needed)
  4. Train LSTM Forecaster     (needs TensorFlow)

Usage: python train_all.py [--skip-lstm]
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))


def banner(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def step_1_generate():
    banner("STEP 1 — Generate Synthetic Dataset")
    from utils.data_generator import generate_pond_data
    csv_path = os.path.join("data", "pond_sensor_data.csv")
    os.makedirs("data", exist_ok=True)
    df = generate_pond_data(days=90)
    df.to_csv(csv_path, index=False)
    print(f"✅ Dataset ready → {csv_path}  ({len(df):,} rows)")
    return df


def step_2_anomaly(df):
    banner("STEP 2 — Train Anomaly Detector (Isolation Forest)")
    t0 = time.time()
    from models.anomaly_detector import train as train_anomaly
    train_anomaly(df.copy())
    print(f"✅ Anomaly model trained in {time.time()-t0:.1f}s")


def step_3_aerator(df):
    banner("STEP 3 — Train Aerator Classifier (RF + XGB)")
    t0 = time.time()
    from models.aerator_classifier import train as train_aerator
    train_aerator(df.copy())
    print(f"✅ Aerator classifier trained in {time.time()-t0:.1f}s")


def step_4_lstm(df):
    banner("STEP 4 — Train LSTM DO Forecaster")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        print("⚠️  TensorFlow not installed — skipping LSTM training.")
        print("    Install with: pip install tensorflow")
        return
    t0 = time.time()
    from models.lstm_forecaster import train as train_lstm
    train_lstm(df.copy())
    print(f"✅ LSTM model trained in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="PondIQ — Train all ML models")
    parser.add_argument("--skip-lstm", action="store_true",
                        help="Skip LSTM training (useful on CPU-only machines)")
    args = parser.parse_args()

    print("\n🐟  PondIQ — Full ML Pipeline")
    print(f"    Working dir: {os.getcwd()}")

    df = step_1_generate()
    step_2_anomaly(df)
    step_3_aerator(df)
    if not args.skip_lstm:
        step_4_lstm(df)
    else:
        print("\n⏭️  LSTM training skipped (--skip-lstm flag).")

    banner("ALL DONE ✅")
    print("Next steps:")
    print("  • Start API   → cd api && uvicorn app:app --reload")
    print("  • Dashboard   → streamlit run dashboard/dashboard.py")
    print("  • API docs    → http://localhost:8000/docs")


if __name__ == "__main__":
    main()
