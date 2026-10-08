"""
PondIQ — Interactive Streamlit Dashboard
Run: streamlit run dashboard/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import streamlit as st
    STREAMLIT = True
except ImportError:
    STREAMLIT = False

from utils.data_generator import generate_pond_data


# ── Page config ───────────────────────────────────────────────────────────────
if STREAMLIT:
    st.set_page_config(
        page_title="PondIQ — Fish Pond Monitor",
        page_icon="🐟",
        layout="wide",
    )

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pond_sensor_data.csv")


@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    else:
        df = generate_pond_data(days=90)
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
    return df


def status_badge(do_value: float) -> str:
    if do_value < 4.0:
        return "🔴 CRITICAL"
    if do_value < 5.0:
        return "🟡 WARNING"
    if do_value < 6.0:
        return "🟠 CAUTION"
    return "🟢 NORMAL"


def run_dashboard():
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🐟 PondIQ — Real-time Fish Pond Health Monitor")
    st.markdown("*Dissolved Oxygen Crisis Predictor & Automated Aerator Controller*")
    st.divider()

    df = load_data()

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Controls")
    days_to_show = st.sidebar.slider("Days to display", 1, 30, 7)
    show_anomalies = st.sidebar.checkbox("Highlight anomalies", value=True)
    show_thresholds = st.sidebar.checkbox("Show DO thresholds", value=True)

    rows = days_to_show * 24 * 4  # 15-min intervals
    df_view = df.tail(rows).reset_index(drop=True)
    latest = df_view.iloc[-1]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💧 Dissolved O₂", f"{latest['dissolved_oxygen']:.2f} mg/L",
                delta=f"{latest['dissolved_oxygen'] - df_view['dissolved_oxygen'].mean():.2f} vs avg")
    col2.metric("🌡️ Temperature",   f"{latest['temperature']:.1f} °C")
    col3.metric("⚗️ pH",            f"{latest['ph']:.2f}")
    col4.metric("🌊 Turbidity",     f"{latest['turbidity']:.1f} NTU")
    col5.metric("🚨 Status",        status_badge(latest["dissolved_oxygen"]))

    st.divider()

    # ── Main DO chart ─────────────────────────────────────────────────────────
    st.subheader("📈 Dissolved Oxygen Timeline")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df_view.index, df_view["dissolved_oxygen"],
            color="steelblue", linewidth=1.2, label="DO (mg/L)")

    if show_thresholds:
        ax.axhline(4.0, color="red",    linestyle="--", alpha=0.7, linewidth=1,
                   label="Crisis (4 mg/L)")
        ax.axhline(5.0, color="orange", linestyle="--", alpha=0.7, linewidth=1,
                   label="Alert (5 mg/L)")
        ax.fill_between(df_view.index, 0, 4.0,
                         alpha=0.05, color="red")

    if show_anomalies:
        crisis_idx = df_view.index[df_view["crisis_event"] == 1]
        ax.scatter(crisis_idx, df_view.loc[crisis_idx, "dissolved_oxygen"],
                   color="red", s=20, zorder=5, label="Crisis Event")

    # Aerator ON regions
    aerator_on = df_view["aerator_on"].values
    in_block = False
    for i, val in enumerate(aerator_on):
        if val == 1 and not in_block:
            start_i = i
            in_block = True
        if (val == 0 or i == len(aerator_on) - 1) and in_block:
            ax.axvspan(start_i, i, alpha=0.10, color="green")
            in_block = False

    green_patch = mpatches.Patch(color="green", alpha=0.3, label="Aerator Active")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [green_patch], fontsize=8, loc="upper right")
    ax.set_xlabel("Time Step (15-min intervals)")
    ax.set_ylabel("DO (mg/L)")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 14)
    st.pyplot(fig)
    plt.close(fig)

    # ── Multi-parameter panel ─────────────────────────────────────────────────
    st.subheader("🌡️ Sensor Parameters")
    col_a, col_b = st.columns(2)

    with col_a:
        fig2, axes2 = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes2[0].plot(df_view["temperature"], color="tomato",    linewidth=1)
        axes2[0].set_ylabel("Temperature (°C)")
        axes2[0].grid(True, alpha=0.2)
        axes2[1].plot(df_view["ph"],          color="mediumseagreen", linewidth=1)
        axes2[1].set_ylabel("pH")
        axes2[1].grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    with col_b:
        fig3, axes3 = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes3[0].plot(df_view["turbidity"], color="saddlebrown", linewidth=1)
        axes3[0].set_ylabel("Turbidity (NTU)")
        axes3[0].grid(True, alpha=0.2)

        # DO distribution
        axes3[1].hist(df_view["dissolved_oxygen"], bins=40,
                      color="steelblue", alpha=0.7, edgecolor="white")
        axes3[1].axvline(4.0, color="red",    linestyle="--", label="Crisis 4 mg/L")
        axes3[1].axvline(5.0, color="orange", linestyle="--", label="Alert 5 mg/L")
        axes3[1].set_xlabel("DO (mg/L)")
        axes3[1].set_ylabel("Count")
        axes3[1].legend(fontsize=8)
        axes3[1].grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    # ── Statistics table ──────────────────────────────────────────────────────
    st.subheader("📊 Summary Statistics")
    stats = df_view[["dissolved_oxygen", "temperature", "ph", "turbidity"]].describe().T
    stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    st.dataframe(stats.round(3), use_container_width=True)

    # ── Crisis events table ───────────────────────────────────────────────────
    crisis_df = df_view[df_view["crisis_event"] == 1][
        ["dissolved_oxygen", "temperature", "ph", "turbidity"]
    ]
    if len(crisis_df):
        st.subheader(f"🚨 Crisis Events in Selected Period ({len(crisis_df)} rows)")
        st.dataframe(crisis_df.reset_index(drop=True), use_container_width=True)
    else:
        st.success("✅ No crisis events detected in the selected period.")

    # ── Model status ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 Model Status")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models = {
        "LSTM Forecaster":         "lstm_do_model.keras",
        "Anomaly Detector":        "anomaly_model.pkl",
        "Aerator Classifier (RF)": "aerator_rf_model.pkl",
        "Aerator Classifier (XGB)":"aerator_xgb_model.pkl",
    }
    mcols = st.columns(len(models))
    for i, (name, fname) in enumerate(models.items()):
        path = os.path.join(model_dir, fname)
        exists = os.path.exists(path)
        mcols[i].metric(name, "✅ Ready" if exists else "❌ Not trained")

    st.info("To train all models, run: **python train_all.py** from the PondIQ folder.")


if __name__ == "__main__":
    if STREAMLIT:
        run_dashboard()
    else:
        print("❌ Streamlit not installed. Run: pip install streamlit")
        print("   Then: streamlit run dashboard/dashboard.py")
