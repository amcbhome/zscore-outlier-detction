import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="Z-Score Outlier Detection", layout="centered")
st.title("📊 Z-Score Outlier Detection with Distribution Analysis")

st.markdown("""
This app detects **outliers** using three methods:
1. **Classic (Mean ± k·σ)** – sensitive to extremes  
2. **Robust (Median/MAD)** – resistant to skew/outliers  
3. **Iterative 3σ Clipping** – recomputes mean & σ until stable  

It also plots the **distribution curve** to show *skewness* and *outlier effects*.
""")

# ──────────────────────────────────────────────
# Data input
# ──────────────────────────────────────────────
st.subheader("1️⃣ Upload or enter numeric data")

src = st.radio("Input method:", ["Manual entry", "Upload CSV"], horizontal=True)

if src == "Manual entry":
    text = st.text_area(
        "Enter comma-separated numbers:",
        "98.3, 101.2, 99.8, 102.5, 97.9, 100.6, 98.7, 101.1, 103.2, 97.4, "
        "100.9, 99.5, 101.7, 98.1, 102.3, 100.2, 99.1, 98.9, 103.4, 101.3, "
        "99.7, 97.6, 100.4, 102.1, 98.8, 99.9, 101.5, 100.8, 98.4, 99.3, "
        "97.8, 102.7, 100.1, 99.6, 98.2, 100.5, 103.1, 99.4, 101.8, 97.7, "
        "250, -50, 300, 400, -75, 500, 99, 100, 101, 98"
    )
    try:
        data = np.array([float(x.strip()) for x in text.split(",") if x.strip()])
    except ValueError:
        st.error("Please enter valid numeric values separated by commas.")
        st.stop()
else:
    file = st.file_uploader("Upload CSV with one numeric column", type=["csv"])
    if not file:
        st.info("Awaiting CSV upload…")
        st.stop()
    df = pd.read_csv(file)
    st.write(df.head())
    col = st.selectbox("Select numeric column", df.columns)
    data = df[col].values

# ──────────────────────────────────────────────
# Method selection
# ──────────────────────────────────────────────
st.subheader("2️⃣ Choose detection method and threshold")

method = st.radio(
    "Detection method:",
    ["Classic (Mean / Std Dev)", "Robust (Median / MAD)", "Iterative 3σ clipping"],
    horizontal=True
)
threshold = st.slider("Threshold", 1.5, 5.0, 3.0, 0.1)

# ──────────────────────────────────────────────
# Outlier computation
# ──────────────────────────────────────────────
if method == "Classic (Mean / Std Dev)":
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z = (data - mean) / std
    outlier = np.abs(z) > threshold
    label = "Z-score"

elif method == "Robust (Median / MAD)":
    med = np.median(data)
    mad = np.median(np.abs(data - med)) or 1e-9
    z = 0.6745 * (data - med) / mad
    outlier = np.abs(z) > 3.5
    label = "Modified Z"

else:
    x = data.astype(float).copy()
    mask = np.ones_like(x, dtype=bool)
    while True:
        mu = np.mean(x[mask])
        sd = np.std(x[mask], ddof=1)
        z_all = (x - mu) / sd
        new_mask = np.abs(z_all) <= threshold
        if np.all(new_mask == mask):
            break
        mask = new_mask
    z = z_all
    outlier = ~mask
    label = "Iterative Z"

# ──────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────
st.subheader("3️⃣ Results Table")

res = pd.DataFrame({"Value": data, label: np.round(z, 3), "Outlier": outlier})
st.dataframe(res, use_container_width=True)
st.success(f"Detected **{outlier.sum()}** outlier(s) out of {len(data)} observations.")

# ──────────────────────────────────────────────
# Visualization 1 — Scatter plot
# ──────────────────────────────────────────────
st.subheader("4️⃣ Scatter Plot")

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.scatter(range(len(data)), data, c=~outlier, cmap="coolwarm", s=80, edgecolors="black")
ax1.axhline(np.mean(data), color="green", linestyle="--", label="Mean")
ax1.set_xlabel("Index")
ax1.set_ylabel("Value")
ax1.legend()
st.pyplot(fig1)

# ──────────────────────────────────────────────
# Visualization 2 — Distribution & Skewness
# ──────────────────────────────────────────────
st.subheader("5️⃣ Distribution and Skewness Analysis")

skewness = skew(data)
kurt = kurtosis(data)
mean = np.mean(data)
std = np.std(data, ddof=1)

fig2, ax2 = plt.subplots(figsize=(8, 4))
# Histogram
count, bins, _ = ax2.hist(data, bins=20, color="lightgray", alpha=0.7, density=True, label="Data")
# Normal PDF curve
x_axis = np.linspace(min(data), max(data), 200)
ax2.plot(x_axis, norm.pdf(x_axis, mean, std), color="blue", lw=2, label="Normal Curve")
ax2.set_title("Histogram with Fitted Normal Distribution")
ax2.legend()
st.pyplot(fig2)

st.info(f"**Skewness:** {skewness:.3f}  **Kurtosis:** {kurt:.3f}")

if skewness > 0.5:
    st.warning("⚠️ The distribution is **positively skewed** (right tail longer).")
elif skewness < -0.5:
    st.warning("⚠️ The distribution is **negatively skewed** (left tail longer).")
else:
    st.success("✅ The distribution is approximately **symmetric**.")

st.markdown("---")
st.caption("Educational demo • Generated by GPT-5 · © Alastair McBride 2025")
