import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="Z-Score Outlier & Distribution Explorer", layout="centered")
st.title("📊 Z-Score Outlier Detection + Interactive Normal Distribution")

st.markdown("""
Explore **Z-scores**, **outliers**, and the **normal distribution** interactively.  
This app combines:
- 🔍 Outlier detection (Classic / Robust / Iterative)
- 📈 Histogram + Normal curve with skewness
- 🧮 Dynamic Z-score and cumulative probability visualizer
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
# Outlier detection
# ──────────────────────────────────────────────
st.subheader("2️⃣ Choose detection method and threshold")

method = st.radio(
    "Detection method:",
    ["Classic (Mean / Std Dev)", "Robust (Median / MAD)", "Iterative 3σ clipping"],
    horizontal=True
)
threshold = st.slider("Threshold", 1.5, 5.0, 3.0, 0.1)

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

res = pd.DataFrame({"Value": data, label: np.round(z, 3), "Outlier": outlier})
st.dataframe(res, use_container_width=True)
st.success(f"Detected **{outlier.sum()}** outlier(s) out of {len(data)} observations.")

# ──────────────────────────────────────────────
# Histogram + Normal Curve
# ──────────────────────────────────────────────
st.subheader("3️⃣ Distribution Analysis")

skewness = skew(data)
kurt = kurtosis(data)
mean = np.mean(data)
std = np.std(data, ddof=1)

fig1, ax = plt.subplots(figsize=(8, 4))
count, bins, _ = ax.hist(data, bins=20, color="lightgray", alpha=0.7, density=True)
x_axis = np.linspace(min(data), max(data), 200)
ax.plot(x_axis, norm.pdf(x_axis, mean, std), color="blue", lw=2, label="Normal Curve")
ax.axvline(mean, color="green", linestyle="--", label="Mean")
for k in [1, 2, 3]:
    ax.axvline(mean + k*std, color="red", linestyle=":", lw=1)
    ax.axvline(mean - k*std, color="red", linestyle=":", lw=1)
ax.legend()
ax.set_title("Histogram with Fitted Normal Distribution")
st.pyplot(fig1)
st.info(f"Skewness = {skewness:.3f} Kurtosis = {kurt:.3f}")

# ──────────────────────────────────────────────
# Interactive Plotly Normal Distribution
# ──────────────────────────────────────────────
st.subheader("4️⃣ Interactive Normal Distribution & Z-Score")

x_vals = np.linspace(mean - 4*std, mean + 4*std, 800)
y_vals = norm.pdf(x_vals, mean, std)

X = st.slider("Select a value X:", float(x_vals.min()), float(x_vals.max()), float(mean))
z_score = (X - mean) / std
cdf = norm.cdf(X, mean, std)

st.metric(label="Z-Score", value=f"{z_score:.3f}")
st.metric(label="Cumulative Probability", value=f"{cdf:.4f}")

fig2 = go.Figure()

# Normal curve
fig2.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='blue', width=2), name="Normal PDF"))

# Shaded area up to X
mask = x_vals <= X
fig2.add_trace(go.Scatter(
    x=np.concatenate(([x_vals[0]], x_vals[mask], [X])),
    y=np.concatenate(([0], y_vals[mask], [0])),
    fill='toself', fillcolor='rgba(0,176,246,0.3)', line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip', name=f'Area ≤ X ({cdf:.3f})'
))

# Mean line
fig2.add_trace(go.Scatter(x=[mean, mean], y=[0, max(y_vals)],
                          mode='lines', line=dict(color='green', dash='dash'),
                          name='Mean (μ)'))

# X marker
fig2.add_trace(go.Scatter(
    x=[X, X],
    y=[0, norm.pdf(X, mean, std)],
    mode='lines+text',
    line=dict(color='red', width=2),
    name=f'X = {X:.2f}',
    text=[f"Z = {z_score:.2f}"],
    textposition="top right"
))

# ±σ markers
for k in [1, 2, 3]:
    for side in [-1, 1]:
        x_pos = mean + side * k * std
        fig2.add_trace(go.Scatter(
            x=[x_pos, x_pos],
            y=[0, norm.pdf(x_pos, mean, std)],
            mode='lines',
            line=dict(color='gray', dash='dot', width=1),
            showlegend=False
        ))

fig2.update_layout(
    title="Normal Distribution with Dynamic Z-Score",
    xaxis_title="Value (X)",
    yaxis_title="Probability Density",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"""
**Interpretation:**  
- Z-score = {z_score:.2f} → {abs(z_score):.2f} σ {'above' if z_score>0 else 'below'} the mean  
- Probability of value ≤ X: **{cdf:.3%}**
""")

st.caption("Educational demo • Generated by GPT-5 · © Alastair McBride 2025")
