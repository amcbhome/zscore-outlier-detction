import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="Classic Z-Score Analyzer", layout="centered")
st.title("📊 Classic Z-Score Analyzer")

st.markdown("""
Upload a CSV with **one numeric column**.  
This app will:
- Compute **summary statistics** and **Z-scores**  
- Describe the **shape** of the distribution (skewness / kurtosis)  
- Plot the **normal curve** with ±1σ, ±2σ, ±3σ markers  
- Ask if **outliers** (|Z| > 3) should be removed and the dataset re-analysed
""")

# ──────────────────────────────────────────────
# File upload
# ──────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV with one numeric column", type=["csv"])

if uploaded is None:
    st.info("Awaiting CSV upload…")
    st.stop()

df = pd.read_csv(uploaded)
col = st.selectbox("Select numeric column", df.columns)
data = df[col].dropna().astype(float).values

# ──────────────────────────────────────────────
# Function: classic z-score analysis
# ──────────────────────────────────────────────
def analyze_dataset(data, label="Original Dataset"):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    minimum, maximum = np.min(data), np.max(data)
    skw, krt = skew(data), kurtosis(data)

    summary = pd.DataFrame({
        "n": [n],
        "Min": [minimum],
        "Max": [maximum],
        "Mean": [mean],
        "Std Dev": [std],
        "Skewness": [skw],
        "Kurtosis": [krt]
    }).T.rename(columns={0: "Value"})

    st.subheader(f"📈 Summary Statistics — {label}")
    st.dataframe(summary, use_container_width=True)

    # Distribution description
    if abs(skw) < 0.5:
        shape = "approximately **symmetric** (normal-like)"
    elif skw > 0.5:
        shape = "**positively skewed** (right-tailed)"
    else:
        shape = "**negatively skewed** (left-tailed)"

    if abs(krt) < 1:
        peaked = "has a **normal level of peakedness**."
    elif krt > 1:
        peaked = "is **leptokurtic** (more peaked with heavier tails)."
    else:
        peaked = "is **platykurtic** (flatter with lighter tails)."

    st.markdown(f"""
    The dataset is {shape} and {peaked}  
    - **Mean (μ):** {mean:.3f} **Std Dev (σ):** {std:.3f}  
    - Most observations lie within ±1σ ≈ {mean-std:.2f} to {mean+std:.2f}.  
    """)

    # Z-score calculation
    z_scores = (data - mean) / std
    z_table = pd.DataFrame({"X (Value)": data, "Z-Score": np.round(z_scores, 3)})
    z_table["Outlier?"] = np.abs(z_table["Z-Score"]) > 3
    st.subheader("📊 Z-Scores Table")
    st.dataframe(z_table, use_container_width=True)

    # Visualization

    # ──────────────────────────────────────────────
# Empirical Rule Chart (68–95–99.7%)
# ──────────────────────────────────────────────
st.subheader("📊 Empirical Rule — 68%, 95%, 99.7% Coverage")

import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate standard normal x-axis and PDF
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color="blue", lw=2)

# Shade the regions corresponding to 68%, 95%, and 99.7%
# 1σ region
ax.fill_between(x, 0, y, where=(x > -1) & (x < 1), color="skyblue", alpha=0.5)
# 2σ region
ax.fill_between(x, 0, y, where=(x > -2) & (x < 2), color="lightblue", alpha=0.3)
# 3σ region
ax.fill_between(x, 0, y, where=(x > -3) & (x < 3), color="powderblue", alpha=0.2)

# Vertical dashed lines for ±1σ, ±2σ, ±3σ
for i in range(-3, 4):
    ax.axvline(i, color="red", linestyle="--", lw=1)
    if i != 0:
        ax.text(i, 0.02, f"{i:+d} SD", color="darkred",
                ha="center", fontsize=9, fontweight="bold")

# Mean line (center)
ax.axvline(0, color="black", linestyle="-", lw=1.5)
ax.text(0, 0.43, "Mean (μ)", ha="center", fontsize=10, fontweight="bold")

# Percentage labels for the empirical rule
ax.text(0, 0.2, "68%", ha="center", fontsize=11, color="navy", fontweight="bold")
ax.text(0, 0.1, "95%", ha="center", fontsize=11, color="navy", fontweight="bold")
ax.text(0, 0.03, "99.7%", ha="center", fontsize=11, color="navy", fontweight="bold")

# Tidy up
ax.set_title("Standard Normal Distribution — Empirical Rule (68–95–99.7%)", fontsize=12)
ax.set_xlabel("Standard Deviations from the Mean (Z)")
ax.set_ylabel("Density")
ax.set_xlim(-4, 4)
ax.set_ylim(0, 0.45)
ax.grid(False)

st.pyplot(fig)

    st.subheader("📈 Normal Distribution Visualization")
    x = np.linspace(mean - 4*std, mean + 4*std, 800)
    y = norm.pdf(x, mean, std)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue", width=2),
                             name="Normal PDF"))
    fig.add_trace(go.Scatter(x=[mean, mean], y=[0, max(y)],
                             mode="lines", line=dict(color="green", dash="dash"),
                             name="Mean (μ)"))

    for k in [1, 2, 3]:
        fig.add_trace(go.Scatter(x=[mean + k*std, mean + k*std],
                                 y=[0, norm.pdf(mean + k*std, mean, std)],
                                 mode="lines", line=dict(color="gray", dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[mean - k*std, mean - k*std],
                                 y=[0, norm.pdf(mean - k*std, mean, std)],
                                 mode="lines", line=dict(color="gray", dash="dot"),
                                 showlegend=False))

    # Overlay data points
    colors = np.where(np.abs(z_scores) > 3, "crimson", "red")
    fig.add_trace(go.Scatter(
        x=data, y=norm.pdf(data, mean, std),
        mode="markers", name="Data Points",
        marker=dict(color=colors, size=6, symbol="circle")
    ))

    fig.update_layout(
        title=f"Normal Distribution — {label}",
        xaxis_title="Value (X)",
        yaxis_title="Probability Density",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    return z_table, mean, std


# ──────────────────────────────────────────────
# Run initial analysis
# ──────────────────────────────────────────────
z_table, mean, std = analyze_dataset(data, "Original Dataset")

# ──────────────────────────────────────────────
# Ask user about outlier removal
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("🧹 Outlier Removal Option")

num_outliers = np.sum(np.abs((data - mean) / std) > 3)
if num_outliers > 0:
    st.warning(f"{num_outliers} potential outlier(s) detected (|Z| > 3).")
    if st.button("Remove Outliers and Re-analyse"):
        clean_data = data[np.abs((data - mean) / std) <= 3]
        st.success(f"Removed {num_outliers} outlier(s). Recalculating statistics…")
        analyze_dataset(clean_data, "After Outlier Removal")
else:
    st.success("No outliers detected — no removal necessary.")

st.caption("Educational demo • Classic Z-Score method • Generated by GPT-5 · © Alastair McBride 2025")

