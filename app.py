import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import norm, skew, kurtosis

# ──────────────────────────────────────────────
# Page setup
# ──────────────────────────────────────────────
st.set_page_config(page_title="Classic Z-Score Analyzer", layout="centered")
st.title("📊 Classic Z-Score Analyzer")

st.markdown("""
Upload a CSV file containing **one numeric column**.  
This app will:
- Compute **summary statistics** and **Z-scores**  
- Describe the **shape** of the distribution (skewness / kurtosis)  
- Identify and optionally **remove outliers** (|Z| > 3)  
- Display the **Empirical Rule (68–95–99.7%)** using your dataset’s mean and standard deviation  
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
# Function: Classic Z-score analysis
# ──────────────────────────────────────────────
def analyze_dataset(data, label="Dataset"):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    minimum, maximum = np.min(data), np.max(data)
    skw, krt = skew(data), kurtosis(data)

    # Summary stats
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
    st.subheader("🧠 Distribution Description")

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

    # Return for re-analysis
    return z_table, mean, std


# ──────────────────────────────────────────────
# Initial analysis
# ──────────────────────────────────────────────
z_table, mean, std = analyze_dataset(data, "Original Dataset")

# ──────────────────────────────────────────────
# Outlier removal step
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
        mean = np.mean(clean_data)
        std = np.std(clean_data, ddof=1)
        data = clean_data
else:
    st.success("No outliers detected — no removal necessary.")

# ──────────────────────────────────────────────
# Empirical Rule Chart (68–95–99.7%) using dataset μ and σ
# ──────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Empirical Rule — 68%, 95%, 99.7% Coverage")

x = np.linspace(mean - 4*std, mean + 4*std, 1000)
y = norm.pdf(x, mean, std)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color="blue", lw=2)

# Shade the ±1σ, ±2σ, ±3σ regions
ax.fill_between(x, 0, y, where=(x > mean - std) & (x < mean + std),
                color="skyblue", alpha=0.5, label="68% within ±1σ")
ax.fill_between(x, 0, y, where=(x > mean - 2*std) & (x < mean + 2*std),
                color="lightblue", alpha=0.3, label="95% within ±2σ")
ax.fill_between(x, 0, y, where=(x > mean - 3*std) & (x < mean + 3*std),
                color="powderblue", alpha=0.2, label="99.7% within ±3σ")

# Vertical lines at ±1σ, ±2σ, ±3σ
for i in range(-3, 4):
    x_pos = mean + i*std
    ax.axvline(x_pos, color="red", linestyle="--", lw=1)
    if i != 0:
        ax.text(x_pos, 0.01, f"{i:+d}σ", color="darkred",
                ha="center", fontsize=9, fontweight="bold")

# Mean line and label
ax.axvline(mean, color="black", lw=1.5)
ax.text(mean, max(y)*0.93, "Mean (μ)", ha="center", fontsize=10, fontweight="bold")

# Labels for coverage
ax.text(mean, max(y)*0.35, "68%", ha="center", fontsize=11, color="navy", fontweight="bold")
ax.text(mean, max(y)*0.20, "95%", ha="center", fontsize=11, color="navy", fontweight="bold")
ax.text(mean, max(y)*0.08, "99.7%", ha="center", fontsize=11, color="navy", fontweight="bold")

# Style
ax.set_title("Normal Distribution — Empirical Rule (68–95–99.7%)", fontsize=12)
ax.set_xlabel("Value (X)")
ax.set_ylabel("Density")
ax.set_xlim(mean - 4*std, mean + 4*std)
ax.set_ylim(0, max(y)*1.1)
ax.grid(False)
st.pyplot(fig)

# Explanation text
st.markdown("""
The **Empirical Rule** (or **68–95–99.7 Rule**) describes how data behave in a normal distribution:

- About **68%** of values lie within **±1 standard deviation** of the mean  
- About **95%** lie within **±2 standard deviations**  
- About **99.7%** lie within **±3 standard deviations**  

Values beyond ±3σ are typically considered **outliers** in classical Z-score analysis.
""")

st.caption("Educational demo • Classic Z-Score method • Generated by GPT-5 · © Alastair McBride 2025")
