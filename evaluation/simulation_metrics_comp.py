import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os

# ==============================
# Configuration
# ==============================

DAY_FILE = "/home/vgtu/simulation_metrics_day.csv"
NIGHT_FILE = "/home/vgtu/simulation_metrics_night.csv"
OUTPUT_DIR = "day_vs_night_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Load data
# ==============================

day = pd.read_csv(DAY_FILE)
night = pd.read_csv(NIGHT_FILE)

print("Day shape:", day.shape)
print("Night shape:", night.shape)

# ==============================
# Find common numeric columns
# ==============================

numeric_day = day.select_dtypes(include=np.number)
numeric_night = night.select_dtypes(include=np.number)

common_columns = sorted(
    set(numeric_day.columns).intersection(
        set(numeric_night.columns)
    )
)

summary = []

# ==============================
# Statistics
# ==============================

for col in common_columns:

    d = numeric_day[col].dropna()
    n = numeric_night[col].dropna()

    if len(d) < 2 or len(n) < 2:
        continue

    t_stat, p_value = ttest_ind(
        d,
        n,
        equal_var=False
    )

    day_mean = d.mean()
    night_mean = n.mean()

    if abs(day_mean) > 1e-9:
        percent_change = (
            (night_mean - day_mean)
            / abs(day_mean)
        ) * 100
    else:
        percent_change = np.nan

    summary.append({
        "Metric": col,

        "Day Mean": day_mean,
        "Night Mean": night_mean,

        "Day Std": d.std(),
        "Night Std": n.std(),

        "Day Min": d.min(),
        "Night Min": n.min(),

        "Day Max": d.max(),
        "Night Max": n.max(),

        "% Change": percent_change,

        "t statistic": t_stat,
        "p value": p_value,

        "Significant": p_value < 0.05
    })

# ==============================
# Save statistics
# ==============================

summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values("p value")

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "day_vs_night_statistics.csv"
    ),
    index=False
)

print(summary_df)

# ==============================
# Generate plots
# ==============================

for col in common_columns:

    d = numeric_day[col].dropna()
    n = numeric_night[col].dropna()

    if len(d) == 0 or len(n) == 0:
        continue

    plt.figure(figsize=(7,5))

    plt.boxplot(
        [d, n],
        labels=["Day", "Night"]
    )

    plt.ylabel(col)
    plt.title(f"Day vs Night : {col}")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{col}_boxplot.png"
        ),
        dpi=300
    )

    plt.close()

# ==============================
# Mean comparison bar chart
# ==============================

plt.figure(figsize=(12,6))

x = np.arange(len(summary_df))

width = 0.35

plt.bar(
    x - width/2,
    summary_df["Day Mean"],
    width,
    label="Day"
)

plt.bar(
    x + width/2,
    summary_df["Night Mean"],
    width,
    label="Night"
)

plt.xticks(
    x,
    summary_df["Metric"],
    rotation=90
)

plt.ylabel("Mean")

plt.title("Mean Metric Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "mean_comparison.png"
    ),
    dpi=300
)

plt.close()

print("\nResults saved to:", OUTPUT_DIR)