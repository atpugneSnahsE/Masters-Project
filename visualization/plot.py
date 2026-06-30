import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ========================= CONFIG =========================
DATA_ROOT = "lane_dataset"
JSON_PATH = os.path.join(DATA_ROOT, "class_distribution_profile.json")

# Define clean, human-readable labels for your paper figures
CLEAN_NAMES = {
    "CLASS_BG": "Background / Sky / Buildings",
    "CLASS_EGO": "Ego Lane Road Surface",
    "CLASS_LEFT_DASHED": "Left Dashed Lane Line",
    "CLASS_LEFT_SOLID": "Left Solid Lane Line",
    "CLASS_RIGHT_DASHED": "Right Dashed Lane Line",
    "CLASS_RIGHT_SOLID": "Right Solid Lane Line",
    "CLASS_OTHER": "Other Vehicles / Obstacles",
    "CLASS_STOP": "Stop Line Markings",
    "CLASS_CROSS": "Crosswalk Markings"
}

# ==========================================================

# 1. LOAD & PREPROCESS CLASS FREQUENCY DATA
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Could not find {JSON_PATH}. Please run the profiling script first.")

with open(JSON_PATH, "r") as f:
    raw_data = json.load(f)

# Convert to DataFrame for easier manipulation and sorting
data_list = []
for k, v in raw_data.items():
    data_list.append({
        "Class": CLEAN_NAMES.get(k, k),
        "Pixel Count": v["pixel_count"],
        "Percentage": v["percentage"]
    })

df = pd.DataFrame(data_list)

# Rule: Always sort bars for clarity (ascending order for horizontal bar chart)
df = df.sort_values(by="Pixel Count", ascending=True)

# 2. GENERATE CLASS IMBALANCE BAR CHART
plt.style.use("seaborn-v0_8-whitegrid")

# Increased horizontal width to 13 inches to give text labels breathing room
fig, ax = plt.subplots(figsize=(13, 6.5))

colors = sns.color_palette("viridis", len(df))
bars = ax.barh(df["Class"], df["Pixel Count"], color=colors, edgecolor="none", height=0.55)

# Set logarithmic scale to manage extreme pixel imbalances gracefully
ax.set_xscale("log")

# Aesthetic adjustments
ax.set_title("CARLA Semantic Segmentation Class Distribution (Log Scale)", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Total Pixel Frequency Count (Log Scale)", fontsize=11, fontweight="bold", labelpad=10)
ax.set_ylabel("Semantic Target Class", fontsize=11, fontweight="bold", labelpad=10)

# Increase font size of class labels slightly for clear presentation reading
ax.tick_params(axis='y', labelsize=11)

# Annotate each bar with its exact percentage representation
for bar, pct in zip(bars, df["Percentage"]):
    width = bar.get_width()
    # Using 1.25 multiplier ensures text safely clears the end of the log bar
    ax.text(width * 1.25, bar.get_y() + bar.get_height()/2, f"{pct:.3f}%", 
            va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')

# Clean up layout borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# --- THE FIX FOR CLIPPING ---
# Force explicit layout padding: left=0.25 guarantees 25% width allocation for text strings
plt.subplots_adjust(left=0.25, right=0.90, top=0.90, bottom=0.12)

chart_out = os.path.join(DATA_ROOT, "class_frequency_imbalance.png")
plt.savefig(chart_out, dpi=300)
plt.close()
print(f"🎉 Bar chart figure saved successfully with fixed dimensions to: {chart_out}")