import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# HOUSE DATASET – LOAD CLEANED DATA
print("\n" + "=" * 50)
print("HOUSE DATASET – LOAD CLEANED DATA (TASK 3)")
print("=" * 50)
# Get base directory (this script location)
BASE_DIR = Path(__file__).resolve().parent
# Path to cleaned dataset (from Task 1)
DATA_PATH = BASE_DIR.parent / "Task_1" / "house_prediction_cleaned.csv"
print("Loading from:", DATA_PATH)

# Load dataset
house_df = pd.read_csv(DATA_PATH)
print("Shape:", house_df.shape)
print("\nColumns:")
print(house_df.columns.tolist())
print("\nFirst 5 rows:")
print(house_df.head())

# CREATE PLOTS DIRECTORY
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
print("\nPlots will be saved to:", PLOTS_DIR)

# BAR PLOT – AVERAGE HOUSE PRICE BY CHAS
print("\n" + "=" * 50)
print("HOUSE DATASET – BAR PLOT (MEDV BY CHAS)")
print("=" * 50)
# Calculate average price by CHAS
avg_price_by_chas = house_df.groupby("CHAS")["MEDV"].mean()
print(avg_price_by_chas)
# Create bar plot
plt.figure(figsize=(8, 5))
sns.barplot(
    x=avg_price_by_chas.index,
    y=avg_price_by_chas.values
)
# Customize plot
plt.title("Average House Price by Charles River Proximity")
plt.xlabel("Borders Charles River (0 = No, 1 = Yes)")
plt.ylabel("Average House Price (MEDV)")
plt.tight_layout()
# Save plot
plot_path = PLOTS_DIR / "avg_house_price_by_chas.png"
plt.savefig(plot_path)
plt.show()
print("Bar plot saved to:", plot_path)

# LINE CHART – HOUSE PRICE VS AVERAGE ROOMS
print("\n" + "=" * 50)
print("HOUSE DATASET – LINE CHART (MEDV VS RM)")
print("=" * 50)
# Sort values for proper line plotting
rm_medv_df = house_df.sort_values(by="RM")
# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(
    rm_medv_df["RM"],
    rm_medv_df["MEDV"]
)
# Customize plot
plt.title("House Price vs Average Number of Rooms")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price (MEDV)")
plt.tight_layout()
# Save plot
plot_path = PLOTS_DIR / "house_price_vs_rooms.png"
plt.savefig(plot_path)
plt.show()
print("Line chart saved to:", plot_path)

# SCATTER PLOT – HOUSE PRICE VS LSTAT
print("\n" + "=" * 50)
print("HOUSE DATASET – SCATTER PLOT (MEDV VS LSTAT)")
print("=" * 50)
plt.figure(figsize=(10, 6))
plt.scatter(
    house_df["LSTAT"],
    house_df["MEDV"],
    alpha=0.7
)
# Customize plot
plt.title("House Price vs Lower Status Population (LSTAT)")
plt.xlabel("LSTAT (% Lower Status Population)")
plt.ylabel("House Price (MEDV)")
plt.tight_layout()
# Save plot
plot_path = PLOTS_DIR / "house_price_vs_lstat.png"
plt.savefig(plot_path)
plt.show()
print("Scatter plot saved to:", plot_path)

# CORRELATION MATRIX
print("\n" + "=" * 50)
print("HOUSE DATASET – CORRELATION MATRIX")
print("=" * 50)
correlation_matrix = house_df.corr()
# Display correlation with target variable (MEDV)
medv_corr = correlation_matrix["MEDV"].sort_values(ascending=False)
print("\nCorrelation with MEDV:\n")
print(medv_corr)

# CORRELATION HEATMAP
print("\n" + "=" * 50)
print("HOUSE DATASET – CORRELATION HEATMAP")
print("=" * 50)
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)
plt.title("Correlation Heatmap – House Dataset")
plt.tight_layout()
# Save heatmap
heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
plt.savefig(heatmap_path)
plt.show()
print("Correlation heatmap saved to:", heatmap_path)

# DISTRIBUTION PLOT – HOUSE PRICE (MEDV)
print("\n" + "=" * 50)
print("HOUSE DATASET – DISTRIBUTION PLOT (MEDV HISTOGRAM)")
print("=" * 50)
plt.figure(figsize=(10, 6))
sns.histplot(
    house_df["MEDV"],
    bins=30,
    kde=True
)
# Customize plot
plt.title("Distribution of Median House Prices (MEDV)")
plt.xlabel("Median House Price (MEDV)")
plt.ylabel("Frequency")
plt.tight_layout()
# Save plot
plot_path = PLOTS_DIR / "medv_distribution_histogram.png"
plt.savefig(plot_path)
plt.show()
print("Histogram saved to:", plot_path)

# COUNT PLOT – CHAS VARIABLE
print("\n" + "=" * 50)
print("HOUSE DATASET – COUNT PLOT (CHAS)")
print("=" * 50)
plt.figure(figsize=(6, 5))
sns.countplot(
    x="CHAS",
    data=house_df
)
# Customize plot
plt.title("Distribution of Houses Near Charles River")
plt.xlabel("Borders Charles River (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
# Save plot
plot_path = PLOTS_DIR / "chas_countplot.png"
plt.savefig(plot_path)
plt.show()
print("Count plot saved to:", plot_path)

