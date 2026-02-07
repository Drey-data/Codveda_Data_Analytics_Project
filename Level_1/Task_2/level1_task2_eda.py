from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# HOUSE DATASET – LOAD CLEANED DATA
print("\n" + "=" * 50)
print("HOUSE DATASET – LOAD CLEANED DATA")
print("=" * 50)
# Path handling (robust & OS-safe)
BASE_DIR = Path(__file__).resolve().parent      # Level_1/Task_2
LEVEL_1_DIR = BASE_DIR.parent                  # Level_1
PROJECT_ROOT = LEVEL_1_DIR.parent              # Codveda_Data_Analytics

HOUSE_DATA_PATH = (
    LEVEL_1_DIR
    / "Task_1"
    / "house_prediction_cleaned.csv"
)
print("Loading from:", HOUSE_DATA_PATH)

# Load dataset
house_df = pd.read_csv(HOUSE_DATA_PATH)
print("Shape:", house_df.shape)
print("\nFirst 5 rows:")
print(house_df.head())

# HOUSE DATASET – SUMMARY STATISTICS
print("\n" + "=" * 50)
print("HOUSE DATASET – SUMMARY STATISTICS")
print("=" * 50)
summary_stats = house_df.describe()
print(summary_stats)

# House dataset – Histograms
print("\n" + "="*50)
print("HOUSE DATASET – HISTOGRAMS")
print("="*50)
house_df.hist(figsize=(14, 10), bins=20)
plt.tight_layout()
plt.show()

# House dataset – Boxplots (Selected Features)
print("\n" + "="*50)
print("HOUSE DATASET – BOXPLOTS (SELECTED FEATURES)")
print("="*50)
features = ["CRIM", "RM", "LSTAT", "MEDV"]
plt.figure(figsize=(10, 6))
house_df[features].boxplot()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# House dataset – Scatter plots
print("\n" + "="*50)
print("HOUSE DATASET – SCATTER PLOTS")
print("="*50)
# RM vs MEDV
plt.figure(figsize=(6, 4))
plt.scatter(house_df["RM"], house_df["MEDV"])
plt.xlabel("Average number of rooms (RM)")
plt.ylabel("Median value of homes (MEDV)")
plt.title("RM vs MEDV")
plt.tight_layout()
plt.show()
# LSTAT vs MEDV
plt.figure(figsize=(6, 4))
plt.scatter(house_df["LSTAT"], house_df["MEDV"])
plt.xlabel("Lower status population (LSTAT)")
plt.ylabel("Median value of homes (MEDV)")
plt.title("LSTAT vs MEDV")
plt.tight_layout()
plt.show()

# House dataset – Correlation heatmap
print("\n" + "="*50)
print("HOUSE DATASET – CORRELATION HEATMAP")
print("="*50)
plt.figure(figsize=(12, 8))
corr_matrix = house_df.corr()
sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap of House Dataset")
plt.tight_layout()
plt.show()

