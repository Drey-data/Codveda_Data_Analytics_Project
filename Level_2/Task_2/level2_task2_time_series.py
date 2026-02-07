# LEVEL 2 – TASK 2: TIME SERIES ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose


print("\n" + "=" * 50)
print("LEVEL 2 – TASK 2: TIME SERIES ANALYSIS")
print("=" * 50)

# Base directory (this script location)
BASE_DIR = Path(__file__).resolve().parent

# Path to cleaned stock price dataset (from Level 1, Task 1)
DATA_PATH = (
    BASE_DIR.parent.parent
    / "Level_1"
    / "Task_1"
    / "stock_prices_cleaned.csv"
)
print("Loading dataset from:")
print(DATA_PATH)

# Safety check (very important)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

# Load dataset
df = pd.read_csv(DATA_PATH)
print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

SYMBOL = "AAPL"   # you can change this to any symbol in the dataset

# FILTER DATA FOR ONE STOCK & PREPARE TIME SERIES
stock_df = df[df["symbol"] == SYMBOL].copy()

# Convert date column to datetime
stock_df["date"] = pd.to_datetime(stock_df["date"])

# Set date as index
stock_df.set_index("date", inplace=True)

# Sort by date
stock_df.sort_index(inplace=True)
print(f"\nSelected stock: {SYMBOL}")
print("Time range:")
print(stock_df.index.min(), "to", stock_df.index.max())

# MOVING AVERAGE SMOOTHING
print("\n" + "=" * 50)
print("MOVING AVERAGE SMOOTHING")
print("=" * 50)
stock_df["MA_30"] = stock_df["close"].rolling(window=30).mean()
stock_df["MA_90"] = stock_df["close"].rolling(window=90).mean()
plt.figure(figsize=(12, 6))
plt.plot(stock_df.index, stock_df["close"], label="Close Price", alpha=0.6)
plt.plot(stock_df.index, stock_df["MA_30"], label="30-Day MA")
plt.plot(stock_df.index, stock_df["MA_90"], label="90-Day MA")
plt.title(f"{SYMBOL} Stock Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

#SEASONAL DECOMPOSOTION
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
# Ensure the index is datetime
df.index = pd.to_datetime(df.index)

# Optional but recommended: set frequency
df = df.asfreq('D')  # Daily data

# Handle missing values (important for decomposition)
df['close'] = df['close'].interpolate()

# Perform seasonal decomposition
decomposition = seasonal_decompose(
    df['close'],
    model='additive',
    period=30  # monthly seasonality assumption
)

# Plot decomposition
decomposition.plot()
plt.suptitle("Seasonal Decomposition of Stock Closing Prices", fontsize=14)
plt.tight_layout()
plt.show()
