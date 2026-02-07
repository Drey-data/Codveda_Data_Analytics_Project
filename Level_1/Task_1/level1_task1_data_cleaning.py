import pandas as pd

# Load the stock price dataset
df = pd.read_csv("2) Stock Prices Data Set.csv")

# Check the first few rows of the dataset to ensure it's loaded
print(df.head())

# Check number of rows and columns
print(df.shape)

# Check column name and data type
print(df.info())

# Check missing values per column
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

# Verify missing values are removed
print(df.isnull().sum())
print(df.shape)

# Check for duplicate rows
duplicates = df.duplicated().sum()
print("Number of duplicate rows:", duplicates)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])
# Confirm the change
print(df.dtypes)

# Sort dataset by date
df = df.sort_values(by='date')
# Reset index after sorting
df = df.reset_index(drop=True)
# Confirm sorting
print(df.head())
print(df.tail())

# Final dataset overview
print("Final shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Save the cleaned dataset
df.to_csv("Level_1/stock_prices_cleaned.csv", index=False)
print("Cleaned dataset saved successfully.")

#Load House Prediction Dataset
print("\n" + "="*50)
print("HOUSE PREDICTION DATASET – RAW LOAD")
print("="*50)

print("\n" + "="*50)
print("HOUSE PREDICTION DATASET – RAW LOAD (FIXED)")
print("="*50)

# Load whitespace-separated dataset with no header
house_df = pd.read_csv(
    "4) house Prediction Data Set.csv",
    sep=r"\s+",
    header=None
    )
print("Shape:", house_df.shape)
print("\nFirst 5 rows:")
print(house_df.head())

# Assign column names
house_df.columns = [
    "CRIM",    # Per capita crime rate
    "ZN",      # Residential land zoned
    "INDUS",   # Non-retail business acres
    "CHAS",    # Charles River dummy variable
    "NOX",     # Nitric oxides concentration
    "RM",      # Average rooms per dwelling
    "AGE",     # Owner-occupied units built before 1940
    "DIS",     # Distance to employment centres
    "RAD",     # Accessibility to highways
    "TAX",     # Property tax rate
    "PTRATIO", # Pupil-teacher ratio
    "B",       # Proportion of Black residents
    "LSTAT",   # % lower status population
    "MEDV"     # Median home value (target) 
]
print("\nColumn names assigned successfully:")
print(house_df.columns)

#Check for Missing Rows
print("\nMissing values per column:")
print(house_df.isnull().sum())

#Check for Data Types
print("\nData types:")
print(house_df.dtypes)

#Check Dataset Information
print("\nDataset info:")
print(house_df.info())

# Basic Statistical Summary
print("\nBasic statistical summary:")
print(house_df.describe())

# Unique Values in CHAS Column
print("\nUnique values in CHAS:")
print(house_df["CHAS"].unique())

# Unique Values in RAD Column
print("\nUnique values in RAD:")
print(sorted(house_df["RAD"].unique()))

# Save the cleaned House Prediction dataset
print("\n" + "="*50)
print("SAVING HOUSE PREDICTION CLEANED DATASET")
print("="*50)
house_df.to_csv("Level_1/house_prediction_cleaned.csv", index=False)
print("House prediction cleaned dataset saved successfully.")

# Load Churn Dataset (churn-bigml-20)
print("\n" + "="*50)
print("CHURN DATASET – RAW LOAD")
print("="*50)
churn_df = pd.read_csv("churn-bigml-20.csv")
print("Shape:", churn_df.shape)
print("\nFirst 5 rows:")
print(churn_df.head())
print("\nColumn names:")
print(churn_df.columns)

# Check for Missing Values
print("\n" + "="*50)
print("CHURN DATASET – MISSING VALUES CHECK")
print("="*50)
print(churn_df.isnull().sum())

# Check for Duplicate Rows
print("\n" + "="*50)
print("CHURN DATASET – DUPLICATE CHECK")
print("="*50)
print("Number of duplicate rows:", churn_df.duplicated().sum())

# Check Data Types & Format Consistency
print("\n" + "="*50)
print("CHURN DATASET – DATA TYPES")
print("="*50)
print(churn_df.dtypes)

# Standardize Column Names
churn_df.columns = churn_df.columns.str.strip().str.lower().str.replace(" ", "_")
print("\nUpdated column names:")
print(churn_df.columns)

# Save Cleaned Churn Dataset
churn_df.to_csv("Level_1/churn-bigml-20_cleaned.csv", index=False)
print("\nCleaned churn dataset saved successfully.")

# Load Churn Dataset (churn-bigml-80)
print("\n" + "="*50)
print("CHURN DATASET (80%) – RAW LOAD")
print("="*50)
churn80_df = pd.read_csv("churn-bigml-80.csv")
print("Shape:", churn80_df.shape)
print("\nFirst 5 rows:")
print(churn80_df.head())
print("\nColumn names:")
print(churn80_df.columns)

# Check for Missing Values
print("\n" + "="*50)
print("CHURN DATASET (80%) – MISSING VALUES CHECK")
print("="*50)
print(churn80_df.isnull().sum())

# Check for Duplicate Rows
print("\n" + "="*50)
print("CHURN DATASET (80%) – DUPLICATE CHECK")
print("="*50)
print("Number of duplicate rows:", churn80_df.duplicated().sum())

# Check Data Types
print("\n" + "="*50)
print("CHURN DATASET (80%) – DATA TYPES")
print("="*50)
print(churn80_df.dtypes)

# Check Data Types
print("\n" + "="*50)
print("CHURN DATASET (80%) – DATA TYPES")
print("="*50)
print(churn80_df.dtypes)

# Standardize Column Names
churn80_df.columns = (
    churn80_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
print("\nUpdated column names:")
print(churn80_df.columns)

# Save the Cleaned Dataset
churn80_df.to_csv("Level_1/churn_bigml_80_cleaned.csv", index=False)
print("Cleaned churn (80%) dataset saved successfully.")

# Load Iris Dataset (1) iris)
print("\n" + "="*50)
print("IRIS DATASET – RAW LOAD")
print("="*50)
iris_df = pd.read_csv("1) iris.csv")
print("Shape:", iris_df.shape)
print("\nFirst 5 rows:")
print(iris_df.head())
print("\nColumn names:")
print(iris_df.columns)

# Check for Missing Values
print("\n" + "="*50)
print("IRIS DATASET – MISSING VALUES CHECK")
print("="*50)
print(iris_df.isnull().sum())

# Check for Duplicate Rows
print("\n" + "="*50)
print("IRIS DATASET – DUPLICATE CHECK")
print("="*50)
print("Number of duplicate rows:", iris_df.duplicated().sum())

# Remove Duplicate Rows
iris_df = iris_df.drop_duplicates()
print("\nDuplicates removed.")
print("New shape:", iris_df.shape)

# Check Data Types
print("\n" + "="*50)
print("IRIS DATASET – DATA TYPES")
print("="*50)
print(iris_df.dtypes)

# Save Cleaned Iris Dataset
iris_df.to_csv("Level_1/iris_cleaned.csv", index=False)
print("Cleaned Iris dataset saved successfully.")

# Load Sentiment Dataset (Raw)
print("\n" + "="*50)
print("SENTIMENT DATASET – RAW LOAD")
print("="*50)
sentiment_df = pd.read_csv("3) Sentiment dataset.csv")
print("Shape:", sentiment_df.shape)
print("\nFirst 5 rows:")
print(sentiment_df.head())
print("\nColumn names:")
print(sentiment_df.columns)

# Remove Unnecessary Index Columns
print("\n" + "="*50)
print("SENTIMENT DATASET – DROP UNUSED COLUMNS")
print("="*50)
sentiment_df.drop(columns=["Unnamed: 0", "Unnamed: 0.1"], inplace=True)
print("Updated columns:")
print(sentiment_df.columns)

# Check for Missing Values
print("\n" + "="*50)
print("SENTIMENT DATASET – MISSING VALUES CHECK")
print("="*50)
print(sentiment_df.isnull().sum())

# Check for Duplicate Rows
print("\n" + "="*50)
print("SENTIMENT DATASET – DUPLICATE CHECK")
print("="*50)
print("Number of duplicate rows:", sentiment_df.duplicated().sum())

# Remove Duplicate Rows
print("\n" + "="*50)
print("SENTIMENT DATASET – REMOVE DUPLICATES")
print("="*50)
sentiment_df = sentiment_df.drop_duplicates()
print("New shape after removing duplicates:", sentiment_df.shape)

# Check Data Types
print("\n" + "="*50)
print("SENTIMENT DATASET – DATA TYPES")
print("="*50)
print(sentiment_df.dtypes)

# Save Cleaned Sentiment Dataset
sentiment_df.to_csv("Level_1/sentiment_cleaned.csv", index=False)
print("Cleaned sentiment dataset saved successfully.")

