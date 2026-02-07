# LEVEL 2 – TASK 1: REGRESSION ANALYSIS
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


print("\n" + "=" * 50)
print("LEVEL 2 – TASK 1: REGRESSION ANALYSIS")
print("=" * 50)

# Base directory (this script location)
BASE_DIR = Path(__file__).resolve().parent

# Path to cleaned house dataset from Level 1
DATA_PATH = BASE_DIR.parent.parent / "Level_1" / "Task_1" / "house_prediction_cleaned.csv"
print("Loading dataset from:")
print(DATA_PATH)

# Load dataset
house_df = pd.read_csv(DATA_PATH)
print("\nDataset shape:", house_df.shape)
print("\nColumns:")
print(house_df.columns.tolist())

# STEP 3: Define features and target
# Feature (X) and Target (y)
X = house_df[['RM']]   # Independent variable
y = house_df['MEDV']   # Dependent variable (house price)
print("\nSelected Feature: RM (Average number of rooms)")
print("Target Variable: MEDV (Median house value)")

# STEP 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
print("\nData split completed:")
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

# STEP 5: Train the Linear Regression model
# Initialize model
model = LinearRegression()
# Train model
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully!")
# Model parameters
print("\nModel Parameters:")
print("Intercept:", model.intercept_)
print("Coefficient (RM):", model.coef_[0])

# STEP 6: Make predictions and evaluate the model
# Predict on test data
y_pred = model.predict(X_test)
# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual Values")
plt.scatter(X_test, y_pred, label="Predicted Values")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median House Value (MEDV)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.show()

