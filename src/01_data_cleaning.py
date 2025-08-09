# src/01_data_cleaning.py
# This script is the first step in our pipeline. It takes the raw dataset,
# cleans it by handling missing values, and saves a clean version for the next step.

import pandas as pd
import os

# --- Configuration ---
# Define file paths relative to the script's location. This makes sure
# the script works correctly no matter where the project is located.
RAW_DATA_PATH = "data/kidney_disease.csv"
CLEANED_DATA_PATH = "data/cleaned_ckd_data.csv"

# --- Data Loading ---
print("🔹 Loading the raw dataset...")
try:
    df = pd.read_csv(RAW_DATA_PATH)
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: Raw data file not found at '{RAW_DATA_PATH}'")
    print("Please make sure 'kidney_disease.csv' is inside the 'data' folder.")
    exit()

# --- Handle Missing Values ---
# Real-world data is messy. We'll fill missing numerical data with the median
# and missing text data with the mode (the most common value).
print("\n🔧 Cleaning data by handling missing values...")

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("✅ Missing values handled.")
print("Missing values after cleaning:\n", df.isnull().sum().head())


# --- Save Cleaned Data ---
# The output of this script is the input for the model training script.
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"\n💾 Cleaned dataset saved to '{CLEANED_DATA_PATH}'.")
