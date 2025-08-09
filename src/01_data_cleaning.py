# src/01_data_cleaning.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
# Define file paths
# This makes the script runnable from anywhere in the project
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "kidney_disease.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_ckd_data.csv")

# --- Data Loading ---
print("🔹 Loading the dataset...")
try:
    # We assume the raw data CSV is in a /data folder at the project root
    df = pd.read_csv(RAW_DATA_PATH)
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: File not found at {RAW_DATA_PATH}")
    print("Please make sure 'kidney_disease.csv' is inside a 'data' folder at the project root.")
    exit()

# --- Initial Inspection (Optional, for debugging) ---
print("\n🔹 Initial Data Inspection:")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values before cleaning:\n", df.isnull().sum())

# --- Handle Missing Values ---
print("\n🔧 Handling missing values...")

# Fill numeric columns with their median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with their mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n✅ Missing values handled.")
print("Missing values after cleaning:\n", df.isnull().sum())

# --- Save Cleaned Data ---
# Ensure the /data directory exists
os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)

df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"\n💾 Cleaned dataset saved to '{CLEANED_DATA_PATH}'.")