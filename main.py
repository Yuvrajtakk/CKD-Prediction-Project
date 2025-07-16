# ckd_initial_inspection.py

# -------------------------------------
#  Load Dataset and Perform Inspection
# Project: Early Prediction of CKD
# By: Deep (Lead Data Analyst, The Intellectuals)
# -------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Load the dataset
try:
    df = pd.read_csv("Chronic_Kidney_Dsease_data.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: File not found. Make sure 'Chronic_Kidney_Disease_data.csv' is in the same folder.")
    exit()

#  View first 5 rows
print("\n🔹 First 5 rows:")
print(df.head())

#  Dataset shape
print("\n🔹 Dataset shape:", df.shape)

#  Column names
print("\n🔹 Columns:")
print(df.columns.tolist())

#  Info about datatypes and missing values
print("\n🔹 Data Types and Nulls:")
df.info()

#  Numerical summary
print("\n🔹 Numerical Summary:")
print(df.describe())

#  Categorical summary
print("\n🔹 Categorical Summary:")
print(df.describe(include='object'))

#  Missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

#  Visualize missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.show()
