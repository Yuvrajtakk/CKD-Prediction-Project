# ckd_initial_inspection.py

# -------------------------------------
#  Load Dataset and Perform Inspection
#  + Handle Missing Values
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
    print("❌ Error: File not found. Make sure 'Chronic_Kidney_Dsease_data.csv' is in the same folder.")
    exit()

# 🔹 First 5 rows
print("\n🔹 First 5 rows:")
print(df.head())

# 🔹 Dataset shape
print("\n🔹 Dataset shape:", df.shape)

# 🔹 Column names
print("\n🔹 Columns:")
print(df.columns.tolist())

# 🔹 Data types and nulls
print("\n🔹 Data Types and Nulls:")
df.info()

# 🔹 Numerical summary
print("\n🔹 Numerical Summary:")
print(df.describe())

# 🔹 Categorical summary
print("\n🔹 Categorical Summary:")
print(df.describe(include='object'))

# 🔹 Missing values
print("\n🔹 Missing values per column:")
print(df.isnull().sum())

# 🔹 Heatmap of missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap (Before Cleaning)")
plt.tight_layout()
plt.show()

# -------------------------------------
# 🔧 Handle Missing Values
# -------------------------------------
print("\n🔧 Handling missing values...")

# Fill numeric with median
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 🔁 Check again
print("\n✅ Missing values handled.")
print(df.isnull().sum())

# 🔹 Save cleaned data
df.to_csv("cleaned_ckd_data.csv", index=False)
print("\n💾 Cleaned dataset saved as 'cleaned_ckd_data.csv'.")
