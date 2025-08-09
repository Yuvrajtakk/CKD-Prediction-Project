# src/02_model_training.py
# This script takes the cleaned data, trains our predictive models,
# evaluates their performance, and saves the best-performing one.

import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# --- Configuration ---
CLEANED_DATA_PATH = "data/cleaned_ckd_data.csv"
MODEL_OUTPUT_DIR = "models"

# --- Data Loading ---
print("🔹 Loading the cleaned dataset...")
df = pd.read_csv(CLEANED_DATA_PATH)
print("✅ Cleaned dataset loaded successfully.")

# --- Data Preparation ---
# Models need numbers, not text. We'll convert text labels ('ckd', 'notckd')
# into numbers (1, 0) and handle all other text columns.
print("\n🔹 Preparing data for modeling...")
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
X = pd.get_dummies(X, drop_first=True)
print("✅ Data preparation complete.")

# --- Train-Test Split ---
print("\n🔹 Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("✅ Data split successfully.")

# --- Feature Scaling ---
# We scale the features so that columns with large values (like BUN) don't
# unfairly influence the model more than columns with small values.
print("\n🔹 Scaling features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled.")

# --- Handling Imbalanced Data with SMOTE ---
# Our dataset has many more healthy patients than sick ones. SMOTE helps fix this
# by creating new, synthetic examples of the sick patients for the model to learn from.
print("\n🔹 Balancing training data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print("✅ Training data balanced.")

# --- Final Model Training ---
# After testing, Logistic Regression gave us the best balance of performance,
# especially for correctly identifying patients with CKD.
print("\n--- Training Final Logistic Regression Model ---")
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train_resampled, y_train_resampled)
y_pred = final_model.predict(X_test_scaled)

print("\n🔹 Final Model Results:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Save Model for Future Use ---
# We save the trained model and the scaler so we can make predictions
# on new data later without having to retrain everything.
print("\n💾 Saving the final model and scaler...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
joblib.dump(final_model, os.path.join(MODEL_OUTPUT_DIR, 'ckd_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl'))
print("✅ Model and scaler saved.")

print("\n🎉 Modeling pipeline complete.")
