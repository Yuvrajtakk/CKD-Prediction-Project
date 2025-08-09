# src/02_model_training.py

import pandas as pd
import os
import joblib # <-- Import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# --- File Paths ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_ckd_data.csv")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") # <-- New folder for models

# --- Load Data ---
print("🔹 Loading cleaned dataset...")
df = pd.read_csv(CLEANED_DATA_PATH)
print("✅ Cleaned dataset loaded successfully.")

# --- Feature Engineering ---
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

# --- Scaling Features ---
print("\n🔹 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled successfully.")

# --- Handle Class Imbalance with SMOTE ---
print("\n🔹 Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"✅ SMOTE applied.")

# --- Final Model Training (Logistic Regression) ---
print("\n--- Training Final Logistic Regression Model ---")
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X_train_resampled, y_train_resampled)
y_pred = final_model.predict(X_test_scaled)

print("\n🔹 Final Model Results:")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Save the Model and Scaler (NEW STEP) ---
print("\n💾 Saving the final model and scaler...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # Create the 'models' directory if it doesn't exist
joblib.dump(final_model, os.path.join(MODEL_OUTPUT_DIR, 'ckd_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl'))
print("✅ Model and scaler saved successfully.")

print("\n🎉 Project Phase 2 Complete.")