import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("dataset/chronickidneydisease.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Drop id column if exists
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Target column
target = 'classification'

# ✅ USE REAL COLUMN NAMES FROM DATASET
selected_features = [
    'rbc',   # red blood cells
    'pc',    # pus cell
    'bgr',   # blood glucose random
    'bu',    # blood urea
    'pe',    # pedal edema
    'ane',   # anemia
    'dm',    # diabetes mellitus
    'cad'    # coronary artery disease
]

X = df[selected_features]
y = df[target]

# Convert target
y = y.apply(lambda x: 1 if str(x).strip() == 'ckd' else 0)

# Column types
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline.fit(X_train, y_train)

# Save
pickle.dump(pipeline, open("pipeline.pkl", "wb"))

print("✅ Pipeline trained and saved as pipeline.pkl")