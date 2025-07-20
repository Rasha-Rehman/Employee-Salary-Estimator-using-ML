# ✅ clean_data.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("adult 3.csv")

# Drop rows with missing target
if 'income' not in df.columns:
    raise ValueError("Target column 'income' is missing in the dataset")
df.dropna(subset=['income'], inplace=True)

# Keep only required columns
features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'gender', 'hours-per-week']
X = df[features]
y = df['income']

# Save for app reference
df[features + ['income']].to_csv("clean_data.csv", index=False)
print("✅ clean_data.csv generated successfully")
