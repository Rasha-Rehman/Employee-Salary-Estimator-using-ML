# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load clean data
df = pd.read_csv("clean_data.csv")

X = df.drop("income", axis=1)
y = df["income"]

# Categorical and numerical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'gender']
numeric_cols = ['age', 'hours-per-week']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough"
)

# Model pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Train/test split and fit
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "best_model.pkl")
print("✅ Model trained and saved as best_model.pkl")
