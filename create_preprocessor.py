import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Load dataset
df = pd.read_csv("data.csv")

target_column = "charges"
X = df.drop(columns=[target_column])

# Numerical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipeline
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols)
])

# Fit preprocessor
preprocessor.fit(X)

# Save preprocessor
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.pkl")

print("✅ Preprocessor saved successfully!")