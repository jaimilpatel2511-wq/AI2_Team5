import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('dataset.csv')
print("Original shape:", df.shape)

# Preprocessing
numerical_cols = df.select_dtypes(include=[np.number]).columns.drop('target')
categorical_cols = df.select_dtypes(include=['object']).columns

# Imputation
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Encoding
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Scaling
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("✅ Preprocessed shape:", df.shape)
print("✅ Missing values:", df.isnull().sum().sum())
print("\nSample:")
print(df.head())

# Save preprocessors
joblib.dump({'num_imputer': num_imputer, 'scaler': scaler, 'le_dict': le_dict}, 'preprocessors.pkl')
df.to_csv('preprocessed_data.csv', index=False)
print("✅ Preprocessing completed!")
