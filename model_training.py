import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load CSV file
data = pd.read_csv('insurance_data_linear.csv')

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Training evaluation
y_pred_train = model.predict(X_train)
print("✅ Training MSE:", mean_squared_error(y_train, y_pred_train))
print("✅ Training R²:", r2_score(y_train, y_pred_train))

# Test evaluation
y_pred_test = model.predict(X_test)
print("\n✅ Test MSE:", mean_squared_error(y_test, y_pred_test))
print("✅ Test R²:", r2_score(y_test, y_pred_test))

# Save model
joblib.dump(model, 'linear_model.pkl')
print("✅ Model trained and saved!")