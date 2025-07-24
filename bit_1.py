import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import joblib

# Load the CSV file
df = pd.read_csv('C:/Users/Student/Downloads/Bitcoin.csv')# Ensure this file is in the same directory
print(df.head())

# Convert 'Date' to datetime and drop invalid rows
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# Extract date features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day

# Drop unused columns
df = df.drop(columns=['timestamp', 'name', 'timeOpen', 'timeClose', 'timeHigh', 'timeLow'], errors='ignore')

# Convert columns to numeric types
numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna()

# Feature-target split
X = df.drop(columns=['Close', 'Date'])  # Drop target and datetime column
y = df['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Linear Regression
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nLinear Regression Model Trained Successfully!")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Predicted values (first 5): {y_pred[:5]}")

# Random Forest with Cross-validation
rf_model = RandomForestRegressor(random_state=42)

cv_rmse = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=1
)

print("Cross-validation RMSE (before tuning):", -cv_rmse.mean())

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    rf_model,
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# Evaluate best Random Forest model
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE: {mae:.2f}")
print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Residual plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted (Residual Plot)')
plt.grid(True)
plt.show()

# Preprocessing pipeline
numeric_features = X_train.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features)
    ]
)

# Final pipeline with best model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_rf)
])
pipeline.fit(X_train, y_train)

# Full pipeline for hyperparameter tuning
full_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])
param_grid_pipeline = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10],
    'model__min_samples_split': [2, 5]
}
grid_pipeline = GridSearchCV(
    full_pipeline,
    param_grid=param_grid_pipeline,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=1  # from -1 to 1
)

grid_pipeline.fit(X_train, y_train)

print("Best Pipeline Parameters:", grid_pipeline.best_params_)

# Save the final model
best_pipeline_model = grid_pipeline.best_estimator_
joblib.dump(best_pipeline_model, 'bitcoin_price_pipeline.joblib')
print("\nModel pipeline saved as 'bitcoin_price_pipeline.joblib'")
