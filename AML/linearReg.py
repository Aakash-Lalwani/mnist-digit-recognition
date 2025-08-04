#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      gunee
#
# Created:     01/08/2025
# Copyright:   (c) gunee 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Simple Linear Regression - Salary Prediction
print("=== Linear Regression: Salary Prediction ===")
# Create realistic salary data: salary increases with experience
np.random.seed(42)
experience = np.random.uniform(0, 15, 1000).reshape(-1, 1)  # Years of experience
salary = 40000 + 3000 * experience.flatten() + np.random.normal(0, 5000, 1000)  # Base + growth + noise

# Split the data: 80% training, 20% testing
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(experience, salary, test_size=0.2, random_state=42)

# Train Linear Regression
lr_simple = LinearRegression()
lr_simple.fit(X_train_s, y_train_s)

# Predictions and metrics
y_pred_s = lr_simple.predict(X_test_s)
r2 = r2_score(y_test_s, y_pred_s)
mae = mean_absolute_error(y_test_s, y_pred_s)

print(f"R² Score: {r2:.3f} (explains {r2 * 100:.1f}% of variance)")
print(f"Mean Absolute Error: ${mae:,.0f}")
print(f"Equation: Salary = ${lr_simple.intercept_:,.0f} + ${lr_simple.coef_[0]:,.0f} × Experience")

# Multiple Linear Regression - House Price Prediction
print("\n=== Multiple Linear Regression: House Price Prediction ===")
# More realistic house features
np.random.seed(42)
n_houses = 1000
house_data = {
    'sqft': np.random.uniform(800, 4000, n_houses),
    'bedrooms': np.random.randint(1, 6, n_houses),
    'age': np.random.uniform(0, 50, n_houses),
    'location_score': np.random.uniform(1, 10, n_houses)  # 1=rural, 10=city center
}

# Create realistic price relationship
house_prices = (
    50000 +  # Base price
    100 * house_data['sqft'] +  # $100 per sqft
    15000 * house_data['bedrooms'] +  # $15k per bedroom
    -500 * house_data['age'] +  # Depreciation
    25000 * house_data['location_score'] +  # Location premium
    np.random.normal(0, 20000, n_houses)  # Market noise
)

# Prepare data
X_house_multi = np.column_stack([house_data['sqft'], house_data['bedrooms'],
                                  house_data['age'], house_data['location_score']])
feature_names = ['SqFt', 'Bedrooms', 'Age', 'Location_Score']

# Split the house data: 80% training, 20% testing
X_train_hm, X_test_hm, y_train_hm, y_test_hm = train_test_split(
    X_house_multi, house_prices, test_size=0.2, random_state=42)

# Standardize features for better interpretation
scaler = StandardScaler()
X_train_hm_scaled = scaler.fit_transform(X_train_hm)
X_test_hm_scaled = scaler.transform(X_test_hm)

# Train model
lr_multi = LinearRegression()
lr_multi.fit(X_train_hm_scaled, y_train_hm)

# Evaluate
y_pred_hm = lr_multi.predict(X_test_hm_scaled)
r2_multi = r2_score(y_test_hm, y_pred_hm)
mae_multi = mean_absolute_error(y_test_hm, y_pred_hm)

print(f"R² Score: {r2_multi:.3f}")
print(f"Mean Absolute Error: ${mae_multi:,.0f}")
print("Feature Coefficients (standardized):")
for name, coef in zip(feature_names, lr_multi.coef_):
    print(f"  {name}: {coef:,.0f}")

# Show example prediction
print(f"\nExample prediction:")
print(f"Actual price: ${y_test_hm[0]:,.0f}")
print(f"Predicted price: ${y_pred_hm[0]:,.0f}")
