#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      gunee
#
# Created:     01/08/2025
# Copyright:   (c) gunee 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import load_iris, make_regression
import matplotlib.pyplot as plt

# CART Classification Example - Iris Dataset (Real-life: Species identification)
print("=== CART Classification: Flower Species Identification ===")
iris = load_iris()
X, y = iris.data, iris.target

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train CART classifier
cart_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
cart_classifier.fit(X_train, y_train)

# Make predictions
y_pred = cart_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Training samples: {len(X_train)}")
print(f"Test accuracy: {accuracy:.3f}")
print(f"Feature importance: {dict(zip(iris.feature_names, cart_classifier.feature_importances_))}")

# CART Regression Example - House Price Prediction
print("\n=== CART Regression: House Price Prediction ===")

# Generate synthetic house data (features: size, rooms, age)
np.random.seed(42)
X_house = np.random.rand(1000, 3) * [3000, 8, 50]  # [sq_ft, rooms, age]
y_house = 100000 + 150 * X_house[:, 0] + 20000 * X_house[:, 1] - 1000 * X_house[:, 2] + np.random.normal(0, 10000, 1000)

# Split the house data: 80% training, 20% testing
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_house, y_house, test_size=0.2, random_state=42)

# Train CART regressor
cart_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
cart_regressor.fit(X_train_h, y_train_h)

# Predictions and evaluation
y_pred_h = cart_regressor.predict(X_test_h)
mse = mean_squared_error(y_test_h, y_pred_h)
rmse = np.sqrt(mse)

print(f"RMSE: ${rmse:,.0f}")
print(f"Feature importance: Size={cart_regressor.feature_importances_[0]:.3f}, "
      f"Rooms={cart_regressor.feature_importances_[1]:.3f}, "
      f"Age={cart_regressor.feature_importances_[2]:.3f}")
