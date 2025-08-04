#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      gunee
#
# Created:     02/08/2025
# Copyright:   (c) gunee 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#SVM using IRIS dataset
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noise to the features
noise = np.random.normal(0, 0.5, X.shape)  # Mean 0, standard deviation 0.5
X_noisy = X + noise

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


