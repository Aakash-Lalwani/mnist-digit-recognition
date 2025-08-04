#-------------------------------------------------------------------------------
# Name:        module3
# Purpose:
#
# Author:      gunee
#
# Created:     01/08/2025
# Copyright:   (c) gunee 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------
# Importing required libraries
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Create the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play Sports?': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                     'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Preprocess the data
# Convert categorical variables to numerical
df['Outlook'] = df['Outlook'].map({'Sunny': 0, 'Overcast': 1, 'Rain': 2})
df['Temperature'] = df['Temperature'].map({'Hot': 0, 'Mild': 1, 'Cool': 2})
df['Humidity'] = df['Humidity'].map({'High': 0, 'Normal': 1})
df['Wind'] = df['Wind'].map({'Weak': 0, 'Strong': 1})
df['Play Sports?'] = df['Play Sports?'].map({'No': 0, 'Yes': 1})

# Step 3: Split the dataset into features and target variable
X = df.drop('Play Sports?', axis=1)  # Features
y = df['Play Sports?']  # Target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Create and train the KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print('Accuracy of the KNN model:', accuracy)
print('Confusion Matrix:\n', conf_matrix)

# Example usage: Predicting for a new instance
new_instance = [[0, 1, 0, 0]]  # Example: Outlook=Sunny, Temperature=Mild, Humidity=High, Wind=Weak
predicted = model.predict(new_instance)
predicted_label = 'Yes' if predicted[0] == 1 else 'No'
print('Predicted class for the new instance:', predicted_label)

