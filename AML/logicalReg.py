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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Binary Classification - Email Spam Detection
print("=== Logistic Regression: Email Spam Detection ===")
# Create synthetic email features (word counts, sender reputation, etc.)
np.random.seed(42)
X_email, y_email = make_classification(
    n_samples=2000, n_features=5, n_informative=3, n_redundant=1,
    n_clusters_per_class=1, random_state=42
)

# Feature names for interpretation
email_features = ['suspicious_words', 'sender_reputation', 'link_count',
                  'caps_ratio', 'exclamation_marks']
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_email, y_email, test_size=0.2, random_state=42
)

# Train Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_e, y_train_e)

# Predictions and probabilities
y_pred_e = log_reg.predict(X_test_e)
y_prob_e = log_reg.predict_proba(X_test_e)[:, 1]  # Probability of spam

# Evaluation
accuracy_e = accuracy_score(y_test_e, y_pred_e)
auc_score = roc_auc_score(y_test_e, y_prob_e)

print(f"Accuracy: {accuracy_e:.3f}")
print(f"AUC-ROC: {auc_score:.3f}")
print("\nFeature Coefficients (impact on spam probability):")
for feature, coef in zip(email_features, log_reg.coef_[0]):
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {feature}: {coef:.3f} ({direction} spam probability)")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test_e, y_pred_e)
print(f"True Neg: {cm[0, 0]}, False Pos: {cm[0, 1]}")
print(f"False Neg: {cm[1, 0]}, True Pos: {cm[1, 1]}")

# Multiclass Classification - Customer Satisfaction
print("\n=== Multiclass Logistic Regression: Customer Satisfaction ===")
# Create customer data: satisfaction levels (Low=0, Medium=1, High=2)
np.random.seed(42)
n_customers = 1500

customer_features = {
    'service_rating': np.random.uniform(1, 5, n_customers),
    'price_satisfaction': np.random.uniform(1, 5, n_customers),
    'response_time': np.random.uniform(1, 10, n_customers),  # days
    'previous_issues': np.random.poisson(2, n_customers)  # count
}

# Create satisfaction based on realistic relationships
satisfaction_score = (
    0.8 * customer_features['service_rating'] +
    0.6 * customer_features['price_satisfaction'] -
    0.2 * customer_features['response_time'] -
    0.3 * customer_features['previous_issues'] +
    np.random.normal(0, 0.5, n_customers)
)

# Convert to categories: Low (0), Medium (1), High (2)
satisfaction_labels = np.where(satisfaction_score < 1.5,
                                0,
                                np.where(satisfaction_score < 3.0, 1, 2))

X_cust = np.column_stack([customer_features['service_rating'],
                           customer_features['price_satisfaction'],
                           customer_features['response_time'],
                           customer_features['previous_issues']])

cust_feature_names = ['Service_Rating', 'Price_Satisfaction', 'Response_Time',
                      'Previous_Issues']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cust, satisfaction_labels, test_size=0.2, random_state=42
)

# Train multiclass logistic regression
log_reg_multi = LogisticRegression(random_state=42, max_iter=1000)
log_reg_multi.fit(X_train_c, y_train_c)

# Predictions
y_pred_c = log_reg_multi.predict(X_test_c)
y_prob_c = log_reg_multi.predict_proba(X_test_c)

accuracy_c = accuracy_score(y_test_c, y_pred_c)
print(f"Multiclass Accuracy: {accuracy_c:.3f}")

print("\nClassification Report:")
class_names = ['Low', 'Medium', 'High']
print(classification_report(y_test_c, y_pred_c, target_names=class_names))

# Show probability example
print(f"\nExample Customer Prediction:")
sample_idx = 0
print(f"Features: Service={X_test_c[sample_idx, 0]:.1f}, Price={X_test_c[sample_idx, 1]:.1f}, "
      f"Response={X_test_c[sample_idx, 2]:.1f}, Issues={X_test_c[sample_idx, 3]:.0f}")
print(f"Predicted: {class_names[y_pred_c[sample_idx]]}")
print(f"Probabilities: Low={y_prob_c[sample_idx, 0]:.3f}, "
      f"Medium={y_prob_c[sample_idx, 1]:.3f}, High={y_prob_c[sample_idx, 2]:.3f}")

