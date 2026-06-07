"""
DecisionTree_Practice.ipynb - Standalone runner
Salary prediction using Decision Tree Classifier
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("salaries.csv")
print("=== Dataset ===")
print(df)
print()

# Prepare features and target
X = df.drop(["salary_more_then_100k"], axis=1)
y = df.salary_more_then_100k

# Encode categorical features
encoder = LabelEncoder()
X['company_n'] = encoder.fit_transform(X.company)
X['job_n'] = encoder.fit_transform(X.job)
X['degree_n'] = encoder.fit_transform(X.degree)
X.drop(['company', 'job', 'degree'], axis=1, inplace=True)

print("=== Encoded Features (X) ===")
print(X)
print()
print("=== Target (y) ===")
print(y)
print()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("=== Predictions on Test Set ===")
print(f"Predicted: {y_pred}")
print(f"Actual:    {y_test.values}")
print()

# Score
score = model.score(X_test, y_test)
print(f"=== Model Accuracy: {score:.2%} ===")
