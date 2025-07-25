# Step 1 
import numpy as np                      # Importing numpy for numerical operations
import pandas as pd                     # Importing pandas for data handling
from sklearn.model_selection import train_test_split   # For splitting data into training and test sets
from sklearn.linear_model import LinearRegression       # For performing linear regression

# Step 2
sharaddata = {'Hours_study': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'Exam_score': [50, 60, 65, 70, 75, 80, 85, 90, 95]}
# Dictionary with hours studied and corresponding exam scores

# Step 3
df = pd.DataFrame(sharaddata)   # Converting dictionary into a pandas DataFrame

# Step 4
X = df[['Hours_study']]         # Feature (independent variable) as a DataFrame
y = df[['Exam_score']]          # Target (dependent variable) as a DataFrame

# Step 5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Splitting data into training and testing sets (80% train, 20% test)

# Step 6
model = LinearRegression()      # Creating a Linear Regression model

# Step 7
model.fit(X_train, y_train)     # Training the model on the training data

# Step 8
user_input = float(input("Enter the number of Hours you Study :"))     # Taking user input for study hours
predicted_score = model.predict([[user_input]])                         # Predicting exam score based on input

# Step 9
print(f"Predicted Exam Score: {predicted_score[0][0]:.2f}")   # Printing the predicted score