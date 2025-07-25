import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Feature matrix: [Age, Time Spent, Added to Cart]
X = np.array([[25, 30, 0], [30, 40, 1], [20, 35, 0], [35, 45, 1]])

# Target vector: 1=purchased, 0=not purchased
y = np.array([0, 1, 0, 1])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Take user input and predict
user_age = float(input("Enter Customer Age: "))
user_time_spent = float(input("Enter Time spent on Website: "))
user_added_to_cart = int(input("Enter 1 if added to cart else Enter 0: "))
user_data = np.array([[user_age, user_time_spent, user_added_to_cart]])
prediction = model.predict(user_data)
if prediction[0] == 1:
    print("The Customer is likely to purchase")
else:
    print("The Customer is unlikely to purchase")
