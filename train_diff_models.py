from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Classes.DataReader import DataReader

dataset = DataReader()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.X_not_normalized, dataset.Y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model on all features
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("All Features:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

feature_to_remove = "HEART RATE"
# Drop the selected features
X_train_reduced = X_train.drop(columns=[feature_to_remove])
X_test_reduced = X_test.drop(columns=[feature_to_remove])

# Train the model on reduced features
model_reduced = RandomForestClassifier(random_state=42)
model_reduced.fit(X_train_reduced, y_train)

# Make predictions
y_pred_reduced = model_reduced.predict(X_test_reduced)

# Evaluate the reduced model
print("\nWithout " + feature_to_remove + ":")
print("Accuracy:", accuracy_score(y_test, y_pred_reduced))
print("Precision:", precision_score(y_test, y_pred_reduced))
print("Recall:", recall_score(y_test, y_pred_reduced))
