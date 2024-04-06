import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv('Database/diabetes.csv')

# Split the dataset into features and labels
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('diabetes_model.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Optionally, evaluate the model on the test set
# accuracy = clf.score(X_test, y_test)
# print(f"Model accuracy: {accuracy}")
