import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset from a CSV file
df = pd.read_csv('Database\CKD_Preprocessed.csv')  # Replace 'your_dataset.csv' with the path to your CSV file

# Assuming the last column contains the target variable ('Targets')
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]  # Only the last column

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the model to disk
filename = 'kidney_disease_prediction_model.pkl'
pickle.dump(model, open(filename, 'wb'))

print("Model has been successfully saved to kidney_disease_prediction_model.pkl")
