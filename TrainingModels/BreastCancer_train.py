import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
df = pd.read_csv('Database\BreastCancer.csv')

# Preprocess the dataset
# Assuming 'diagnosis' is the target variable and the rest are features
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
with open('breast_cancer_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('Breast_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
