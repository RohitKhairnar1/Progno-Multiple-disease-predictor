import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Load your dataset
df = pd.read_csv('Database\Liver.csv')

# Convert categorical variables (e.g., Gender) to numerical
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# Split the dataset into features and target
X = df.drop('Dataset', axis=1)
y = df['Dataset']

# Handle missing values by filling them with the median of each column
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Extra Trees model
model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
with open('liver_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('Liver_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
