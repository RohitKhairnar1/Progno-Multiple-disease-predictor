import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
df = pd.read_csv('Database\Kidney.csv')

# Preprocess the dataset
# Assuming 'classification' is the target variable and the rest are features
X = df.drop('classification', axis=1)
y = df['classification']

# Handle missing values for numerical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

# Handle missing values for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

# Encode categorical variables
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Convert to DataFrame
X = pd.DataFrame(X, columns=df.columns[:-1])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model and scaler
with open('kidney_disease_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('kidney_disease_scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)