import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Load the diabetes dataset (replace 'your_diabetes_dataset.csv' with your actual dataset)
df = pd.read_csv('Progno-Multiple-disease-predictor\Dataset\diabetes.csv')

# Assume 'X' contains your features and 'y' contains your target variable
X = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]
y = df['Outcome']  # Assuming 'Outcome' is the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Create a pipeline with SMOTE and Random Forest
pipeline_smote = Pipeline([
    ('smote', SMOTE(sampling_strategy='auto')),
    ('rf', rf_classifier)
])

# Train the model with SMOTE
pipeline_smote.fit(X_train, y_train)

# Save the trained model to a single PKL file
joblib.dump(pipeline_smote, 'Progno-Multiple-disease-predictor\Models\diabetes_model.pkl')

print('Model trained and saved successfully.')
