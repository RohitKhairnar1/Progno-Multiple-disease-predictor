import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

# Load the dataset
data = pd.read_csv('Database/Heart.csv') # Ensure this path is correct

# Split the dataset into features and labels
X = data.drop('target', axis=1) # Assuming 'target' is the column name for the outcome
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Save the trained model as a pickle file
with open('heart_lda_model.pkl', 'wb') as file:
    pickle.dump(lda, file)
