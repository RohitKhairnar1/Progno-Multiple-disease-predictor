from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import os


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('PAGE1.html')

@app.route('/PAGE1',endpoint='page1')
def page1():
    return render_template('PAGE1.html')

@app.route('/PAGE2',endpoint='page2')
def page2():
    return render_template('PAGE2.html')

@app.route('/DiabetesPAGE',endpoint='diabetespage')
def diabetes_page():
    return render_template('DiabetesPAGE.html')

@app.route('/KidneyPAGE',endpoint='kidneypage')
def kidney_page():
    return render_template('KidneyPAGE.html')

@app.route('/HeartPAGE',endpoint='heartpage')
def heart_page():
    return render_template('HeartPAGE.html')

@app.route('/MalariaPAGE',endpoint='malaria')
def malaria_page():
    return render_template('MalariaPAGE.html')

@app.route('/PneumoniaPAGE',endpoint='pneumonia')
def pneumonia_page():
    return render_template('PneumoniaPAGE.html')

@app.route('/BreastPAGE',endpoint='breastpage')
def breast_page():
    return render_template('BreastPAGE.html')

@app.route('/LiverPAGE',endpoint='liver')
def liver_page():
    return render_template('LiverPAGE.html')

@app.route('/About',endpoint='about')
def about():
    return render_template('About.html')

@app.route('/Account',endpoint='account')
def account():
    return render_template('Account.html')

@app.route('/Signup',endpoint='signup')
def account():
    return render_template('Signup.html')
    

@app.route('/Login',endpoint='login')
def page1():
    return render_template('Login.html')

# Load the trained model for diabetes
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/predict_diabetes', methods=['POST'], endpoint='predict_diabetes')
def predict_diabetes():
    data = request.form
    # Ensure the field names match those in your HTML form
    features = [float(data[feature]) for feature in ['pregnancies', 'glucose', 'bloodPressure', 'skinThickness', 'insulin', 'bmi', 'pedigreeFunction', 'age']]
    prediction = diabetes_model.predict_proba([features])[0][1] * 100
    severity_percentage = round(prediction, 2)
    if severity_percentage > 70:
        recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
    elif severity_percentage > 40:
        recommendation = "Consider making some lifestyle changes and schedule regular health check-ups."
    else:
        recommendation = "Keep an eye on your health, but no immediate action is required."
    return render_template('Result.html', prediction_text='Predicted Severity: {}%'.format(severity_percentage), recommendation=recommendation)

# Load the trained model for Heart
lda_model = pickle.load(open('heart_lda_model.pkl', 'rb'))

@app.route('/predict_heart', methods=['POST'] ,endpoint='predict_heart')
def predict_heart():
    data = request.form
    # Ensure the field names match those in your HTML form
    features = [float(data[feature]) for feature in ['age', 'sex', 'chestPainType', 'bp', 'cholesterol', 'fbsOver120', 'ekgResults', 'maxHR', 'exerciseAngina', 'stDepression', 'slopeOfST', 'numOfVesselsFluro', 'thallium']]
    # Use predict_proba to get the probability of the positive class (heart disease)
    prediction = lda_model.predict_proba([features])[0][1] * 100
    severity_percentage = round(prediction, 2)
    if severity_percentage > 70:
        recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
    elif severity_percentage > 40:
        recommendation = "Consider making some lifestyle changes and schedule regular health check-ups."
    else:
        recommendation = "Keep an eye on your health, but no immediate action is required."
    return render_template('Result.html', prediction_text='Predicted Severity: {}%'.format(severity_percentage), recommendation=recommendation)


## liver :

# Load the trained model and scaler
with open('liver_model.pkl', 'rb') as file:
    liver_model = pickle.load(file)
with open('Liver_scaler.pkl', 'rb') as file:
    Liver_scaler = pickle.load(file)

@app.route('/predict_liver', methods=['POST'], endpoint='predict_liver')
def predict_liver():
    data = request.form
    gender = 0 if data['Gender'] == 'Female' else 1
    # Ensure this list matches the features used during training
    features = [float(data[feature]) for feature in ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
    features.insert(1, gender)
    imputer = SimpleImputer(strategy='median')
    features_imputed = imputer.fit_transform([features])
    features_scaled = Liver_scaler.transform(features_imputed)
    prediction = liver_model.predict_proba(features_scaled)[0][1] * 100
    severity_percentage = round(prediction, 2)
    if severity_percentage > 70:
        recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
    elif severity_percentage > 40:
        recommendation = "Consider making some lifestyle changes and schedule regular health check-ups."
    else:
        recommendation = "Keep an eye on your health, but no immediate action is required."
    return render_template('Result.html', prediction_text='Predicted Severity: {}%'.format(severity_percentage), recommendation=recommendation)

# Load the trained model and scaler
with open('breast_cancer_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('Breast_scaler.pkl', 'rb') as file:
    Breast_scaler = pickle.load(file)

@app.route('/predict_breast_cancer', methods=['POST'], endpoint='predict_breast_cancer')
def predict_breast_cancer():
    data = request.form
    # Assuming all features are numerical and directly taken from the form
    features = [float(data[feature]) for feature in ['radiusMean', 'radiusSe', 'radiusWorst', 'textureMean', 'textureSe', 'textureWorst', 'perimeterMean', 'perimeterSe', 'perimeterWorst', 'areaMean', 'areaSe', 'areaWorst', 'smoothnessMean', 'smoothnessSe', 'smoothnessWorst', 'compactnessMean', 'compactnessSe', 'compactnessWorst', 'concavityMean', 'concavitySe', 'concavityWorst', 'concavePointsMean', 'concavePointsSe', 'concavePointsWorst', 'symmetryMean', 'symmetrySe', 'symmetryWorst', 'fractalDimensionMean', 'fractalDimensionSe', 'fractalDimensionWorst']]
    
    # Standardize the features
    features_scaled = Breast_scaler.transform([features])
    
    # Make a prediction
    prediction = model.predict(features_scaled)[0]
    
    # Convert the prediction to a string
    prediction_text = 'Malignant' if prediction == 1 else 'Benign'
    
    # Determine the recommendation based on the prediction
    if prediction == 1:
        recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
    else:
        recommendation = "Keep an eye on your health, but no immediate action is required."
    
    return render_template('Result.html', prediction_text=prediction_text, recommendation=recommendation)

# Load the trained model and scaler
with open('kidney_disease_model.pkl', 'rb') as file:
    kidney_model = pickle.load(file)
with open('kidney_disease_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

from sklearn.preprocessing import LabelEncoder

@app.route('/predict_kidney', methods=['POST'], endpoint='predict_kidney')
def predict_kidney():
    # Create a mutable copy of the form data
    data = request.form.copy()
    
    # Assuming 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane' are categorical features
    categorical_features = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    le = LabelEncoder()
    
    # Encode categorical features
    for feature in categorical_features:
        # Check if the key exists in the data dictionary
        if feature in data:
            data[feature] = le.fit_transform([data[feature]])[0]
        else:
            # Handle missing keys, e.g., by skipping encoding or providing a default value
            print(f"Warning: The key '{feature}' was not found in the form data.")
            # Optionally, you can skip encoding for this feature or provide a default value
    
    # Convert all features to float
    features = [float(data[feature]) for feature in ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]
    
    # Standardize the features
    features_scaled = scaler.transform([features])
    
    # Make a prediction and get the probability of the positive class
    probabilities = kidney_model.predict_proba(features_scaled)[0]
    prediction = kidney_model.predict(features_scaled)[0]
    severity_percentage = round(probabilities[1] * 100, 2) # Assuming the second element is the probability of CKD
    
    # Convert the prediction to a string
    prediction_text = 'CKD' if prediction == 1 else 'No CKD'
    
    # Determine the recommendation based on the severity
    if severity_percentage > 70:
        recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
    elif severity_percentage > 40:
        recommendation = "Consider making some lifestyle changes and schedule regular health check-ups."
    else:
        recommendation = "Keep an eye on your health, but no immediate action is required."
    
    return render_template('Result.html', prediction_text='Predicted Severity: {}%'.format(severity_percentage), recommendation=recommendation)


# Load the malaria detection model
malaria_model = load_model('malaria.h5')

@app.route('/predict_malaria', methods=['POST'], endpoint='predict_malaria')
def predict_malaria():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the file to a temporary location
        filename = 'temp_image.png'
        file.save(filename)
        
        # Preprocess the image
        img = image.load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Predict
        prediction = malaria_model.predict(x)[0][0]
        prediction_text = 'Parasitized' if prediction > 0.3 else 'Uninfected'
        
        # Clean up the temporary file
        os.remove(filename)
        
        return render_template('Result.html', prediction_text=prediction_text)
    
if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/results')
# def results():
#     prediction_text = request.args.get('prediction_text', '')
#     recommendation = request.args.get('recommendation', '')
#     return render_template('Result.html', prediction_text=prediction_text, recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)


