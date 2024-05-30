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

@app.route('/MalariaPAGE',endpoint='malariaPAGE')
def malaria_page():
    return render_template('MalariaPAGE.html')

@app.route('/malaria',endpoint='malaria')
def malaria_page():
    return render_template('malaria.html')

@app.route('/pneumonia',endpoint='pneumonia')
def malaria_page():
    return render_template('pneumonia.html')

@app.route('/malaria_predict',endpoint='malaria_predict')
def malaria_page():
    return render_template('malaria_predict.html')

@app.route('/PneumoniaPAGE',endpoint='pneumoniaPAGE')
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
        recommendation = (
            "Due to the high predicted severity of diabetes, immediate action is necessary. "
            "You should schedule an appointment with a healthcare provider as soon as possible. Here are some steps you should consider:\n\n"
            "Book an appointment with an endocrinologist.\n"
            "You might need medication to manage your blood sugar levels.\n"
            "Follow a strict diet plan recommended by your healthcare provider, focusing on low glycemic index foods.\n"
            "Incorporate regular physical activity, such as walking, swimming, or cycling.\n"
            "Regularly monitor your blood glucose levels and keep a record.\n"
            "Attend diabetes education classes to better understand and manage your condition."
        )
        detailed_explanation = (
            "A severity level above 70% indicates a high risk of developing complications related to diabetes. "
            "These complications can include cardiovascular diseases, neuropathy, retinopathy, and kidney damage. "
            "Early and aggressive management can help prevent these complications and improve your quality of life."
        )

    elif severity_percentage > 40:
        recommendation = (
            "Your risk of diabetes is moderate.It is important to take steps to manage your health and prevent the progression of the disease. Consider the following actions:\n\n"
            "Schedule a check-up with your primary care physician.\n"
            "Adopt a balanced diet rich in fiber, vegetables, and lean proteins.\n"
            "Engage in at least 150 minutes of moderate-intensity exercise each week.\n"
            "Keep track of your blood sugar levels, especially if you experience symptoms like increased thirst or frequent urination.\n"
            "Avoid smoking and limit alcohol intake."
        )
        detailed_explanation = (
            "A severity level between 40% and 70% suggests that you have a moderate risk of developing diabetes. "
            "This means that with appropriate lifestyle changes and regular medical check-ups, you can significantly reduce your risk and potentially avoid the onset of diabetes. "
            "It’s crucial to maintain a healthy lifestyle to prevent the disease from progressing."
        )

    else:
        recommendation = (
            "Your predicted severity is low, which means that while you have some risk factors, immediate action may not be necessary. "
            "However, it's still important to maintain a healthy lifestyle to keep your risk low. Here are some recommendations:\n\n"
            "Eat a balanced diet and maintain a healthy weight.\n"
            "Engage in regular physical activity.\n"
            "Be aware of the symptoms of diabetes and monitor your health periodically.\n"
            "Continue with regular health check-ups to monitor any changes in your risk."
        )
        detailed_explanation = (
            "A severity level below 40% indicates a lower risk of developing diabetes. This is a good sign, but it doesn't mean that you should ignore your health. "
            "Continuing to follow a healthy lifestyle will help you maintain this low risk and prevent the development of diabetes in the future."
        )

    return render_template('ResultDiabetes.html', 
                           prediction_text='Predicted Severity: {}%'.format(severity_percentage),
                           recommendation=recommendation,
                           detailed_explanation=detailed_explanation,
                           Disease="Diabetes",
                           features=data)

###### heart !**!*!*

# Load the trained model for heart disease
heart_model = pickle.load(open('heart_lda_model.pkl', 'rb'))

@app.route('/predict_heart', methods=['POST'], endpoint='predict_heart')
def predict_heart():
    data = request.form
    # Ensure the field names match those in your HTML form
    features = [float(data[feature]) for feature in [
        'age', 'sex', 'chestPainType', 'bp', 'cholesterol', 
        'fbsOver120', 'ekgResults', 'maxHR', 'exerciseAngina', 
        'stDepression', 'slopeOfST', 'numOfVesselsFluro', 'thallium'
    ]]
    prediction = heart_model.predict_proba([features])[0][1] * 100
    severity_percentage = round(prediction, 2)
    
    if severity_percentage > 70:
        recommendation = (
            "Due to the high predicted severity of heart disease, immediate action is necessary. "
            "You should schedule an appointment with a healthcare provider as soon as possible. Here are some steps you should consider:\n\n"
            "1. Book an appointment with a cardiologist.\n"
            "2. You might need medication to manage your heart condition.\n"
            "3. Follow a strict diet plan recommended by your healthcare provider, focusing on low-fat, low-sodium foods.\n"
            "4. Incorporate regular physical activity, such as walking, swimming, or cycling, as recommended by your doctor.\n"
            "5. Regularly monitor your blood pressure and cholesterol levels and keep a record.\n"
            "6. Attend heart health education classes to better understand and manage your condition."
        )
        detailed_explanation = (
            "A severity level above 70% indicates a high risk of developing complications related to heart disease. "
            "These complications can include heart attack, stroke, and other cardiovascular events. "
            "Early and aggressive management can help prevent these complications and improve your quality of life."
        )

    elif severity_percentage > 40:
        recommendation = (
            "Your risk of heart disease is moderate. It is important to take steps to manage your health and prevent the progression of the disease. Consider the following actions:\n\n"
            "1. Schedule a check-up with your primary care physician.\n"
            "2. Adopt a balanced diet rich in fruits, vegetables, and lean proteins.\n"
            "3. Engage in at least 150 minutes of moderate-intensity exercise each week.\n"
            "4. Keep track of your blood pressure and cholesterol levels, especially if you experience symptoms like chest pain or shortness of breath.\n"
            "5. Avoid smoking and limit alcohol intake."
        )
        detailed_explanation = (
            "A severity level between 40% and 70% suggests that you have a moderate risk of developing heart disease. "
            "This means that with appropriate lifestyle changes and regular medical check-ups, you can significantly reduce your risk and potentially avoid the onset of heart disease. "
            "It’s crucial to maintain a healthy lifestyle to prevent the disease from progressing."
        )

    else:
        recommendation = (
            "Your predicted severity is low, which means that while you have some risk factors, immediate action may not be necessary. "
            "However, it's still important to maintain a healthy lifestyle to keep your risk low. Here are some recommendations:\n\n"
            "1. Eat a balanced diet and maintain a healthy weight.\n"
            "2. Engage in regular physical activity.\n"
            "3. Be aware of the symptoms of heart disease and monitor your health periodically.\n"
            "4. Continue with regular health check-ups to monitor any changes in your risk."
        )
        detailed_explanation = (
            "A severity level below 40% indicates a lower risk of developing heart disease. This is a good sign, but it doesn't mean that you should ignore your health. "
            "Continuing to follow a healthy lifestyle will help you maintain this low risk and prevent the development of heart disease in the future."
        )

    return render_template('ResultHeartDisease.html', 
                           prediction_text='Predicted Severity: {}%'.format(severity_percentage),
                           recommendation=recommendation,
                           detailed_explanation=detailed_explanation,
                           Disease="Heart Disease",
                           features=data)

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
        recommendation = (
            "Due to the high predicted severity of liver disease, immediate action is necessary. "
            "You should schedule an appointment with a healthcare provider as soon as possible. Here are some steps you should consider:\n\n"
            "Consult a gastroenterologist or hepatologist for further evaluation and management.\n"
            "Follow a liver-friendly diet low in saturated fats, refined sugars, and alcohol.\n"
            "Avoid alcohol consumption and certain medications that can further harm the liver.\n"
            "Consider additional diagnostic tests or imaging studies as recommended by your healthcare provider.\n"
            "Regularly monitor liver function tests and follow-up with your healthcare team."
        )
        detailed_explanation = (
            "A severity level above 70% indicates a high risk of liver damage or advanced liver disease. "
            "Immediate medical attention is crucial to prevent further complications such as liver failure, cirrhosis, or hepatocellular carcinoma. "
            "Prompt intervention and lifestyle modifications can help improve liver health and overall prognosis."
        )

    elif severity_percentage > 40:
        recommendation = (
            "Your risk of liver disease is moderate. It is important to take steps to manage your health and prevent the progression of the disease. Consider the following actions:\n\n"
            "Schedule a comprehensive liver evaluation with a healthcare provider.\n"
            "Adopt a liver-friendly diet rich in fruits, vegetables, and whole grains.\n"
            "Limit alcohol consumption and avoid risky behaviors that may harm the liver.\n"
            "Engage in regular physical activity to maintain a healthy weight and reduce liver fat.\n"
            "Follow up with routine liver function tests and imaging studies as recommended."
        )
        detailed_explanation = (
            "A severity level between 40% and 70% suggests a moderate risk of liver disease or liver dysfunction. "
            "This may indicate underlying liver conditions such as fatty liver disease, viral hepatitis, or autoimmune liver disorders. "
            "Early detection and intervention can help prevent the progression to advanced liver disease and improve outcomes."
        )

    else:
        recommendation = (
            "Your predicted severity is low, indicating a lower risk of liver disease. However, it's still important to maintain a healthy lifestyle and monitor your liver health regularly. Here are some recommendations:\n\n"
            "Maintain a balanced diet with plenty of fruits, vegetables, and whole grains.\n"
            "Limit alcohol intake to promote liver health.\n"
            "Stay physically active and maintain a healthy weight.\n"
            "Avoid smoking and exposure to toxins that can harm the liver.\n"
            "Follow up with routine healthcare visits for preventive care and screenings."
        )
        detailed_explanation = (
            "A severity level below 40% suggests a lower risk of liver disease or liver dysfunction. "
            "However, it's important to remain vigilant about liver health and address any risk factors or symptoms promptly. "
            "Lifestyle modifications and regular healthcare monitoring can help maintain liver health and prevent the development of liver disease."
        )

    return render_template('ResultLiver.html', 
                           prediction_text='Predicted Severity: {}%'.format(severity_percentage),
                           recommendation=recommendation,
                           detailed_explanation=detailed_explanation,
                           Disease="Liver",
                           features=data)


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
        recommendation = (
            "Due to the high probability of malignancy, it's important to consult with a healthcare professional for further evaluation and management. Here are some recommendations:\n\n"
            "- Schedule a biopsy or further diagnostic tests to confirm the diagnosis.\n"
            "- Discuss treatment options with an oncologist, which may include surgery, chemotherapy, or radiation therapy.\n"
            "- Follow a healthy lifestyle, including a balanced diet and regular exercise, to support overall health and well-being.\n"
            "- Seek emotional support from friends, family, or support groups to cope with the diagnosis and treatment process."
        )
        detailed_explanation = (
            "A prediction of malignancy indicates a higher likelihood of cancerous cells in the breast tissue. "
            "Early detection and prompt intervention are crucial for better treatment outcomes and prognosis. "
            "Consulting with healthcare professionals and following recommended treatment plans can improve chances of successful treatment and recovery."
        )

    else:
        recommendation = (
            "Based on the prediction, the likelihood of malignancy is low. However, it's still important to monitor your health and discuss any concerns with a healthcare professional. Here are some recommendations:\n\n"
            "- Schedule regular breast cancer screenings and follow-ups with your healthcare provider.\n"
            "- Practice breast self-exams and report any changes or abnormalities to your doctor.\n"
            "- Maintain a healthy lifestyle with a balanced diet, regular exercise, and avoidance of harmful habits like smoking.\n"
            "- Stay informed about breast health and cancer prevention strategies to reduce overall risk."
        )
        detailed_explanation = (
            "A prediction of benignity suggests a lower likelihood of cancerous cells in the breast tissue. "
            "However, it's essential to remain vigilant about breast health and continue routine screenings and check-ups. "
            "Early detection and preventive measures play a crucial role in reducing the risk of developing breast cancer."
        )

    return render_template('ResultBreast.html', 
                           prediction_text=prediction_text,
                           recommendation=recommendation,
                           detailed_explanation=detailed_explanation,
                           Disease="Breast",
                           features=data)


# # Load the trained model and scaler
# with open('kidney_disease_model.pkl', 'rb') as file:
#     kidney_model = pickle.load(file)
# with open('kidney_disease_scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)

# from sklearn.preprocessing import LabelEncoder

# @app.route('/predict_kidney', methods=['POST'], endpoint='predict_kidney')
# def predict_kidney():
#     # Create a mutable copy of the form data
#     data = request.form.copy()
    
#     # Assuming 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane' are categorical features
#     categorical_features = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
#     le = LabelEncoder()
    
#     # Encode categorical features
#     for feature in categorical_features:
#         # Check if the key exists in the data dictionary
#         if feature in data:
#             data[feature] = le.fit_transform([data[feature]])[0]
#         else:
#             # Handle missing keys, e.g., by skipping encoding or providing a default value
#             print(f"Warning: The key '{feature}' was not found in the form data.")
#             # Optionally, you can skip encoding for this feature or provide a default value
    
#     # Convert all features to float
#     features = [float(data[feature]) for feature in ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']]
    
#     # Standardize the features
#     features_scaled = scaler.transform([features])
    
#     # Make a prediction and get the probability of the positive class
#     probabilities = kidney_model.predict_proba(features_scaled)[0]
#     prediction = kidney_model.predict(features_scaled)[0]
#     severity_percentage = round(probabilities[1] * 100, 2) # Assuming the second element is the probability of CKD
    
#     # Convert the prediction to a string
#     prediction_text = 'CKD' if prediction == 1 else 'No CKD'
    
#     # Determine the recommendation based on the severity
#     if severity_percentage > 70:
#         recommendation = "It's highly recommended to consult with a healthcare professional for regular check-ups and lifestyle modifications."
#     elif severity_percentage > 40:
#         recommendation = "Consider making some lifestyle changes and schedule regular health check-ups."
#     else:
#         recommendation = "Keep an eye on your health, but no immediate action is required."
    
#     return render_template('Result.html', prediction_text='Predicted Severity: {}%'.format(severity_percentage), recommendation=recommendation)


# # Load the malaria detection model
# malaria_model = load_model('Malaria-Detection-Model.pkl')

# @app.route('/predict_malaria', methods=['POST'], endpoint='predict_malaria')
# def predict_malaria():
#     if 'image' not in request.files:
#         return redirect(request.url)
#     file = request.files['image']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         # Save the file to a temporary location
#         filename = 'temp_image.png'
#         file.save(filename)
        
#         # Preprocess the image
#         img = image.load_img(filename, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
        
#         # Predict
#         prediction = malaria_model.predict(x)[0][0]
#         prediction_text = 'Parasitized' if prediction > 0.3 else 'Uninfected'
        
#         # Clean up the temporary file
#         os.remove(filename)
        
#         return render_template('Result.html', prediction_text=prediction_text)

from flask import Flask, request, render_template
import pickle

# Load the trained model and scaler
with open('kidney_disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict_kidney', methods=['POST'], endpoint='predict_kidney')
def predict_kidney_disease():
    data = request.form
    # Assuming all features are numerical and directly taken from the form
    features = [float(data[feature]) for feature in ['age', 'bp', 'sg', 'al', 'su', 'rbc','pc','pcc','ba','bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad','appet','pe','ane']]
    
    # Standardize the features
    # Assuming Breast_scaler is replaced with a scaler appropriate for kidney disease data
    # features_scaled = Kidney_scaler.transform([features])
    
    # Make a prediction
    prediction = model.predict([features])[0]
    
    # Convert the prediction to a string
    prediction_text = 'At Risk' if prediction == 1 else 'Not At Risk'
    
    # Determine the recommendation based on the prediction
    if prediction == 1:
        recommendation = (
            "The prediction indicates a higher risk of kidney disease. It's crucial to consult with a healthcare professional for a comprehensive evaluation and personalized advice. Recommendations may include:\n\n"
            "- Regular monitoring of blood pressure and glucose levels.\n"
            "- Consider dietary adjustments to manage blood sugar and cholesterol levels.\n"
            "- Stay hydrated and maintain a healthy weight.\n"
            "- Avoid medications that could harm kidney function without consulting a doctor."
        )
        detailed_explanation = (
            "A prediction of being at risk for kidney disease highlights the importance of proactive care and early intervention. "
            "Monitoring kidney function regularly and adhering to prescribed treatments can significantly impact the course of the disease. "
            "Professional guidance is essential for managing risk factors effectively."
        )

    else:
        recommendation = (
            "Based on the prediction, there is a lower risk of kidney disease. However, maintaining good health practices is always beneficial. Recommendations include:\n\n"
            "- Continue regular health check-ups to monitor vital signs.\n"
            "- Engage in physical activity as advised by healthcare professionals.\n"
            "- Maintain a balanced diet rich in fruits and vegetables.\n"
            "- Avoid excessive alcohol and limit salt intake."
        )
        detailed_explanation = (
            "A prediction of not being at risk for kidney disease suggests that current health practices are effective. "
            "It's still important to stay vigilant about general health and wellness, as changes over time can affect risk profiles. "
            "Regular check-ups help in early detection of potential issues."
        )

    return render_template('ResultKidney.html', 
                           prediction_text=prediction_text,
                           recommendation=recommendation,
                           detailed_explanation=detailed_explanation,
                           Disease="Kidney Disease",
                           features=data)

from PIL import Image

@app.route("/malariapredict", methods=['POST', 'GET'], endpoint='malariapredict')
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 3))
                img = img.astype(np.float64)

                model = load_model("malaria.h5")
                pred = np.argmax(model.predict(img)[0])

                prediction_text = 'Infected Malarial Cell' if pred == 1 else 'Uninfected Cell'

                if pred == 1:
                    recommendation = (
                        "The prediction indicates that the cell is infected with malaria. It's important to consult with a healthcare professional for further evaluation and treatment. Recommendations may include:\n\n"
                        "- Antimalarial medication as prescribed by a doctor.\n"
                        "- Regular monitoring and follow-up with healthcare providers.\n"
                        "- Rest and hydration to support recovery.\n"
                        "- Preventive measures to avoid mosquito bites in the future."
                    )
                    detailed_explanation = (
                        "An infected malarial cell indicates the presence of malaria parasites in the blood. "
                        "Timely and appropriate treatment is crucial for effective management of the disease. "
                        "Healthcare professionals can provide guidance on the best course of action based on the severity of the infection."
                    )
                else:
                    recommendation = (
                        "The prediction indicates that the cell is not infected with malaria. To maintain good health and prevent malaria, consider the following recommendations:\n\n"
                        "- Use insect repellent and sleep under mosquito nets.\n"
                        "- Take antimalarial prophylaxis if traveling to malaria-endemic areas.\n"
                        "- Monitor for symptoms and seek medical advice if you feel unwell."
                    )
                    detailed_explanation = (
                        "An uninfected cell suggests no presence of malaria parasites in the blood sample. "
                        "Continued vigilance and preventive measures are essential to avoid future infections. "
                        "Maintaining overall health and seeking prompt medical care if symptoms arise can help prevent malaria."
                    )

                return render_template('malaria_predict.html', 
                                       prediction_text=prediction_text,
                                       recommendation=recommendation,
                                       detailed_explanation=detailed_explanation,
                                       Disease="Malaria",
                                       features=request.files['image'].filename)
        except Exception as e:
            message = "Please upload an Image. Error: {}".format(str(e))
            return render_template('malaria.html', message=message)
    
    return render_template('malaria.html')

@app.route("/pneumoniapredict", methods=['POST', 'GET'], endpoint='pneumoniapredict')
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36, 36))
                img = np.asarray(img)
                img = img.reshape((1, 36, 36, 1))
                img = img / 255.0

                model = load_model("pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])

                prediction_text = 'Pneumonia Detected' if pred == 1 else 'No Pneumonia Detected'

                if pred == 1:
                    recommendation = (
                        "The prediction indicates that the image shows signs of pneumonia. It is important to consult a healthcare professional for a thorough diagnosis and treatment plan. Recommendations may include:\n\n"
                        "- Follow prescribed medication and treatment plans.\n"
                        "- Get plenty of rest and stay hydrated.\n"
                        "- Monitor symptoms and follow up with your healthcare provider.\n"
                        "- Avoid smoking and exposure to secondhand smoke."
                    )
                    detailed_explanation = (
                        "Detection of pneumonia suggests the presence of an infection in the lungs. "
                        "Early and appropriate treatment is crucial to prevent complications. "
                        "Healthcare professionals can provide personalized advice and treatment based on the severity of the condition."
                    )
                else:
                    recommendation = (
                        "The prediction indicates no signs of pneumonia in the image. To maintain good respiratory health, consider the following recommendations:\n\n"
                        "- Avoid smoking and exposure to pollutants.\n"
                        "- Get vaccinated against influenza and pneumococcal pneumonia.\n"
                        "- Practice good hygiene to prevent infections.\n"
                        "- Regularly monitor your health and seek medical advice if you experience any respiratory symptoms."
                    )
                    detailed_explanation = (
                        "A result showing no signs of pneumonia suggests healthy lung condition in the scanned image. "
                        "However, maintaining good respiratory hygiene and regular check-ups are important to prevent future respiratory issues. "
                        "Consulting healthcare providers for any unusual symptoms can help in early detection and treatment of potential problems."
                    )

                return render_template('pneumoniapredict.html', 
                                       prediction_text=prediction_text,
                                       recommendation=recommendation,
                                       detailed_explanation=detailed_explanation,
                                       Disease="Pneumonia",
                                       features=request.files['image'].filename)
        except Exception as e:
            message = "Please upload an Image. Error: {}".format(str(e))
            return render_template('pneumonia.html', message=message)
    
    return render_template('pneumonia.html')

if __name__ == "__main__":
    app.run(debug=True)


# result *!*!*!*!

if __name__ == '__main__':
    app.run(debug=True)

