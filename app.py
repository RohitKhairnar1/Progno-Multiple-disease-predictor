from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

@app.route('/predict_diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    try:
        if request.method == 'POST':
            # Get input data from the frontend
            data = request.form.to_dict()

            # Load the diabetes model
            diabetes_model = joblib.load('Progno-Multiple-disease-predictor\Models\diabetes_model.pkl')

            # Make a prediction
            prediction = diabetes_model.predict([list(data.values())])

            # Process the prediction and prepare the result
            result = "Diabetes Positive" if prediction == 1 else "Diabetes Negative"

            # Return the result to the frontend
            return render_template('Progno-Multiple-disease-predictor\frontend\pages\result.html', prediction_result=result)

        elif request.method == 'GET':
            # Handle GET requests (if needed)
            return render_template('Progno-Multiple-disease-predictor\frontend\pages\result.html')

    except Exception as e:
        # Log the exception to the terminal
        print(f"An error occurred: {str(e)}")

        # Optionally, you can log the exception to a file or a logging service
        # Example: logging.error(f"An error occurred: {str(e)}")

        # Return an error message or redirect to an error page
        return render_template('error.html', error_message="An unexpected error occurred")

if __name__ == '__main__':
    app.run(debug=True)
