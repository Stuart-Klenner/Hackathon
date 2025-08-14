# Import necessary libraries for the Flask application and API functionality
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource, fields
import joblib
import numpy as np
import os
import logging
import re
import nltk
from nltk.corpus import stopwords

# --- NLTK Setup for Text Cleaning ---
# This block attempts to download the necessary stopwords for text cleaning.
# If you get a 'LookupError', run the following in your terminal:
# python -c "import nltk; nltk.download('stopwords')"
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    logging.warning("NLTK stopwords not found. Please download them as instructed above.")
    stop_words = set()

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for all domains to allow
# the frontend to make requests to this API.
CORS(app)

# Initialize Flask-RESTx for API documentation and structure
# This creates a Swagger UI at http://localhost:5000/
api = Api(app, version='1.0', title='Phishing Detection API',
          description='A simple API to predict if a body of text is phishing or not.')

# Define a namespace for the prediction endpoint
# This helps organize the API and its documentation
ns = api.namespace('predict', description='Phishing prediction operations')

# --- Helper Function for Text Cleaning ---
def clean_text_for_prediction(text):
    """
    Cleans a string of text by removing non-alphabetic characters, converting to lowercase,
    and removing stopwords. This should match the preprocessing done during model training.
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Load the trained model and vectorizer
# We use a try-except block to handle cases where the files might not be present
try:
    model_path = os.path.join(os.getcwd(), 'svc_spam_detector_model.joblib')
    vectorizer_path = os.path.join(os.getcwd(), 'tfidf_vectorizer.joblib')
    spam_classifier_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    logging.info("Model and Vectorizer loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model or vectorizer: {e}. Please ensure the .joblib files are in the backend directory.")
    spam_classifier_model = None
    tfidf_vectorizer = None
    # We can't proceed without the model, so we can let the API run but the endpoint will return an error

# Define the data model for the input using Flask-RESTx fields
# This will be used in the API documentation and for input validation
text_input_model = api.model('TextInput', {
    'text': fields.String(required=True, description='The body of text to be analyzed for phishing.')
})

# Define the API resource (the actual endpoint logic)
@ns.route('/')
class PhishingPredictor(Resource):
    @ns.doc('predict_phishing')
    @ns.expect(text_input_model)
    def post(self):
        """
        Predicts if the input text is a phishing attempt and returns the probability.
        """
        if spam_classifier_model is None or tfidf_vectorizer is None:
            api.abort(500, "The model and vectorizer are not loaded. Please check the backend files.")

        # Get the JSON data from the request body
        data = request.json

        # Check if the 'text' key is in the request data
        if not data or 'text' not in data:
            api.abort(400, "Missing 'text' field in the request body.")

        text_to_predict = data['text']

        try:
            # Clean the input text using the new helper function
            cleaned_text = clean_text_for_prediction(text_to_predict)

            # Vectorize the cleaned text using the loaded TfidfVectorizer
            text_vector = tfidf_vectorizer.transform([cleaned_text])
            
            # Make the prediction and get the probabilities
            prediction = spam_classifier_model.predict(text_vector)
            probabilities = spam_classifier_model.predict_proba(text_vector)[0]
            
            # The prediction result is an array, so we take the first element
            result = int(prediction[0])
            
            # Get the probability for the "phishing" class (assuming 1 is phishing)
            phishing_probability = float(probabilities[1])

            # Prepare the response
            response = {
                "prediction": result,
                "message": "Phishing" if result == 1 else "Not Phishing",
                "probability": phishing_probability
            }
            return jsonify(response)
        
        except Exception as e:
            # Handle any errors during prediction
            api.abort(500, f"An error occurred during prediction: {str(e)}")

# This block ensures the app runs only when the script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
