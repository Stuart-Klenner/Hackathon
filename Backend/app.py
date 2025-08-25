# Import necessary libraries for the Flask application and API functionality
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_restx import Api, Namespace, Resource, fields
import joblib
import numpy as np
import os
import logging
import re
from nltk.corpus import stopwords
import ssl

# --- SSL Setup ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- NLTK Setup ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    logging.warning("NLTK stopwords not found. Please download them as instructed above.")
    stop_words = set()

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)

# Initialize the Flask application
app = Flask(__name__)

CORS(app)

# Initialize Flask-RESTx API
# API documentation will be at a separate URL like /api/docs
api = Api(
    app,
    version='1.0',
    title='Phishing Detection API',
    description='A simple API to predict if a body of text is phishing or not.',
    doc='/api/docs'
)

# Define a namespace for the API endpoints
ns = Namespace('api', description='Phishing prediction operations')
api.add_namespace(ns)

# --- Frontend Route ---
# This is the dedicated route for your main web page
@app.route('/')
def index():
    return render_template('index.html')

# --- Helper Function for Text Cleaning ---
def clean_text_for_prediction(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# --- Load Model and Vectorizer ---
try:
    model_path = os.path.join(base_dir, 'svc_spam_detector_model.joblib')
    vectorizer_path = os.path.join(base_dir, 'tfidf_vectorizer.joblib')
    spam_classifier_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    logging.info("Model and Vectorizer loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model or vectorizer: {e}. Please ensure the .joblib files are in the backend directory.")
    spam_classifier_model = None
    tfidf_vectorizer = None

# Define API Models for Swagger/OpenAPI Documentation
text_input_model = api.model('TextInput', {
    'text': fields.String(required=True, description='The body of text to be analyzed for phishing.')
})

predict_response_model = api.model('PredictResponse', {
    "prediction": fields.Integer(description='0 for Not Phishing, 1 for Phishing'),
    "message": fields.String(description='The human-readable prediction.'),
    "probability": fields.Float(description='The confidence score of the prediction.')
})

# --- API Endpoint for Prediction ---
@ns.route('/predict')
class PhishingPredictor(Resource):
    @ns.doc('predict_phishing')
    @ns.expect(text_input_model)
    @ns.marshal_with(predict_response_model)
    def post(self):
        """
        Predicts if the input text is a phishing attempt.
        """
        if spam_classifier_model is None or tfidf_vectorizer is None:
            api.abort(500, "The model and vectorizer are not loaded. Please check the backend files.")
        
        data = request.json
        if not data or 'text' not in data:
            api.abort(400, "Missing 'text' field in the request body.")

        text_to_predict = data['text']
        try:
            cleaned_text = clean_text_for_prediction(text_to_predict)
            text_vector = tfidf_vectorizer.transform([cleaned_text])
            prediction = spam_classifier_model.predict(text_vector)
            probabilities = spam_classifier_model.predict_proba(text_vector)[0]
            result = int(prediction[0])
            phishing_probability = float(probabilities[1])

            response = {
                "prediction": result,
                "message": "Phishing" if result == 1 else "Not Phishing",
                "probability": phishing_probability
            }
            return response
        
        except Exception as e:
            api.abort(500, f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)