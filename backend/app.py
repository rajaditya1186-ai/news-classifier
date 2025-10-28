"""
════════════════════════════════════════════════════════════════════════════════
 BBC NEWS CLASSIFIER - FLASK BACKEND
 Serves ML model API for frontend
════════════════════════════════════════════════════════════════════════════════
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import re
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\n" + "="*80)
print("🚀 LOADING MODEL FILES FROM BACKEND FOLDER...")
print("="*80)

# Load model files from backend directory
try:
    model_path = os.path.join(BASE_DIR, 'naive_bayes_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Naive Bayes model loaded from: {model_path}")
except FileNotFoundError:
    print(f"❌ ERROR: naive_bayes_model.pkl not found in {BASE_DIR}")
    exit(1)

try:
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    print(f"✅ TF-IDF vectorizer loaded from: {tfidf_path}")
except FileNotFoundError:
    print(f"❌ ERROR: tfidf_vectorizer.pkl not found in {BASE_DIR}")
    exit(1)

try:
    encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✅ Label encoder loaded from: {encoder_path}")
except FileNotFoundError:
    print(f"❌ ERROR: label_encoder.pkl not found in {BASE_DIR}")
    exit(1)

print("\n✅ All model files loaded successfully!")
print(f"✅ Categories: {list(label_encoder.classes_)}")
print(f"✅ Vocabulary Size: {len(tfidf.vocabulary_):,}")

def clean_text(text):
    """Clean and normalize text for prediction"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def home():
    """API root endpoint"""
    return jsonify({
        'message': 'BBC News Classifier API',
        'status': 'running',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    News classification endpoint
    
    Request JSON:
        {
            "text": "news article text"
        }
    
    Response JSON:
        {
            "success": true,
            "prediction": {
                "category": "business",
                "confidence": 95.67
            },
            "all_probabilities": {
                "business": 95.67,
                "entertainment": 2.13,
                ...
            }
        }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400
        
        # Extract text
        text = data.get('text', '')
        
        # Validate
        if not text or len(text.strip()) < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short. Minimum 10 characters required.'
            }), 400
        
        # Preprocess
        cleaned_text = clean_text(text)
        
        # Vectorize
        text_tfidf = tfidf.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        # Get category name
        predicted_category = label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100
        
        # All probabilities
        all_probabilities = {}
        for i, category in enumerate(label_encoder.classes_):
            all_probabilities[category] = round(probabilities[i] * 100, 2)
        
        # Sort by probability
        all_probabilities = dict(sorted(all_probabilities.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True))
        
        # Response
        response = {
            'success': True,
            'prediction': {
                'category': predicted_category,
                'confidence': round(confidence, 2)
            },
            'all_probabilities': all_probabilities
        }
        
        # Log
        print(f"\n📰 Prediction: {predicted_category} ({confidence:.2f}%)")
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model': 'Multinomial Naive Bayes',
        'categories': list(label_encoder.classes_),
        'vocab_size': len(tfidf.vocabulary_)
    }), 200

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 BBC NEWS CLASSIFIER - BACKEND SERVER")
    print("="*80)
    print("\n📍 Server Info:")
    print("   URL: http://localhost:5000")
    print("   API: http://localhost:5000/predict")
    print("   Health: http://localhost:5000/health")
    print("\n📊 Model Info:")
    print(f"   Algorithm: Multinomial Naive Bayes")
    print(f"   Categories: {len(label_encoder.classes_)}")
    print(f"   Features: {len(tfidf.vocabulary_):,}")
    print("\n🌐 Frontend:")
    print("   Open: ../frontend/index.html in browser")
    print("\n🎯 Ready to classify news!")
    print("="*80)
    print("\n⚡ Press CTRL+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
