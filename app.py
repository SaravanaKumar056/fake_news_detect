# app.py

import flask
import pickle
import re
import string
import pandas as pd
from flask import Flask, request, render_template, jsonify

print("Loading model and vectorizer...")
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))
print("Model and vectorizer loaded successfully.")

def wordopt(text):
    """Cleans and preprocesses the input text."""
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives news text, predicts, and returns JSON."""
    if request.method == 'POST':
        news_text = request.form['news_text']

        if not news_text.strip():
            return jsonify({'error': 'Input text cannot be empty.'}), 400

        processed_text = wordopt(news_text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)

        if prediction[0] == 1:
            result = "This looks like a Reliable News source."
            label = "reliable"
        else:
            result = "This appears to be Fake News."
            label = "fake"
            
        return jsonify({'prediction_text': result, 'label': label})

if __name__ == '__main__':
    app.run(debug=True)


