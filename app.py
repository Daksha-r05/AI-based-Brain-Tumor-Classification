from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)

MODEL_PATH = '/Users/sriranga/Desktop/brain_tumor_flask_app/brain_tumor_detection_model.keras'
model = load_model(MODEL_PATH)

IMG_SIZE = (150, 150)

# Define the categories as in training
categories = ["glioma", "meningioma", "notumor", "pituitary"]

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        preds = model.predict(img_array)
        pred_class = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        label = categories[pred_class]
        if label == "notumor":
            display_label = "No Tumor"
        else:
            display_label = f"Tumor Detected: {label.capitalize()}"
        return jsonify({'prediction': pred_class, 'confidence': confidence, 'label': label, 'display': display_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve other static files (CSS, JS, images) if needed
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True) 