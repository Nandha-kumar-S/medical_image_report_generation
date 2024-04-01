from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
from data.train_text_enc import preprocess_data
from scripts.infer_mlp import generate_report , load_mlp_model
app = Flask(__name__)

# Load MLP model
model_path = "saved_model\mlp_model.pkl"
mlp_model = load_mlp_model(model_path)

def extract_features(image):
    features = image.flatten()  # Flatten the image
    return features

@app.route('/')
def index():
    return 'MEDICAL IMAGE REPORT GENERATION MODEL'


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})
    
    image = request.files['image'].read()
    if len(image) == 0:
        return jsonify({'error': 'Empty image'})
    
    preprocessed_image = preprocess_data(image)
    features = extract_features(preprocessed_image)
    prediction = generate_report(mlp_model, features)
    
    return jsonify({'Report': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
