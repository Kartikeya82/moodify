from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = load_model("better_emotion_model.h5")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Adjust if needed

def preprocess_image(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert('L').resize((48, 48))
    image = np.array(image) / 255.0
    image = image.reshape(1, 48, 48, 1)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image'].read()
    input_img = preprocess_image(image)

    predictions = model.predict(input_img)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({'emotion': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
