from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from mtcnn import MTCNN
import cv2

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001"  # <-- Add this line
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model = tf.keras.models.load_model(r'd:\moodify\GUI\model\better_emotion_model.h5')

# Define your emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

detector = MTCNN()

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(image)
    faces = detector.detect_faces(img_array)
    print(f"Detected faces: {faces}")  # Log detected faces
    if faces:
        x, y, width, height = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_roi = img_array[y:y+height, x:x+width]
        face_roi = cv2.resize(face_roi, (128, 128))
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
        prediction = model.predict(face_roi)
        print(f"Prediction output: {prediction}")  # Log prediction output
        emotion_idx = int(np.argmax(prediction))
        emotion = EMOTIONS[emotion_idx]
        print(f"Detected emotion: {emotion}")  # Log detected emotion
        return {"emotion": emotion}
    else:
        return {"emotion": "No face detected"}