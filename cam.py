import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# ✅ Load Trained Model
model = load_model(r"D:\moodify\best_emotion_model.h5")
print("✅ Model Loaded Successfully!")

# ✅ Emotion Labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Initialize MTCNN Face Detector
detector = MTCNN()

# ✅ Start Webcam
cap = cv2.VideoCapture(0)

# ✅ Initialize Timer
last_capture_time = time.time()
current_mood = "Detecting..."

while True:
    # ✅ Read Frame from Webcam
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame!")
        break

    # ✅ Convert BGR to RGB for MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ✅ Detect Faces
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Ensure positive values
        
        # ✅ Extract Face ROI
        face_roi = rgb_frame[y:y+height, x:x+width]

        if face_roi.size == 0:
            continue  # Skip empty detections

        # ✅ Capture Emotion Every 5 Seconds
        if time.time() - last_capture_time >= 5:
            try:
                # REMOVE this line:
                # face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

                # Resize to match model input (128x128)
                face_roi = cv2.resize(face_roi, (96, 96))

                # Normalize (0 to 1) & Expand Dimensions
                face_roi = face_roi.astype("float32") / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)  # (1, 128, 128, 3)

                # Predict Emotion
                predictions = model.predict(face_roi)
                emotion_idx = np.argmax(predictions)
                current_mood = emotion_labels[emotion_idx]

                # Update Last Capture Time
                last_capture_time = time.time()

            except Exception as e:
                print(f"⚠️ Error processing face: {e}")

        # ✅ Draw Face Rectangle & Label
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
        cv2.putText(frame, current_mood, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # ✅ Display Current Mood on Webcam Feed
    cv2.putText(frame, f"Current Mood: {current_mood}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # ✅ Show Webcam Feed
    cv2.imshow("Emotion Detection", frame)

    # ✅ Press 'q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release Resources
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam Closed!")
