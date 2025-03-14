import cv2
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore

# Load trained model
model = load_model("../models/emotion_detection_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Define test dataset path
test_dir = r"C:\AICTE\Tagging\test"

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to predict emotion from an image
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]

        results.append({"Image": os.path.basename(image_path), "Emotion": emotion})

    return results

# Process all images from test dataset
all_results = []
for subdir, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(subdir, file)
            all_results.extend(predict_emotion(img_path))

# Define the path for the CSV file
csv_path = "../results/emotion_predictions.csv"

# Ensure the results directory exists
os.makedirs(os.path.dirname(csv_path), exist_ok=True)

# Save results to CSV
df = pd.DataFrame(all_results)
df.to_csv(csv_path, index=False)

print(f"Emotion predictions saved to {csv_path}")
