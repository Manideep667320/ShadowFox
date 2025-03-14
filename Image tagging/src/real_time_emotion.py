import cv2
import numpy as np
import tensorflow as tf
from face_detection import detect_faces

# Load trained model
model = tf.keras.models.load_model("../models/emotion_detection_model.h5")

# Emotion labels
emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    faces = detect_faces(frame)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_roi = cv2.resize(face_roi, (48, 48)) / 255.0
        face_roi = np.reshape(face_roi, (1, 48, 48, 1))

        # Predict emotion
        prediction = model.predict(face_roi)
        emotion = emotion_labels[np.argmax(prediction)]

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show output
    cv2.imshow("Emotion Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
