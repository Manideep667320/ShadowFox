# Emotion Detection using Facial Recognition

## 📌 Overview
This project detects emotions in real-time using a webcam. It uses a pre-trained model to identify emotions from facial expressions.

## 📂 Directory Structure
- `models/` → Trained models
  - `emotion_detection_model.h5` → Pre-trained emotion detection model
- `results/` → Prediction results
  - `emotion_predictions.csv` → CSV file containing emotion predictions
- `src/` → Code for training and real-time detection
  - `face_detection.py` → Script for detecting faces in images
  - `haarcascade_frontalface_default.xml` → Haar Cascade XML file for face detection
  - `predict_emotion.py` → Script for predicting emotions from detected faces
  - `real_time_emotion.py` → Script for real-time emotion detection using webcam
  - `train_model.py` → Script for training the emotion detection model

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python src/train_model.py`
3. Run real-time detection: `python src/real_time_emotion.py`

## 🛠️ Dependencies
- Python 3.7+
- TensorFlow
- OpenCV
- NumPy
- Pandas

## 📊 Results
The results of the emotion detection can be found in the `results/emotion_predictions.csv` file.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License
This project is licensed under the MIT License.
