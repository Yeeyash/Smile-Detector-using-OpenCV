# 😄 Smile Detector using OpenCV & MediaPipe

A real-time smile detection web application built with **OpenCV**, **MediaPipe**, and **FastAPI**. The app uses a webcam to detect faces and determine whether the user is smiling, displaying the results on a frontend built with vanila HTML/CSS.

---

## 🚀 Features

- Real-time face and smile detection
- Accurate results using MediaPipe's face mesh
- Lightweight and fast performance via FastAPI backend
- Simple and clean web interface using HTML & CSS
- Project also includes files for detecting Faces and Hands.

---

## 🧰 Tech Stack

- **Python**
  - `opencv-python` (`cv2`) for video capture and image processing
  - `mediapipe` for face mesh and smile detection logic, also tracks and detects lips.
  - `fastapi` for serving the backend and APIs
- **Frontend**
  - `HTML` and `CSS` for UI layout and styling

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Yeeyash/Smile-Detector-using-OpenCV.git
```
---

## ▶️ Running the App
Start the FastAPI server:
```bash
uvicorn smilepage:app --reload
```
Then open your browser and navigate to:
```bash
http://127.0.0.1:8000
```

---

## 🧠 How It Works

1. OpenCV captures real-time video from the webcam.
2. MediaPipe's face mesh model detects key facial landmarks.
3.A smile is detected by analyzing the distances and movement between key mouth landmarks.
4.  Results are streamed to the browser using FastAPI endpoints.
5. The HTML/CSS frontend renders the webcam feed and updates smile status dynamically.

## 🙌 Acknowledgements

- <a href="https://opencv.org/">OpenCV</a>

- <a href="https://ai.google.dev/edge/mediapipe/solutions/guide">MediaPipe</a>

- <a href="https://fastapi.tiangolo.com/">FastAPI</a>
