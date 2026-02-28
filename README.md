<p align="center">
  <h1 align="center">Elderly Abnormal Behavior Detection System</h1>
</p>

<p align="center">
  <b>Computer Vision • Deep Learning • Pose Estimation • Streamlit • Healthcare AI</b>
</p>

---

## 📌 Overview

The Elderly Abnormal Behavior Detection System is a computer vision-based application designed to detect abnormal behaviors such as falls in elderly individuals using video input.

This system combines:
- YOLOv11m for person detection
- MediaPipe Pose for human pose estimation (33 keypoints)
- Custom logic for abnormal behavior classification
- Streamlit for interactive web interface

The application allows users to upload videos, detect falls, visualize alerts, and download processed output videos.

---

## 🚀 Features

- Real-time person detection using YOLOv11m
- Pose estimation using MediaPipe
- Automatic fall detection and classification
- Visual alert overlay ("FALL DETECTED")
- Streamlit-based interactive UI
- Upload and process video files
- Download processed output video
- Fast and efficient detection pipeline

---

## 🧠 System Workflow

1. Person detection using YOLOv11m
2. Pose estimation using MediaPipe Pose
3. Extract body keypoints
4. Analyze posture and body angles
5. Classify behavior as Normal or Fall
6. Display alert if fall detected
7. Show results in Streamlit interface

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10.11 |
| Object Detection | Ultralytics YOLOv11m |
| Pose Estimation | MediaPipe Pose |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Web Interface | Streamlit |

---

## 📦 Dependencies

| Library | Version |
|---|---|
| mediapipe | 0.10.9 |
| streamlit | 1.54.0 |
| opencv-python | 4.13.0.92 |
| ultralytics | 8.4.18 |
| numpy | 2.2.6 |

---

## ⚙️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/elderly-abnormal-behavior-detection.git
cd elderly-abnormal-behavior-detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

## 📦 Dependencies

| Library | Version |
|---|---|
| Python | 3.10.11 |
| mediapipe | 0.10.9 |
| streamlit | 1.54.0 |
| opencv-python | 4.13.0.92 |
| ultralytics | 8.4.18 |
| numpy | 2.2.6 |

---

## ▶️ Run the Application

Navigate to the project folder:
```bash
cd path/to/project/folder
```

Run the Streamlit application:
```bash
streamlit run app.py
```

Open your browser and go to:
```
http://localhost:8501
```

---

## 🖼️ Application Preview

### Streamlit Interface
<p align="center">
  <img src="https://i.ibb.co/xt59yFKJ/Streamlit-UI.png" width="70%">
</p>

### Person Detection
<p align="center">
  <img src="https://i.ibb.co/Xx11GhBL/person-detection.png" width="45%">
</p>

### Fall Detection Alert
<p align="center">
  <img src="https://i.ibb.co/p6tbrx6c/Fall-Detected.png" width="45%">
</p>

---

## 📊 Output

The system provides:
- Person detection with bounding boxes
- Pose keypoints extraction
- Fall detection alerts
- Processed output video
- Downloadable processed results
---

## 🎯 Applications

- Elderly monitoring systems
- Healthcare monitoring systems
- Smart surveillance systems
- Assisted living safety systems
- Fall prevention and detection systems
