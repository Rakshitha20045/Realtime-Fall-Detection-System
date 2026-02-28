<p align="center">
  <h1 align="center">🚨 Real-Time Human Fall Detection System</h1>
</p>

<p align="center">
  <b>Computer Vision &nbsp;•&nbsp; Deep Learning &nbsp;•&nbsp; Pose Estimation &nbsp;•&nbsp; Streamlit &nbsp;•&nbsp; Healthcare AI &nbsp;•&nbsp; Real-Time Alerting</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10.11-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/YOLO-v11m-00FFAE?style=for-the-badge&logo=yolo"/>
  <img src="https://img.shields.io/badge/MediaPipe-Pose-FF6F00?style=for-the-badge&logo=google"/>
  <img src="https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv"/>
  <img src="https://img.shields.io/badge/Twilio-SMS%20%7C%20WhatsApp-F22F46?style=for-the-badge&logo=twilio"/>
</p>

---

## 📌 Overview

The **Real-Time Human Fall Detection System** is a production-grade AI surveillance application that monitors live camera streams, CCTV/RTSP feeds, or uploaded video files to automatically detect human falls — and immediately fires emergency alerts via SMS, WhatsApp, and voice calls.

Designed for **elderly care**, **hospitals**, **smart homes**, and **industrial safety**, this system bridges the gap between passive surveillance and active emergency response by combining state-of-the-art computer vision with multi-channel real-time notifications.

---
## 🖼️ Application Screenshots

---

### 🖥️ Streamlit Dashboard

<p align="center">
  <img src="images/streamlit-ui.png" width="48%">
  &nbsp;
  <img src="images/streamlit-ui-1.png" width="48%">
</p>

---

### 📹 Live Camera / CCTV Fall Detection

<p align="center">
  <img src="images/Live-fall-detect.png" width="48%">
  &nbsp;
  <img src="images/Live-fall-detect-1.png" width="48%">
</p>

---

### 🎞️ Video File-Based Fall Detection

<p align="center">
  <img src="images/video-fall-detect.png" width="48%">
  &nbsp;
  <img src="images/video-fall-detect-1.png" width="48%">
</p>

---

### 📲 Emergency Alerts

<p align="center">
  <img src="images/sms-alert.jpg" width="48%">
  &nbsp;
  <img src="images/whatsapp-alert.jpg" width="48%">
</p>

<p align="center">
<b>Left:</b> SMS Alert &nbsp;&nbsp;|&nbsp;&nbsp; <b>Right:</b> WhatsApp Alert — triggered automatically on fall detection
</p>

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 🎯 **YOLO11m Person Detection** | Detects persons in each frame with high accuracy using YOLOv11 Medium |
| 🦴 **MediaPipe Pose Estimation** | Extracts 33 full-body keypoints for precise body posture analysis |
| 📐 **Angle-Based Fall Logic** | Calculates aspect ratios and joint angles to classify falls intelligently |
| 📹 **Live CCTV / Webcam Stream** | Supports webcam index or RTSP/IP camera URL for real-time monitoring |
| 🎞️ **Video File Analysis** | Upload `.mp4`, `.avi`, `.mov`, `.mkv` files for batch fall detection |
| 📱 **SMS Alerts (Twilio)** | Instant SMS to emergency contacts with timestamp and camera location |
| 📞 **Voice Call Alerts (Twilio)** | Automated voice call using Twilio TwiML on fall detection |
| 💬 **WhatsApp Alerts** | Via Twilio WhatsApp API or pywhatkit (no account needed) |
| 🔊 **Emergency Alarm Sound** | Local alarm sound synthesized via pygame on fall detection |
| 📸 **Snapshot on Fall** | Auto-saves fall frame as `.jpg` snapshot for evidence logging |
| 📋 **Fall Event Log** | Persistent JSON log with timestamps and snapshot references |
| ⏱️ **Configurable Cooldown** | Adjustable alert cooldown (5–60s) to prevent notification flooding |
| 📊 **Live Stats Dashboard** | Real-time FPS counter, person count, total falls metric in sidebar |
| ⬇️ **Download Annotated Video** | Download processed output video with detection overlays |

---

## 🧠 System Architecture & Workflow

```
Input Source (Webcam / RTSP / Video File)
         │
         ▼
┌─────────────────────┐
│  Frame Preprocessing │  → Resize to 640×480
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  YOLO11m Detection  │  → Detect persons (conf ≥ 0.45)
└─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  MediaPipe Pose Estimation│  → Extract 33 body keypoints
└──────────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Fall Classification Logic│  → Aspect ratio + joint angle analysis
└──────────────────────────┘
         │
    ┌────┴────┐
    │  FALL?  │
    └────┬────┘
   YES   │    NO
    ▼         ▼
Alert Engine  Green Status
 ├── 🔊 Sound Alarm
 ├── 📱 SMS (Twilio)
 ├── 📞 Voice Call
 ├── 💬 WhatsApp
 └── 📸 Snapshot + Log
         │
         ▼
Streamlit UI (Annotated Frame + Overlays)
```

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.10.11 |
| Object Detection | Ultralytics YOLO11m | 8.4.18 |
| Pose Estimation | MediaPipe Pose | 0.10.9 |
| Image Processing | OpenCV | 4.13.0.92 |
| Numerical Computing | NumPy | 2.2.6 |
| Web UI Framework | Streamlit | 1.54.0 |
| SMS / Voice / WhatsApp | Twilio REST API | latest |
| Free WhatsApp | pywhatkit | latest |
| Audio Alert | pygame | latest |

---

## 📦 Dependencies

```txt
ultralytics==8.4.18
mediapipe==0.10.9
streamlit==1.54.0
opencv-python==4.13.0.92
numpy==2.2.6
twilio
pywhatkit
pygame
```

Install all at once:

```bash
pip install ultralytics==8.4.18 mediapipe==0.10.9 streamlit==1.54.0 opencv-python==4.13.0.92 numpy==2.2.6 twilio pywhatkit pygame
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/realtime-fall-detection-system.git
cd realtime-fall-detection-system
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download YOLO11m Model

The model downloads automatically on first run. Or download manually:

```python
from ultralytics import YOLO
model = YOLO("yolo11m.pt")  # auto-downloads
```

---

## ▶️ Run the Application

```bash
streamlit run Real_time_fall_detect.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 🔍 How to Check Versions Used in This Project

Run this in your terminal to verify all installed versions:

```bash
python -c "
import ultralytics, mediapipe, streamlit, cv2, numpy
print('ultralytics :', ultralytics.__version__)
print('mediapipe   :', mediapipe.__version__)
print('streamlit   :', streamlit.__version__)
print('opencv      :', cv2.__version__)
print('numpy       :', numpy.__version__)
"
```

Expected output:

```
ultralytics : 8.4.18
mediapipe   : 0.10.9
streamlit   : 1.54.0
opencv      : 4.13.0.92
numpy       : 2.2.6
```

---

## 📡 CCTV Integration (RTSP)

This system natively supports **RTSP-based IP cameras and CCTV systems** via OpenCV's `VideoCapture`.

### Connecting a CCTV Camera

In the **Live Camera** tab, select `CCTV / RTSP URL` and enter your stream URL:

```
rtsp://username:password@192.168.1.100:554/stream
rtsp://admin:admin@10.0.0.5:554/Streaming/Channels/101
```

### Supported CCTV Protocols

| Protocol | Support |
|---|---|
| RTSP (Real-Time Streaming Protocol) | ✅ Full Support |
| HTTP/MJPEG Streams | ✅ Full Support |
| Webcam (USB / Built-in) | ✅ Full Support |
| IP Camera ONVIF | ✅ via RTSP URL |
| NVR / DVR Multi-Channel | ✅ via individual RTSP channels |

### Future CCTV Roadmap

| Planned Feature | Description |
|---|---|
| 🎥 Multi-Camera Dashboard | Monitor 4–16 CCTV feeds simultaneously on a grid layout |
| 🔁 Auto-Reconnect | Resilient RTSP stream reconnection on network drop |
| 🗺️ Camera Zone Mapping | Draw regions of interest (ROI) per camera for zone-based alerts |
| ☁️ Cloud DVR Integration | Push annotated video clips to AWS S3 / Google Cloud Storage |
| 🔐 Camera Authentication | Secure credential management for enterprise CCTV systems |
| 🤖 Edge Deployment | Run on NVIDIA Jetson Nano / Raspberry Pi 5 for on-device inference |
| 📊 Analytics Dashboard | Heatmaps of fall locations, time-of-day distribution reports |

---

## 🎯 Real-World Applications

| Domain | Use Case |
|---|---|
| 👴 Elderly Care Homes | 24/7 autonomous monitoring of senior residents |
| 🏥 Hospitals & ICUs | Patient fall detection in wards and recovery rooms |
| 🏠 Smart Home Systems | Home safety monitoring for aging-in-place individuals |
| 🏭 Industrial Safety | Worker fall detection in factories and construction sites |
| 🏫 Rehabilitation Centers | Post-surgery patient movement monitoring |
| 🔒 Smart Surveillance | AI-enhanced CCTV for proactive incident response |

---

## ✅ Advantages Over Traditional Systems

| Traditional CCTV | This System |
|---|---|
| Passive recording only | Active real-time fall detection |
| Requires 24/7 human monitoring | Fully autonomous AI monitoring |
| No instant alerts | Multi-channel alerts in < 1 second |
| No event logs | Automatic JSON log with snapshots |
| Expensive enterprise software | Open-source and free to deploy |
| Fixed rule-based detection | Learned pose estimation with AI |

---

## 🔮 Future Enhancements

- **Multi-Person Tracking** — Assign unique IDs per person using DeepSORT for tracking across frames
- **Skeleton Visualization** — Draw full MediaPipe skeleton overlay on each detected person
- **Behavior History Analysis** — Detect "about to fall" pre-fall posture states using LSTM/GRU temporal modeling
- **Email Alerts** — SMTP-based email notifications with embedded snapshot attachments
- **Mobile App Integration** — Push notifications to iOS/Android via Firebase Cloud Messaging
- **GPU Acceleration** — TensorRT / CUDA optimization for Jetson and RTX deployment
- **Dashboard Analytics** — Weekly/monthly fall frequency reports with charts
- **Multi-Camera Grid View** — Simultaneous monitoring of multiple CCTV channels in a tiled layout
- **Cloud Deployment** — Dockerized deployment on AWS / GCP / Azure with Nginx reverse proxy
- **Active Learning** — Continuously retrain YOLO on new fall scenarios to improve accuracy over time

---
