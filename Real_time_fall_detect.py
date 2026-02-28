import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import tempfile, os, time, threading, json
from datetime import datetime
from pathlib import Path

# ──imports ─────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_OK = True
except ImportError:
    TWILIO_OK = False

try:
    import pywhatkit
    PYWHATKIT_OK = True
except ImportError:
    PYWHATKIT_OK = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model loading (cached — loads only once) ─────────────────
@st.cache_resource
def load_models():
    mp_pose_lib = mp.solutions.pose
    pose_model = mp_pose_lib.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # UPDATED TO YOLOv11 Medium
    yolo_model = YOLO("yolo11m.pt")
    return pose_model, yolo_model, mp_pose_lib

pose_detector, person_model, mp_pose_lib = load_models()

# ── Constants ────────────────────────────────────────────────
PROC_W, PROC_H = 640, 480
YOLO_CONF      = 0.45
VIS_THRESH     = 0.30
FALL_AR_THRESH = 0.70
SNAPSHOT_DIR   = Path("fall_snapshots")
LOG_FILE       = Path("fall_log.json")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# ── Session state defaults ───────────────────────────────────
for k, v in {
    "last_alert_ts": 0,
    "total_falls": 0,
    "fall_log": [],
    "stream_active": False,
    "alert_cooldown": 10,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════
#  ALERT ENGINE
# ════════════════════════════════════════════════════════════

def play_emergency_sound():
    if not PYGAME_OK:
        return
    def _play():
        try:
            sr, seg = 44100, 0.25
            parts = []
            for freq in [880, 1100, 880, 1100]:
                t    = np.linspace(0, seg, int(sr * seg), endpoint=False)
                wave = (np.sin(2 * np.pi * freq * t) * 28000).astype(np.int16)
                parts.append(wave)
            full   = np.concatenate(parts)
            stereo = np.column_stack([full, full])
            pygame.sndarray.make_sound(stereo).play()
            time.sleep(len(full) / sr + 0.2)
        except Exception as e:
            print(f"[Sound Error] {e}")
    threading.Thread(target=_play, daemon=True).start()

def send_twilio_sms(cfg: dict, snapshot_info: str = ""):
    if not TWILIO_OK:
        return
    def _sms():
        try:
            client = TwilioClient(cfg["account_sid"], cfg["auth_token"])
            ts     = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            loc    = cfg.get("location", "Monitored Camera")
            msg    = (
                f"FALL DETECTED\n"
                f"Time    : {ts}\n"
                f"Camera  : {loc}\n\n"
                f"Please check the person immediately!\n"
                f"Emergency: 108 / 112"
            )
            client.messages.create(
                body=msg,
                from_=cfg["twilio_number"],
                to=cfg["recipient_number"],
            )
            print(f"[SMS] Sent to {cfg['recipient_number']}")
        except Exception as e:
            print(f"[SMS Error] {e}")
    threading.Thread(target=_sms, daemon=True).start()


def make_twilio_call(cfg: dict):
    if not TWILIO_OK: return
    def _call():
        try:
            client = TwilioClient(cfg["account_sid"], cfg["auth_token"])
            call = client.calls.create(
                twiml='<Response><Say voice="alice">Emergency! A fall has been detected by the camera system. Please check immediately.</Say></Response>',
                to=cfg["recipient_number"],
                from_=cfg["twilio_number"]
            )
            print(f"[Voice Call] Calling {cfg['recipient_number']}...")
        except Exception as e:
            print(f"[Call Error] {e}")
    threading.Thread(target=_call, daemon=True).start()

def send_twilio_whatsapp(cfg: dict):
    if not TWILIO_OK:
        return
    def _wa():
        try:
            client = TwilioClient(cfg["account_sid"], cfg["auth_token"])
            ts     = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            loc    = cfg.get("location", "Camera")
            client.messages.create(
                body=(
                    f"*🚨 FALL DETECTED*\n"
                    f"📍 Camera  : {loc}\n"
                    f"🕐 Time    : {ts}\n\n"
                    f"Please check immediately!\n"
                    f"Emergency: *108 / 112*"
                ),
                from_="whatsapp:+14155238886",
                to=f"whatsapp:{cfg['recipient_number']}",
            )
            print(f"[WhatsApp-Twilio] Sent to {cfg['recipient_number']}")
        except Exception as e:
            print(f"[WhatsApp-Twilio Error] {e}")
    threading.Thread(target=_wa, daemon=True).start()

def send_free_whatsapp(phone_number: str, location: str = "Camera"):
    if not PYWHATKIT_OK:
        return
    def _send():
        try:
            ts  = datetime.now().strftime("%d %b %Y, %H:%M:%S")
            msg = (
                f"FALL DETECTED\n"
                f"Camera: {location}\n"
                f"Time: {ts}\n"
                f"Check immediately! Emergency: 108/112"
            )
            pywhatkit.sendwhatmsg_instantly(
                phone_no=phone_number,
                message=msg,
                wait_time=15,
                tab_close=True,
                close_time=3,
            )
            print(f"[WhatsApp-Free] Sent to {phone_number}")
        except Exception as e:
            print(f"[WhatsApp-Free Error] {e}")
    threading.Thread(target=_send, daemon=True).start()

def save_snapshot(frame) -> str:
    try:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = str(SNAPSHOT_DIR / f"fall_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path
    except Exception:
        return None

def log_fall(snapshot_path: str):
    ev = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snapshot":  snapshot_path or "",
    }
    st.session_state["fall_log"].append(ev)
    st.session_state["total_falls"] += 1
    try:
        existing = json.loads(LOG_FILE.read_text()) if LOG_FILE.exists() else []
        existing.append(ev)
        LOG_FILE.write_text(json.dumps(existing[-100:], indent=2))
    except Exception:
        pass

def trigger_alerts(frame, cfg: dict):
    now = time.time()
    if now - st.session_state["last_alert_ts"] < st.session_state["alert_cooldown"]:
        return   # still in cooldown
    st.session_state["last_alert_ts"] = now

    snap = save_snapshot(frame)
    log_fall(snap)

    if cfg.get("sound"):
        play_emergency_sound()

    if cfg.get("twilio_sms") and cfg.get("twilio_cfg"):
        send_twilio_sms(cfg["twilio_cfg"])

    if cfg.get("free_wa") and cfg.get("free_wa_number"):
        send_free_whatsapp(
            phone_number=cfg["free_wa_number"],
            location=cfg.get("location", "Camera"),
        )
    # --- NEW CALL BLOCK ---
    if cfg.get("twilio_call") and cfg.get("twilio_cfg"):
        make_twilio_call(cfg["twilio_cfg"])

    if cfg.get("twilio_wa") and cfg.get("twilio_cfg"):
        send_twilio_whatsapp(cfg["twilio_cfg"])

# ════════════════════════════════════════════════════════════
#  FALL DETECTION CORE
# ════════════════════════════════════════════════════════════

def _classify_fall(landmarks) -> bool:
    arr     = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks.landmark])
    visible = arr[arr[:, 2] > VIS_THRESH]
    
    # Lowered the requirement from 5 visible landmarks to 3
    if visible.shape[0] < 3: 
        return False

    bw = (visible[:, 0].max() - visible[:, 0].min()) * PROC_W
    bh = (visible[:, 1].max() - visible[:, 1].min()) * PROC_H
    
    # Criterion 1: Is the pose wider than it is tall?
    ar_fall = bw > 15 and bh > 15 and bh < bw * FALL_AR_THRESH

    kp_fall = False
    try:
        PL   = mp_pose_lib.PoseLandmark
        nose = landmarks.landmark[PL.NOSE]
        lsho = landmarks.landmark[PL.LEFT_SHOULDER]
        rsho = landmarks.landmark[PL.RIGHT_SHOULDER]
        lhip = landmarks.landmark[PL.LEFT_HIP]
        rhip = landmarks.landmark[PL.RIGHT_HIP]
        
        # Check if we can see at least the upper body
        if nose.visibility > VIS_THRESH and lsho.visibility > VIS_THRESH and rsho.visibility > VIS_THRESH:
            sho_y   = (lsho.y + rsho.y) / 2
            
            # If we can see hips, do the full calculation
            if lhip.visibility > VIS_THRESH and rhip.visibility > VIS_THRESH:
                hip_y   = (lhip.y + rhip.y) / 2
                sho_x   = (lsho.x + rsho.x) / 2
                hip_x   = (lhip.x + rhip.x) / 2
                torso_v = abs(sho_y - hip_y)
                torso_h = abs(sho_x - hip_x)
                if torso_h > 0.01 and torso_v < torso_h * 0.8 and nose.y > hip_y - 0.1:
                    kp_fall = True
            
            # Fallback: If hips are blocked by furniture, just check if nose is lower than shoulders
            elif nose.y > sho_y: 
                kp_fall = True
    except Exception:
        pass

    # CHANGED: Now it triggers if EITHER the aspect ratio is wide OR the keypoints say fall
    return ar_fall or kp_fall

def process_frame(frame: np.ndarray):
    orig_h, orig_w = frame.shape[:2]
    sx = orig_w / PROC_W
    sy = orig_h / PROC_H

    small    = cv2.resize(frame, (PROC_W, PROC_H))
    yolo_out = person_model.predict(
        small, imgsz=(PROC_H, PROC_W), conf=YOLO_CONF, classes=[0], verbose=False
    )[0]
    pose_out = pose_detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    display  = frame.copy()
    persons  = []
    fall     = False

    if pose_out.pose_landmarks:
        lm      = pose_out.pose_landmarks
        is_fall = _classify_fall(lm)
        if is_fall:
            fall = True

        pts = np.array([[l.x, l.y] for l in lm.landmark])
        cx  = float(pts[:, 0].mean()) * PROC_W
        cy  = float(pts[:, 1].mean()) * PROC_H
        box = None
        if yolo_out.boxes:
            for b in yolo_out.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                if x1 < cx < x2 and y1 < cy < y2:
                    box = [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)]
                    break
        if box:
            persons.append({"box": box, "fall": is_fall})

    # --- YOLO FALLBACK (If MediaPipe skeleton completely fails) ---
    # --- YOLO FALLBACK (If MediaPipe skeleton completely fails) ---
    if not persons and yolo_out.boxes:
        for b in yolo_out.boxes:
            # Scale coordinates back to original frame size
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            
            # Calculate width and height of the YOLO bounding box
            box_w = x2 - x1
            box_h = y2 - y1
            
            # If the box is significantly wider than it is tall, assume it's a fallen person
            yolo_is_fall = False
            
            # Check for minimum size to ignore small background noise
            if box_w > 40 and box_h > 15: 
                # Aspect ratio check: Fallen persons are wider than they are tall
                if box_h < (box_w * 0.85): 
                    yolo_is_fall = True
                    fall = True # Trigger the global alert for SMS/Call

            # Use Red for Fall, Orange for standing person with unknown pose
            color = (0, 0, 255) if yolo_is_fall else (0, 165, 255)
            label = "FALL DETECTED (YOLO)" if yolo_is_fall else "PERSON"
            
            # Draw on original frame using calculated scaling
            cv2.rectangle(display,
                          (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                          color, 3)
            cv2.putText(display, label, (int(x1*sx), int(y1*sy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    for p in persons:
        x1, y1, x2, y2 = p["box"]
        color = (0, 0, 255) if p["fall"] else (0, 210, 0)
        label = "FALL DETECTED" if p["fall"] else "STANDING"
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x1, y1 - th - 14), (x1 + tw + 8, y1), color, -1)
        cv2.putText(display, label, (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    banner = "!! FALL DETECTED !!" if fall else "Monitoring: Normal"
    bcolor = (0, 0, 255)          if fall else (0, 200, 50)
    if fall:
        cv2.rectangle(display, (0, 0), (orig_w - 1, orig_h - 1), (0, 0, 255), 8)
    cv2.rectangle(display, (0, 0), (orig_w, 60), (20, 20, 20), -1)
    cv2.putText(display, banner, (16, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, bcolor, 2, cv2.LINE_AA)

    ts = datetime.now().strftime("%d %b %Y  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(display, ts, (orig_w - tw - 12, orig_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return display, fall, len(persons)

def process_video_file(in_path: str, out_path: str, alert_cfg: dict) -> bool:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        st.error("Cannot open video file.")
        return False

    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not writer.isOpened():
        st.error("Cannot create output video. Check codec support.")
        cap.release()
        return False

    bar    = st.progress(0)
    status = st.empty()
    falls  = 0
    count  = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            annotated, fall, _ = process_frame(frame)
            if fall:
                falls += 1
                trigger_alerts(annotated, alert_cfg)
            writer.write(annotated)
            if total > 0:
                bar.progress(min(int(count / total * 100), 100))
                status.text(f"Frame {count}/{total}  |  Falls detected: {falls}")
    finally:
        cap.release()
        writer.release()

    bar.progress(100)
    status.text(f"Complete — {count} frames processed, {falls} fall events detected.")
    return True

# ════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📱 Alert Configuration")
    st.divider()

    sound_on = st.toggle("🔊 Emergency Alarm Sound", value=True)
    if not PYGAME_OK and sound_on:
        st.caption("Install: `pip install pygame`")

    st.divider()
    st.subheader("Phone Alerts")

    #st.markdown("**Option A — Twilio** *(professional, for resume)*")
    st.caption("Free trial at twilio.com — gives SMS + WhatsApp")

    twilio_sms_on = st.toggle("📱 SMS via Twilio",       value=False)
    twilio_call_on = st.toggle("📞 Voice Call via Twilio", value=False)
    twilio_wa_on  = st.toggle("💬 WhatsApp via Twilio",  value=False)

    twilio_cfg = {}
    if twilio_sms_on or twilio_wa_on or twilio_call_on:
        if not TWILIO_OK:
            st.warning("Install: `pip install twilio`")
        else:
            with st.expander("Twilio Credentials", expanded=True):
                twilio_cfg["account_sid"]      = st.text_input("Account SID", value="")
                twilio_cfg["auth_token"]       = st.text_input("Auth Token", type="password", value="")
                twilio_cfg["twilio_number"]    = st.text_input("Your Twilio Number", value="")
                twilio_cfg["recipient_number"] = st.text_input("Alert Phone Number", value="")

    st.divider()

    st.markdown("**Option B — Free WhatsApp** *(no account needed)*")
    free_wa_on = st.toggle("💬 Free WhatsApp (pywhatkit)", value=False)
    free_wa_number = ""
    free_wa_location = "Camera"

    if free_wa_on:
        if not PYWHATKIT_OK:
            st.warning("Install: `pip install pywhatkit`")
        else:
            with st.expander("Free WhatsApp Settings", expanded=True):
                free_wa_number = st.text_input("Recipient WhatsApp Number", placeholder="+91XXXXXXXXXX")
                free_wa_location = st.text_input("Camera Label", value="Living Room")

    st.divider()
    cooldown = st.slider("Alert Cooldown (seconds)", 5, 60, 10)
    st.session_state["alert_cooldown"] = cooldown

    st.divider()
    st.subheader("📊 Stats")
    st.metric("Total Falls Detected", st.session_state["total_falls"])

    if st.button("Reset Counter"):
        st.session_state["total_falls"] = 0
        st.session_state["fall_log"]    = []
        st.rerun()

alert_cfg = {
    "sound":          sound_on,
    "twilio_sms":     twilio_sms_on,
    "twilio_wa":      twilio_wa_on,
    "twilio_cfg":     twilio_cfg if (twilio_sms_on or twilio_wa_on or twilio_call_on) else None,
    "free_wa":        free_wa_on,
    "free_wa_number": free_wa_number,
    "location":       free_wa_location,
}

st.title("🚨 Real-Time Human Fall Detection System")
st.caption("YOLO11m + MediaPipe Pose  |  CCTV / Webcam / Video File  |  Phone Alerts")

tab_live, tab_file, tab_log = st.tabs(["📹 Live Camera / CCTV", "📁 Video File Analysis", "📋 Fall Event Log"])

with tab_live:
    col_feed, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        source_type = st.radio("Camera Source", ["Webcam", "CCTV / RTSP URL"])
        if source_type == "Webcam":
            cam_src = int(st.number_input("Camera Index", 0, 10, 0))
        else:
            cam_src = st.text_input("RTSP Stream URL", placeholder="rtsp://...")
        start_btn = st.button("▶ Start Monitoring", type="primary", use_container_width=True)
        stop_btn  = st.button("⏹ Stop Monitoring",                  use_container_width=True)
        fps_ph    = st.empty()
        status_ph = st.empty()
        count_ph  = st.empty()

    with col_feed:
        feed_ph = st.empty()

    if start_btn and cam_src != "":
        st.session_state["stream_active"] = True
    if stop_btn:
        st.session_state["stream_active"] = False

    if st.session_state.get("stream_active") and cam_src != "":
        cap = cv2.VideoCapture(cam_src)
        if not cap.isOpened():
            st.error("Cannot connect to camera.")
            st.session_state["stream_active"] = False
        else:
            prev_t = time.time()
            while st.session_state.get("stream_active"):
                ret, frame = cap.read()
                if not ret:
                    time.sleep(1)
                    cap = cv2.VideoCapture(cam_src)
                    continue

                annotated, fall, n = process_frame(frame)
                if fall:
                    trigger_alerts(annotated, alert_cfg)
                    status_ph.error("🚨 FALL DETECTED!")
                else:
                    status_ph.success("✅ Normal")

                now    = time.time()
                fps_ph.caption(f"FPS: {1.0 / max(now - prev_t, 1e-6):.1f}")
                count_ph.caption(f"Persons in frame: {n}")
                prev_t = now
                feed_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            cap.release()
            feed_ph.empty()
            status_ph.info("Monitoring stopped.")

with tab_file:
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        ext = "." + uploaded.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(uploaded.read())
            in_path = f.name
        st.video(in_path)
        if st.button("🔍 Run Fall Detection", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix="_out.mp4") as f:
                out_path = f.name
            ok = process_video_file(in_path, out_path, alert_cfg)
            if ok and os.path.getsize(out_path) > 0:
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download Annotated Video", f.read(), "fall_detection_output.mp4", "video/mp4")

with tab_log:
    if not st.session_state["fall_log"]:
        st.info("No falls detected yet.")
    else:
        for i, ev in enumerate(reversed(st.session_state["fall_log"])):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.error(f"Fall #{len(st.session_state['fall_log']) - i}")
                st.caption(ev["timestamp"])
            with c2:
                if ev.get("snapshot") and os.path.exists(ev["snapshot"]):
                    st.image(ev["snapshot"], use_container_width=True)
            st.divider()
