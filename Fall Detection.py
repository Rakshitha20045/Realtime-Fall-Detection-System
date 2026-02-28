import streamlit as st
import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import time

# --- Model Initialization (Consider caching for performance in deployed apps) ---
@st.cache_resource # Use st.cache_resource for models
def load_models():
    """Loads the YOLO and MediaPipe models."""
    mp_pose_solution = mp.solutions.pose
    pose_detector_model = mp_pose_solution.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    person_detector_model = YOLO('yolov8n.pt')
    return pose_detector_model, person_detector_model, mp_pose_solution

pose_detector, person_model, mp_pose = load_models()

# --- Processing Parameters ---
RESIZED_WIDTH, RESIZED_HEIGHT = 640, 480
YOLO_CONFIDENCE_THRESHOLD = 0.45
LANDMARK_VISIBILITY_THRESHOLD = 0.6
FALL_ASPECT_RATIO_THRESHOLD = 0.7

# --- Core Processing Function ---
def process_video(input_video_path, output_video_path):
    """
    Processes the input video for fall detection and saves the output.
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the processed video file.
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open input video: {input_video_path}")
        return False

    out = None # Initialize out to None for the finally block

    try:
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if orig_width == 0 or orig_height == 0:
            st.error("Video source has zero width or height. Cannot process.")
            return False # No need to release cap yet as it will be done in finally

        if fps == 0:
            st.warning("FPS could not be determined from video, defaulting to 25.")
            fps = 25.0 # Ensure fps is a float if it might be used in calculations expecting float

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec and create VideoWriter object
        # 'mp4v' is a common codec for .mp4 files.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

        if not out.isOpened():
            st.error(f"Error: Could not open VideoWriter for output path: {output_video_path}. "
                            "This might be due to an issue with OpenCV's video codecs (e.g., FFmpeg backend) "
                            "or permissions to write the file.")
            return False

        scale_x = orig_width / RESIZED_WIDTH
        scale_y = orig_height / RESIZED_HEIGHT

        progress_bar = st.progress(0)
        status_text = st.empty()

        frames_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video or error reading frame

            frames_processed += 1
            if total_frames > 0:
                progress = int((frames_processed / total_frames) * 100)
                progress_bar.progress(min(progress, 100)) # Ensure progress doesn't exceed 100
                status_text.text(f"Processing: {min(progress, 100)}%")
            else:
                status_text.text(f"Processing frame: {frames_processed}")

            # --- Actual frame processing ---
            resized_frame = cv2.resize(frame, (RESIZED_WIDTH, RESIZED_HEIGHT))
            display_frame = frame.copy() # Work on a copy of the original frame for drawing

            # Person detection
            person_results = person_model.predict(resized_frame, imgsz=(RESIZED_HEIGHT, RESIZED_WIDTH), conf=YOLO_CONFIDENCE_THRESHOLD, classes=[0], verbose=False)[0]

            # Pose estimation
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pose_results_mediapipe = pose_detector.process(rgb_frame)

            detected_persons_info = []
            main_pose_landmarks = pose_results_mediapipe.pose_landmarks

            if main_pose_landmarks:
                lm_coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in main_pose_landmarks.landmark])
                visible_landmarks = lm_coords[lm_coords[:, 3] > LANDMARK_VISIBILITY_THRESHOLD]
                current_pose_is_fall = False

                if visible_landmarks.shape[0] >= 5: # Need enough visible landmarks for reliable calculation
                    pose_min_x_norm = np.min(visible_landmarks[:, 0])
                    pose_max_x_norm = np.max(visible_landmarks[:, 0])
                    pose_min_y_norm = np.min(visible_landmarks[:, 1])
                    pose_max_y_norm = np.max(visible_landmarks[:, 1])

                    pose_bbox_width_px = (pose_max_x_norm - pose_min_x_norm) * RESIZED_WIDTH
                    pose_bbox_height_px = (pose_max_y_norm - pose_min_y_norm) * RESIZED_HEIGHT

                    aspect_ratio_indicates_fall = False
                    if pose_bbox_width_px > 10 and pose_bbox_height_px > 10: # Basic check for valid pose box
                        if pose_bbox_height_px < (pose_bbox_width_px * FALL_ASPECT_RATIO_THRESHOLD):
                            aspect_ratio_indicates_fall = True

                    keypoints_indicate_fall = False
                    try:
                        nose = main_pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        l_shoulder = main_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        r_shoulder = main_pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        l_hip = main_pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                        r_hip = main_pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

                        if all(lm.visibility > LANDMARK_VISIBILITY_THRESHOLD for lm in [nose, l_shoulder, r_shoulder, l_hip, r_hip]):
                            avg_shoulder_y_norm = (l_shoulder.y + r_shoulder.y) / 2
                            avg_hip_y_norm = (l_hip.y + r_hip.y) / 2
                            torso_vertical_diff_norm = abs(avg_shoulder_y_norm - avg_hip_y_norm)
                            avg_shoulder_x_norm = (l_shoulder.x + r_shoulder.x) / 2
                            avg_hip_x_norm = (l_hip.x + r_hip.x) / 2
                            torso_horizontal_diff_norm = abs(avg_shoulder_x_norm - avg_hip_x_norm)

                            if torso_horizontal_diff_norm > 0.01 and torso_vertical_diff_norm < torso_horizontal_diff_norm * 0.7:
                                if nose.y > avg_hip_y_norm - 0.05: # nose.y > avg_hip.y means nose is physically lower
                                    keypoints_indicate_fall = True
                    except Exception: # Broad exception for landmark access issues
                        pass

                    if aspect_ratio_indicates_fall and keypoints_indicate_fall:
                        current_pose_is_fall = True

                    # Associate with YOLO box
                    pose_center_x_px = (pose_min_x_norm + pose_max_x_norm) / 2 * RESIZED_WIDTH
                    pose_center_y_px = (pose_min_y_norm + pose_max_y_norm) / 2 * RESIZED_HEIGHT
                    best_matching_yolo_box_coords = None

                    if person_results.boxes:
                        for person_box_data in person_results.boxes:
                            x1_y, y1_y, x2_y, y2_y = person_box_data.xyxy[0].cpu().numpy()
                            if (x1_y < pose_center_x_px < x2_y and y1_y < pose_center_y_px < y2_y):
                                best_matching_yolo_box_coords = [x1_y, y1_y, x2_y, y2_y]
                                break # Found a matching box for this pose

                    if best_matching_yolo_box_coords:
                        detected_persons_info.append({
                            "bbox_yolo_orig": [int(c * scale_x) if i % 2 == 0 else int(c * scale_y) for i, c in enumerate(best_matching_yolo_box_coords)],
                            "is_fall": current_pose_is_fall,
                        })
            # --- End of frame processing ---

            # --- Drawing and Status Update ---
            overall_status = "Status: Normal"
            overall_color = (0, 255, 0) # Green
            any_fall_detected_this_frame = any(p_info["is_fall"] for p_info in detected_persons_info)

            if any_fall_detected_this_frame:
                overall_status = "ALERT: FALL DETECTED"
                overall_color = (0, 0, 255) # Red
            elif detected_persons_info: # People processed by MediaPipe, no falls
                overall_status = "Status: Normal (Person(s) Standing)"
                overall_color = (0, 255, 0) # Green
            elif person_results.boxes: # YOLO detected people, but MediaPipe didn't yield info
                overall_status = "Person(s) Detected (Pose Undetermined)"
                overall_color = (0, 165, 255) # Orange
                for person_box_data in person_results.boxes: # Draw raw YOLO if pose undetermined
                    x1, y1, x2, y2 = person_box_data.xyxy[0].cpu().numpy()
                    x1_o, y1_o, x2_o, y2_o = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                    cv2.rectangle(display_frame, (x1_o, y1_o), (x2_o, y2_o), overall_color, 2)
                    cv2.putText(display_frame, "PERSON?", (x1_o, y1_o - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, overall_color, 2)
            else: # No YOLO boxes at all
                overall_status = "ALERT: FALL DETECTED"
                overall_color = (0, 0, 255) # Gray

            for person_info in detected_persons_info: # Draw for persons processed by MediaPipe
                x1, y1, x2, y2 = person_info["bbox_yolo_orig"]
                label = "FALL" if person_info["is_fall"] else "PERSON"
                color = (0, 0, 255) if person_info["is_fall"] else (0, 255, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(display_frame, overall_status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, overall_color, 2, cv2.LINE_AA)
            out.write(display_frame) # Write the frame with drawings

        # After loop finishes
        if total_frames > 0 and frames_processed >= total_frames:
            progress_bar.progress(100) # Ensure it hits 100%
            status_text.text("Processing Complete!")
        elif total_frames == 0 and frames_processed > 0: # For streams or unknown length
            status_text.text(f"Processing Complete! {frames_processed} frames processed.")
        elif frames_processed < total_frames :
            status_text.text(f"Processing ended prematurely after {frames_processed}/{total_frames} frames.")
        else: # No frames processed or other edge cases
            status_text.text("Processing Finished.")

        return True

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print full traceback for debugging
        return False
    finally:
        if cap:
            cap.release()
        if out:
            out.release()

# --- Streamlit UI ---
# st.set_page_config(layout="wide", page_title="Human Fall Detection")

st.title("🏃‍♂️ Human Fall Detection using YOLO & MediaPipe Pose 🤸‍♀️")
st.markdown("""
Upload a video file to detect falls. The system uses YOLOv8 for person detection
and MediaPipe Pose for analyzing posture to determine if a fall has occurred.
""")

uploaded_file = st.file_uploader("Choose a video file (.mp4, .avi, .mov, .mkv)", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    temp_input_video_path = None
    temp_output_video_path = None
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tfile:
            tfile.write(uploaded_file.read())
            temp_input_video_path = tfile.name

        st.video(temp_input_video_path) # Show the uploaded video

        if st.button("🕵️ Process Video for Fall Detection"):
            # Create a temporary path for the output video
            with tempfile.NamedTemporaryFile(delete=False, suffix='_output.mp4') as outfile: # Ensure .mp4 suffix
                temp_output_video_path = outfile.name

            st.info("Processing video... This may take a few moments depending on the video length.")
            processing_placeholder = st.empty() # For dynamic messages like progress bar and status

            start_time = time.time()

            # The process_video function will now update progress within Streamlit context
            success = process_video(temp_input_video_path, temp_output_video_path)

            end_time = time.time()
            processing_time = end_time - start_time

            if success:
                st.success(f"Video processed successfully in {processing_time:.2f} seconds!")
                # st.subheader("Processed Video Output:")

                if os.path.exists(temp_output_video_path) and os.path.getsize(temp_output_video_path) > 0:
                    try:
                        with open(temp_output_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        # st.video(video_bytes)  # Try displaying the video bytes

                        # --- Download Button ---
                        with open(temp_output_video_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Processed Video",
                                data=f.read(),
                                file_name="processed_video.mp4",
                                mime="video/mp4",
                            )
                        # --- End Download Button ---

                    except Exception as e:
                        st.error(f"Error displaying or providing download for processed video: {e}")
                else:
                    st.error("Output video file was not created, is empty, or is corrupted. "
                            "Please check the console for errors from OpenCV/FFmpeg if any, "
                            "or try a different video file/format.")
            else:
                st.error("Video processing failed. Please check the error messages above or try another video.")

    finally: # Ensure temporary files are cleaned up
        if temp_input_video_path and os.path.exists(temp_input_video_path):
            try:
                os.remove(temp_input_video_path)
            except Exception as e:
                st.warning(f"Could not remove temporary input file: {temp_input_video_path}. Error: {e}")
        if temp_output_video_path and os.path.exists(temp_output_video_path):
            try:
                os.remove(temp_output_video_path) # Clean up output file after displaying
            except Exception as e:
                st.warning(f"Could not remove temporary output file: {temp_output_video_path}. Error: {e}")
else:
    st.info("Upload a video file to begin.")

st.markdown("---")