import os
import cv2
import math
import numpy as np
import mediapipe as mp
import moviepy.editor as mp_editor
import librosa
from faster_whisper import WhisperModel
from groq import Groq
from dotenv import load_dotenv

# Load API Key
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Models
# 'base' is a good balance of speed/accuracy. Use 'tiny' if on an older laptop.
audio_model = WhisperModel("base", device="cpu", compute_type="int8")

# --- MediaPipe Configuration ---
mp_face_mesh = mp.solutions.face_mesh
# "refine_landmarks=True" is CRITICAL. It adds the 10 iris landmarks (468-477).
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices (The "Map" of the face)
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def extract_audio(video_path):
    """Extracts audio from video and saves as temp.wav"""
    audio_path = "temp_data/temp.wav"
    # Delete existing audio if it exists to avoid conflicts
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    video = mp_editor.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
    return audio_path

def transcribe_audio(audio_path):
    """Transcribes audio using Faster-Whisper"""
    segments, _ = audio_model.transcribe(audio_path)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

def analyze_audio_tone(audio_path):
    """
    Analyzes audio pitch to detect if the speaker is monotone or expressive.
    Returns: 'Monotone', 'Normal', or 'Dynamic'
    """
    try:
        y, sr = librosa.load(audio_path)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Select valid pitches (where magnitude > threshold)
        pitches = pitches[magnitudes > np.median(magnitudes)]
        pitches = pitches[pitches > 0] # Remove zeros
        
        if len(pitches) == 0:
            return "Uncertain"
            
        # Calculate variation (Standard Deviation)
        pitch_std = np.std(pitches)
        
        # Heuristic thresholds (tweak based on testing)
        if pitch_std < 20:
            return "Monotone (Robot-like)"
        elif pitch_std < 40:
            return "Normal (Conversational)"
        else:
            return "Dynamic (Expressive)"
            
    except Exception as e:
        return f"Error: {str(e)}"

def _get_landmark_point(landmarks, idx, width, height):
    """Helper to convert normalized coordinates to pixel coordinates"""
    point = landmarks[idx]
    return int(point.x * width), int(point.y * height)

def calculate_gaze_ratio(eye_points, iris_center):
    """
    Calculates the horizontal position of the iris.
    Returns a ratio: 0.0 (Left) --- 0.5 (Center) --- 1.0 (Right)
    """
    # Distance from Left Corner to Iris Center
    dist_left = math.hypot(iris_center[0] - eye_points[0][0], iris_center[1] - eye_points[0][1])
    # Distance from Right Corner to Iris Center
    dist_right = math.hypot(iris_center[0] - eye_points[1][0], iris_center[1] - eye_points[1][1])
    
    # Avoid division by zero
    total_dist = dist_left + dist_right
    if total_dist == 0:
        return 0.5
        
    ratio = dist_left / total_dist
    return ratio

def analyze_video_eye_contact(video_path):
    """
    Advanced Logic: Tracks Iris movement to detect eye contact.
    Returns: Score (0-100) and a path to a 'debug image' showing the tracking.
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    total_frames = 0
    center_gaze_frames = 0
    debug_image_path = "temp_data/debug_frame.jpg"
    saved_debug_image = False
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
            
        total_frames += 1
        # Skip frames for speed (process every 3rd frame)
        if total_frames % 3 != 0:
            continue
            
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # --- Get Coordinates ---
            # Left Eye
            l_corner_1 = _get_landmark_point(landmarks, LEFT_EYE_CORNERS[0], width, height)
            l_corner_2 = _get_landmark_point(landmarks, LEFT_EYE_CORNERS[1], width, height)
            l_iris_center = _get_landmark_point(landmarks, 468, width, height)
            
            # Right Eye
            r_corner_1 = _get_landmark_point(landmarks, RIGHT_EYE_CORNERS[0], width, height)
            r_corner_2 = _get_landmark_point(landmarks, RIGHT_EYE_CORNERS[1], width, height)
            r_iris_center = _get_landmark_point(landmarks, 473, width, height)
            
            # --- Calculate Ratios ---
            left_ratio = calculate_gaze_ratio([l_corner_1, l_corner_2], l_iris_center)
            right_ratio = calculate_gaze_ratio([r_corner_1, r_corner_2], r_iris_center)
            avg_ratio = (left_ratio + right_ratio) / 2
            
            # --- Determine "Center" ---
            # A ratio between 0.42 and 0.58 usually means looking at the camera/screen
            if 0.42 < avg_ratio < 0.58:
                center_gaze_frames += 1
            
            # --- Save ONE frame for the UI (UPDATED VISUALS) ---
            if not saved_debug_image and total_frames > 20:
                # Left Eye
                cv2.circle(image, l_iris_center, 8, (0, 255, 255), -1) # Yellow Dot (Big)
                cv2.line(image, l_corner_1, l_corner_2, (0, 0, 255), 2) # Red Line
                
                # Right Eye
                cv2.circle(image, r_iris_center, 8, (0, 255, 255), -1) # Yellow Dot (Big)
                cv2.line(image, r_corner_1, r_corner_2, (0, 0, 255), 2) # Red Line
                
                # Add text label
                cv2.putText(image, f"Gaze: {avg_ratio:.2f}", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                cv2.imwrite(debug_image_path, image)
                saved_debug_image = True

    cap.release()
    
    # Calculate final score
    if total_frames == 0: return 0, None
    
    # Normalize: We processed 1/3rd of frames, so adjust total
    processed_frames = total_frames / 3
    score = (center_gaze_frames / processed_frames) * 100
    
    return min(score, 100), debug_image_path

def generate_feedback(transcript, eye_contact_score):
    """Sends data to Llama-3 via Groq for feedback"""
    prompt = f"""
    You are an expert Interview Coach. 
    Analyze this candidate's response.
    
    TRANSCRIPT: "{transcript}"
    EYE CONTACT SCORE: {eye_contact_score:.1f}% (Low means they looked away often).
    
    Give feedback in this format:
    1. **Executive Summary**: One sentence summary.
    2. **Strengths**: 2 bullet points.
    3. **Areas for Improvement**: 2 bullet points (mention eye contact if score < 70%).
    4. **Refined Answer**: A professional rewrite of their answer.
    """
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful coach."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
    )
    return chat_completion.choices[0].message.content