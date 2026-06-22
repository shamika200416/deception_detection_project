import cv2
import numpy as np
import dlib
import os

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Download predictor if not exists
if not os.path.exists(predictor_path):
    import urllib.request
    import bz2
    import shutil
    print("Downloading shape predictor...")
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, "tmp.bz2")
    with bz2.open("tmp.bz2", 'rb') as src, open(predictor_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.remove("tmp.bz2")

predictor = dlib.shape_predictor(predictor_path)

# Constants
LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))
MOUTH_OUTER_INDICES = list(range(48, 60))
NOSE_TIP = 30
CHIN = 8
LEFT_EYE_CORNER = 36
RIGHT_EYE_CORNER = 45
LEFT_MOUTH = 48
RIGHT_MOUTH = 54
UPPER_LIP = 51
LOWER_LIP = 57

# 3D model points for head pose
HEAD_POSE_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

def get_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

def eye_aspect_ratio(landmarks, eye_indices):
    pts = np.array([landmarks[i] for i in eye_indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0

def mouth_aspect_ratio(landmarks):
    outer = np.array([landmarks[i] for i in MOUTH_OUTER_INDICES])
    if len(outer) > 9:
        A = np.linalg.norm(outer[3] - outer[9])
        B = np.linalg.norm(outer[0] - outer[6])
        return A / B if B > 0 else 0
    return 0

def lip_compression(landmarks):
    upper = np.array(landmarks[UPPER_LIP])
    lower = np.array(landmarks[LOWER_LIP])
    return np.linalg.norm(upper - lower)

def facial_asymmetry(landmarks):
    left_indices = [36, 37, 38, 48, 49, 50, 17, 18, 19]
    right_indices = [45, 46, 47, 54, 53, 52, 26, 25, 24]
    left_pts = np.array([landmarks[i] for i in left_indices if i < len(landmarks)])
    right_pts = np.array([landmarks[i] for i in right_indices if i < len(landmarks)])
    return np.linalg.norm(np.mean(left_pts, axis=0) - np.mean(right_pts, axis=0))

def gaze_direction(landmarks, frame_width):
    left_eye = np.mean([landmarks[i] for i in LEFT_EYE_INDICES], axis=0)
    right_eye = np.mean([landmarks[i] for i in RIGHT_EYE_INDICES], axis=0)
    eye_center = (left_eye + right_eye) / 2
    nose = np.array(landmarks[NOSE_TIP])
    offset = nose[0] - eye_center[0]
    threshold = frame_width * 0.05
    if offset > threshold:
        return "right"
    elif offset < -threshold:
        return "left"
    return "center"

def head_pose(landmarks, w, h):
    image_pts = np.array([
        landmarks[NOSE_TIP], landmarks[CHIN],
        landmarks[LEFT_EYE_CORNER], landmarks[RIGHT_EYE_CORNER],
        landmarks[LEFT_MOUTH], landmarks[RIGHT_MOUTH]
    ], dtype=np.float64)
    
    focal = w
    center = (w/2, h/2)
    cam_mat = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4,1))
    
    success, rvec, _ = cv2.solvePnP(HEAD_POSE_POINTS, image_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0, 0, 0
    
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]

def micro_expression_magnitude(prev_landmarks, curr_landmarks):
    if prev_landmarks is None:
        return 0
    movements = [np.linalg.norm(np.array(p) - np.array(c)) for p, c in zip(prev_landmarks, curr_landmarks)]
    return np.mean(movements)

def extract_features_from_video(video_path, fps_target=5, max_duration_sec=60):
    """Extract all features from a single video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if duration > max_duration_sec:
        cap.release()
        return None, duration
    
    frame_interval = max(1, int(fps / fps_target))
    
    # Initialize trackers
    blinks = 0
    blink_durations = []
    eye_closed_start = None
    gaze_durations = {'left': 0, 'right': 0, 'center': 0}
    frame_count = 0
    mar_values = []
    asymmetry_values = []
    lip_comp_values = []
    head_pitch_values = []
    head_roll_values = []
    head_yaw_values = []
    nod_count = 0
    shake_count = 0
    head_tilt_count = 0
    prev_head_pitch = None
    prev_head_yaw = None
    prev_head_roll = None
    micro_expression_frames = 0
    prev_landmarks = None
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue
        
        processed_frames += 1
        h, w, _ = frame.shape
        
        landmarks = get_landmarks(frame)
        if landmarks is None:
            continue
        
        # Eye blink detection
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_INDICES)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_INDICES)
        ear = (left_ear + right_ear) / 2.0
        if ear < 0.2:
            if eye_closed_start is None:
                eye_closed_start = frame_count
        else:
            if eye_closed_start is not None:
                blink_duration = (frame_count - eye_closed_start) / fps
                blink_durations.append(blink_duration)
                blinks += 1
                eye_closed_start = None
        
        # Gaze
        gaze = gaze_direction(landmarks, w)
        gaze_durations[gaze] += 1
        
        # Mouth and facial features
        mar_values.append(mouth_aspect_ratio(landmarks))
        asymmetry_values.append(facial_asymmetry(landmarks))
        lip_comp_values.append(lip_compression(landmarks))
        
        # Head pose
        pitch, yaw, roll = head_pose(landmarks, w, h)
        head_pitch_values.append(pitch)
        head_roll_values.append(roll)
        head_yaw_values.append(yaw)
        
        if prev_head_pitch is not None:
            if abs(prev_head_pitch - pitch) > 5:
                nod_count += 1
            if prev_head_yaw is not None and abs(yaw - prev_head_yaw) > 10:
                shake_count += 1
            if prev_head_roll is not None and abs(roll - prev_head_roll) > 8:
                head_tilt_count += 1
        
        prev_head_pitch, prev_head_yaw, prev_head_roll = pitch, yaw, roll
        
        # Micro-expressions
        if prev_landmarks is not None:
            movement = micro_expression_magnitude(prev_landmarks, landmarks)
            if movement > 5.0:
                micro_expression_frames += 1
        prev_landmarks = landmarks
    
    cap.release()
    
    total_frames = processed_frames
    if total_frames == 0:
        return None, duration
    
    duration_sec = frame_count / fps
    
    features = {
        'blink_rate': blinks / duration_sec * 60,
        'avg_blink_duration': np.mean(blink_durations) if blink_durations else 0,
        'gaze_left_ratio': gaze_durations['left'] / total_frames,
        'gaze_right_ratio': gaze_durations['right'] / total_frames,
        'gaze_center_ratio': gaze_durations['center'] / total_frames,
        'avg_mouth_open_ratio': np.mean(mar_values) if mar_values else 0,
        'std_mouth_open_ratio': np.std(mar_values) if mar_values else 0,
        'avg_facial_asymmetry': np.mean(asymmetry_values) if asymmetry_values else 0,
        'std_facial_asymmetry': np.std(asymmetry_values) if asymmetry_values else 0,
        'avg_lip_compression': np.mean(lip_comp_values) if lip_comp_values else 0,
        'micro_expression_frequency': micro_expression_frames / duration_sec,
        'avg_head_pitch': np.mean(head_pitch_values) if head_pitch_values else 0,
        'std_head_pitch': np.std(head_pitch_values) if head_pitch_values else 0,
        'avg_head_roll': np.mean(head_roll_values) if head_roll_values else 0,
        'std_head_roll': np.std(head_roll_values) if head_roll_values else 0,
        'avg_head_yaw': np.mean(head_yaw_values) if head_yaw_values else 0,
        'std_head_yaw': np.std(head_yaw_values) if head_yaw_values else 0,
        'head_nod_frequency': nod_count / duration_sec,
        'head_shake_frequency': shake_count / duration_sec,
        'head_tilt_frequency': head_tilt_count / duration_sec,
        'duration_seconds': duration_sec
    }
    
    return features, duration_sec