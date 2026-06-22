import numpy as np
import cv2
import dlib

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

# 3D model points
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
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

def micro_expression_magnitude(prev_landmarks, curr_landmarks):
    if prev_landmarks is None:
        return 0
    movements = [np.linalg.norm(np.array(p) - np.array(c)) for p, c in zip(prev_landmarks, curr_landmarks)]
    return np.mean(movements)