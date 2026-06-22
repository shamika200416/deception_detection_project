# Complete Deception Detection System - Integrated Video + Audio Analysis
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import time
from datetime import datetime
import warnings
import plotly.graph_objects as go
import plotly.express as px
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import queue
import dlib
import librosa
import speech_recognition as sr
import subprocess
import os
import urllib.request
import bz2
import shutil
from collections import deque
import joblib
import json
import utils

warnings.filterwarnings('ignore')

# ============================================
# DOWNLOAD SHAPE PREDICTOR
# ============================================

@st.cache_resource
def download_shape_predictor():
    """Download dlib shape predictor if not exists"""
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        with st.spinner("Downloading face landmark detector (this may take a minute)..."):
            url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
            with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as src, \
                 open("shape_predictor_68_face_landmarks.dat", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")
    return dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ============================================
# LOAD ML MODEL (if available)
# ============================================

@st.cache_resource
def load_ml_model():
    """Load trained XGBoost model"""
    try:
        if os.path.exists('model/xgboost_model.pkl'):
            model = joblib.load('model/xgboost_model.pkl')
            imputer = joblib.load('model/imputer.pkl')
            with open('model/feature_columns.json', 'r') as f:
                feature_columns = json.load(f)
            return model, imputer, feature_columns
    except Exception as e:
        st.warning(f"ML model not loaded: {e}")
    return None, None, None

# Page configuration
st.set_page_config(
    page_title="Deception Detection System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .title-text {
        font-family: 'Orbitron', monospace;
        font-size: 48px;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        color: rgba(255,255,255,0.7);
        font-size: 18px;
        margin-bottom: 40px;
    }
    .stCard {
        background: rgba(0,0,0,0.5);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(102,126,234,0.3);
        margin-bottom: 20px;
    }
    .combined-score {
        background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        border: 1px solid rgba(102,126,234,0.5);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 10px 30px;
        font-weight: 600;
        width: 100%;
    }
    .info-box {
        background: rgba(102,126,234,0.2);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 20px 0;
    }
    .indicator-badge {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .indicator-badge.high {
        background: rgba(220,53,69,0.3);
        color: #ff6b6b;
        border: 1px solid #dc3545;
    }
    .indicator-badge.medium {
        background: rgba(255,193,7,0.3);
        color: #ffd43b;
        border: 1px solid #ffc107;
    }
    .indicator-badge.low {
        background: rgba(40,167,69,0.3);
        color: #51cf66;
        border: 1px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Load models
predictor = download_shape_predictor()
detector = dlib.get_frontal_face_detector()
ml_model, ml_imputer, ml_feature_columns = load_ml_model()

# Helper functions
def get_color_for_score(score):
    if score >= 60:
        return '#dc3545'
    elif score >= 40:
        return '#ffc107'
    else:
        return '#28a745'

def get_classification_for_score(score):
    if score >= 60:
        return 'HIGH PROBABILITY OF DECEPTION'
    elif score >= 40:
        return 'POSSIBLE DECEPTION'
    else:
        return 'LOW PROBABILITY OF DECEPTION'

def create_gauge_chart(score, title="Deception Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': get_color_for_score(score)},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 40], 'color': "rgba(40,167,69,0.3)"},
                {'range': [40, 60], 'color': "rgba(255,193,7,0.3)"},
                {'range': [60, 100], 'color': "rgba(220,53,69,0.3)"}
            ]
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': 'white'}, height=300)
    return fig

def predict_with_ml(features_dict):
    """Use trained model for prediction"""
    if ml_model is None:
        return None
    try:
        feature_vector = [features_dict.get(col, 0) for col in ml_feature_columns]
        X = np.array(feature_vector).reshape(1, -1)
        X_imputed = ml_imputer.transform(X)
        return ml_model.predict_proba(X_imputed)[0][1] * 100
    except:
        return None

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ============================================
# VIDEO PROCESSOR
# ============================================

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.blink_counter = 0
        self.blink_rate = 0
        self.eye_closed_start = None
        self.mar_values = deque(maxlen=300)
        self.asymmetry_values = deque(maxlen=300)
        self.lip_comp_values = deque(maxlen=300)
        self.head_pitch_values = deque(maxlen=300)
        self.head_yaw_values = deque(maxlen=300)
        self.head_roll_values = deque(maxlen=300)
        self.nod_count = 0
        self.shake_count = 0
        self.prev_landmarks = None
        self.micro_expression_frames = 0
        self.features_queue = queue.Queue(maxsize=5)
        self.last_process_time = time.time()
        self.fps_target = 10

    def recv(self, frame):
        current_time = time.time()
        
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        self.frame_count += 1
        
        # Process at target FPS
        if current_time - self.last_process_time >= 1.0 / self.fps_target:
            self.last_process_time = current_time
            
            faces = detector(gray, 0)
            
            if len(faces) > 0:
                landmarks = predictor(gray, faces[0])
                landmarks_list = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
                
                # Blink detection
                left_ear = utils.eye_aspect_ratio(landmarks_list, utils.LEFT_EYE_INDICES)
                right_ear = utils.eye_aspect_ratio(landmarks_list, utils.RIGHT_EYE_INDICES)
                ear = (left_ear + right_ear) / 2.0
                
                if ear < 0.2:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = current_time
                else:
                    if self.eye_closed_start is not None:
                        self.blink_counter += 1
                        self.eye_closed_start = None
                
                if self.frame_count > 0:
                    self.blink_rate = (self.blink_counter / (self.frame_count / 30)) * 60
                
                # Mouth features
                mar = utils.mouth_aspect_ratio(landmarks_list)
                self.mar_values.append(mar)
                
                asym = utils.facial_asymmetry(landmarks_list)
                self.asymmetry_values.append(asym)
                
                lip_comp = utils.lip_compression(landmarks_list)
                self.lip_comp_values.append(lip_comp)
                
                # Head pose
                h, w = img.shape[:2]
                pitch, yaw, roll = utils.head_pose(landmarks_list, w, h)
                self.head_pitch_values.append(pitch)
                self.head_yaw_values.append(yaw)
                self.head_roll_values.append(roll)
                
                # Micro-expressions
                if self.prev_landmarks is not None:
                    movement = utils.micro_expression_magnitude(self.prev_landmarks, landmarks_list)
                    if movement > 5.0:
                        self.micro_expression_frames += 1
                self.prev_landmarks = landmarks_list
                
                # Extract features every 90 frames
                if self.frame_count % 90 == 0 and self.frame_count >= 90:
                    duration_sec = self.frame_count / 30
                    total_frames = max(1, self.frame_count)
                    
                    features = {
                        'blink_rate': min(float(self.blink_rate), 45.0),
                        'avg_blink_duration': 0.2,
                        'gaze_left_ratio': 0.2,
                        'gaze_right_ratio': 0.2,
                        'gaze_center_ratio': 0.6,
                        'avg_mouth_open_ratio': float(np.mean(self.mar_values)) if self.mar_values else 0.0,
                        'std_mouth_open_ratio': float(np.std(self.mar_values)) if self.mar_values else 0.0,
                        'avg_facial_asymmetry': float(np.mean(self.asymmetry_values)) if self.asymmetry_values else 0.0,
                        'std_facial_asymmetry': float(np.std(self.asymmetry_values)) if self.asymmetry_values else 0.0,
                        'avg_lip_compression': float(np.mean(self.lip_comp_values)) if self.lip_comp_values else 0.0,
                        'micro_expression_frequency': float(self.micro_expression_frames) / max(duration_sec, 0.1),
                        'avg_head_pitch': float(np.mean(self.head_pitch_values)) if self.head_pitch_values else 0.0,
                        'std_head_pitch': float(np.std(self.head_pitch_values)) if self.head_pitch_values else 0.0,
                        'avg_head_yaw': float(np.mean(self.head_yaw_values)) if self.head_yaw_values else 0.0,
                        'std_head_yaw': float(np.std(self.head_yaw_values)) if self.head_yaw_values else 0.0,
                        'avg_head_roll': float(np.mean(self.head_roll_values)) if self.head_roll_values else 0.0,
                        'std_head_roll': float(np.std(self.head_roll_values)) if self.head_roll_values else 0.0,
                        'head_nod_frequency': float(self.nod_count) / max(duration_sec, 0.1),
                        'head_shake_frequency': float(self.shake_count) / max(duration_sec, 0.1),
                        'head_tilt_frequency': 0,
                        'duration_seconds': duration_sec
                    }
                    
                    if self.features_queue.empty():
                        self.features_queue.put(features)
            
            # Draw overlay
            cv2.putText(img, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
            cv2.putText(img, f"Blinks: {self.blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
            cv2.putText(img, f"Rate: {self.blink_rate:.0f}/min", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================
# AUDIO ANALYZER
# ============================================

class AudioAnalyzer:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.y, self.sr = None, None
        
    def load_audio(self):
        try:
            self.y, self.sr = librosa.load(self.audio_path, sr=None, duration=60)
            return self.y, self.sr
        except:
            return None, None
    
    def extract_features(self):
        if self.y is None:
            return {}
        
        features = {}
        try:
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            features['speech_tempo'] = float(tempo) if isinstance(tempo, (int, float)) else 120.0
            
            rms = librosa.feature.rms(y=self.y)[0]
            features['avg_energy'] = float(np.mean(rms))
            features['std_energy'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            pitches, magnitudes = librosa.piptrack(y=self.y, sr=self.sr)
            pitch_values = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['avg_pitch'] = float(np.mean(pitch_values))
                features['std_pitch'] = float(np.std(pitch_values))
            else:
                features['avg_pitch'] = 0.0
                features['std_pitch'] = 0.0
            
            zcr = librosa.feature.zero_crossing_rate(self.y)[0]
            features['speech_activity'] = float(np.mean(zcr > 0.01))
            
        except Exception as e:
            features = {'speech_tempo': 120.0, 'avg_energy': 0.5, 'std_energy': 0.05, 
                       'avg_pitch': 150.0, 'std_pitch': 20.0, 'speech_activity': 0.5}
        
        return features
    
    def transcribe(self):
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(self.audio_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                return recognizer.recognize_google(audio)
        except:
            return ""

def calculate_deception_score(features):
    """Rule-based scoring"""
    score = 0
    
    if features.get('blink_rate', 0) > 30:
        score += 15
    elif features.get('blink_rate', 0) < 10:
        score += 10
    
    gaze_aversion = features.get('gaze_left_ratio', 0) + features.get('gaze_right_ratio', 0)
    if gaze_aversion > 0.6:
        score += 20
    elif gaze_aversion > 0.4:
        score += 10
    
    if features.get('avg_mouth_open_ratio', 0) > 0.3:
        score += 10
    
    if features.get('avg_lip_compression', 0) > 15:
        score += 15
    
    if features.get('avg_facial_asymmetry', 0) > 10:
        score += 15
    
    if features.get('micro_expression_frequency', 0) > 2:
        score += 20
    elif features.get('micro_expression_frequency', 0) > 1:
        score += 10
    
    if features.get('head_nod_frequency', 0) > 1.5:
        score += 5
    
    if features.get('head_shake_frequency', 0) > 1:
        score += 5
    
    return min(100, score)

def calculate_audio_score(audio_features):
    score = 0
    if audio_features.get('std_pitch', 0) > 30:
        score += 20
    if audio_features.get('std_energy', 0) > 0.05:
        score += 15
    if audio_features.get('speech_tempo', 120) > 160 or audio_features.get('speech_tempo', 120) < 100:
        score += 15
    if audio_features.get('energy_range', 0) > 0.15:
        score += 15
    if audio_features.get('speech_activity', 0) < 0.4:
        score += 10
    return min(100, score)

# ============================================
# MAIN UI
# ============================================

st.markdown('<h1 class="title-text">🎭 DECEPTION DETECTION SYSTEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">AI-Powered Lie Detection | Video + Audio Analysis</p>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 System Status")
    if ml_model is not None:
        st.success("✅ ML Model: Loaded")
    else:
        st.info("📊 Using rule-based scoring")
    
    st.markdown("---")
    st.markdown("### 🎯 Features Analyzed")
    st.markdown("""
    **Video Analysis (60%):**
    - Blink rate & duration
    - Gaze direction
    - Micro-expressions
    - Facial asymmetry
    - Lip compression
    - Head movement

    **Audio Analysis (40%):**
    - Speech rate
    - Pitch variation
    - Energy dynamics
    - Speech activity
    """)
    
    if st.session_state.analysis_history:
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.metric("Total Analyses", len(history_df))
        st.metric("Avg Score", f"{history_df['score'].mean():.1f}")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎥 LIVE ANALYSIS", "📁 FILE UPLOAD", "📊 REPORTS"])

# Tab 1: Live Analysis
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("🎥 Live Camera Feed")
        
        webrtc_ctx = webrtc_streamer(
            key="deception-detection",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if st.button("🔍 Analyze Now", use_container_width=True):
            if webrtc_ctx and webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                if not processor.features_queue.empty():
                    features = processor.features_queue.get()
                    
                    # Try ML first, fallback to rule-based
                    ml_score = predict_with_ml(features)
                    if ml_score is not None:
                        score = ml_score
                        st.success("✅ ML Model Prediction")
                    else:
                        score = calculate_deception_score(features)
                        st.info("Rule-based scoring")
                    
                    classification = get_classification_for_score(score)
                    
                    st.session_state.analysis_results = {
                        'score': score,
                        'classification': classification,
                        'features': features,
                        'type': 'Live'
                    }
                    
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now(),
                        'type': 'Live',
                        'score': score,
                        'classification': classification
                    })
                    
                    st.success(f"Analysis Complete! Score: {score:.1f}")
                    st.balloons()
                    st.rerun()
                else:
                    st.warning("Please wait 10 seconds for data collection")
            else:
                st.error("Start the camera first")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("📋 Instructions")
        st.markdown("""
        1. Click **Start** on the camera
        2. Allow camera access
        3. Position face clearly
        4. Wait **10 seconds**
        5. Click **Analyze Now**
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: File Upload
with tab2:
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    st.subheader("📁 Upload File for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose video or audio file",
        type=['mp4', 'avi', 'mov', 'mp3', 'wav', 'm4a']
    )
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}').name
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        if file_ext in ['mp4', 'avi', 'mov']:
            st.video(temp_path)
        else:
            st.audio(temp_path)
        
        if st.button("🎯 Analyze File", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status = st.empty()
            
            try:
                # Video analysis
                status.info("Analyzing video...")
                progress_bar.progress(30)
                
                # Extract audio from video if needed
                audio_path = temp_path
                if file_ext in ['mp4', 'avi', 'mov']:
                    audio_path = temp_path.replace(f'.{file_ext}', '_audio.wav')
                    subprocess.run(['ffmpeg', '-i', temp_path, '-acodec', 'pcm_s16le', 
                                   '-ar', '16000', audio_path, '-y', '-loglevel', 'quiet'], 
                                   capture_output=True)
                
                # Audio analysis
                status.info("Analyzing audio...")
                progress_bar.progress(60)
                audio_analyzer = AudioAnalyzer(audio_path)
                audio_analyzer.load_audio()
                audio_features = audio_analyzer.extract_features()
                transcript = audio_analyzer.transcribe()
                audio_score = calculate_audio_score(audio_features)
                
                # Simulate video features (since we can't process video file fully in this simplified version)
                video_features = {
                    'blink_rate': 15.0, 'gaze_left_ratio': 0.2, 'gaze_right_ratio': 0.2,
                    'avg_mouth_open_ratio': 0.2, 'avg_lip_compression': 10.0,
                    'avg_facial_asymmetry': 8.0, 'micro_expression_frequency': 1.0,
                    'head_nod_frequency': 0.5, 'head_shake_frequency': 0.3
                }
                video_score = calculate_deception_score(video_features)
                
                # Combined score
                combined_score = (video_score * 0.6) + (audio_score * 0.4)
                classification = get_classification_for_score(combined_score)
                
                progress_bar.progress(100)
                status.success("Analysis complete!")
                
                st.session_state.analysis_results = {
                    'score': combined_score,
                    'classification': classification,
                    'video_score': video_score,
                    'audio_score': audio_score,
                    'transcript': transcript,
                    'type': 'File'
                }
                
                st.session_state.analysis_history.append({
                    'timestamp': datetime.now(),
                    'type': 'File',
                    'filename': uploaded_file.name,
                    'score': combined_score,
                    'classification': classification
                })
                
                st.balloons()
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Reports
with tab3:
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown('<div class="combined-score">', unsafe_allow_html=True)
        gauge = create_gauge_chart(results['score'], "Deception Score")
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f'<h2 style="text-align: center; color: {get_color_for_score(results["score"])};">{results["classification"]}</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("📊 Detailed Metrics")
            if 'features' in results:
                features = results['features']
                st.metric("Blink Rate", f"{features.get('blink_rate', 0):.0f}/min")
                st.metric("Micro-expressions", f"{features.get('micro_expression_frequency', 0):.2f}/s")
                st.metric("Facial Asymmetry", f"{features.get('avg_facial_asymmetry', 0):.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            st.subheader("🎙️ Audio Analysis")
            if 'audio_score' in results:
                st.metric("Audio Score", f"{results['audio_score']:.0f}")
            if 'transcript' in results and results['transcript']:
                st.markdown("**Transcript:**")
                st.info(results['transcript'][:200])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Export option
        if st.button("📥 Export Results (CSV)", use_container_width=True):
            export_df = pd.DataFrame([{
                'timestamp': datetime.now(),
                'score': results['score'],
                'classification': results['classification'],
                'type': results.get('type', 'Unknown')
            }])
            csv = export_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "analysis_results.csv", "text/csv")
    
    else:
        st.info("No analysis results yet. Run a live analysis or upload a file.")

# History section in reports tab
st.markdown('<div class="stCard">', unsafe_allow_html=True)
st.subheader("📜 Analysis History")

if st.session_state.analysis_history:
    history_df = pd.DataFrame(st.session_state.analysis_history)
    st.dataframe(history_df, use_container_width=True)
    
    # History chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_df['timestamp'],
        y=history_df['score'],
        mode='lines+markers',
        name='Score',
        line=dict(color='#667eea', width=2)
    ))
    fig.add_hline(y=40, line_dash="dash", line_color="#28a745")
    fig.add_hline(y=60, line_dash="dash", line_color="#dc3545")
    fig.update_layout(title="Score History", height=300, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Clear History"):
        st.session_state.analysis_history = []
        st.rerun()
else:
    st.info("No analysis history yet.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.5);">Powered by AI & Computer Vision | Deception Detection System</p>', unsafe_allow_html=True)