# 🎭 Deception Detection System

AI-powered lie detection using facial analysis with dlib and audio analysis with librosa.

## Features

- **Real-time Video Analysis**: Live camera feed with 68-point facial landmark detection
- **Blink Rate Analysis**: Detects stress indicators through eye blink patterns
- **Gaze Direction Tracking**: Monitors eye contact and gaze aversion
- **Micro-expression Detection**: Captures brief facial expressions
- **Head Movement Analysis**: Tracks nodding, shaking, and tilting
- **Audio Analysis**: Speech tempo, pitch variation, and energy dynamics
- **Combined Scoring**: 60% video + 40% audio weighted deception score

## Deployment on Streamlit Cloud

1. Push this code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy!

## Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/deception-detection

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev \
    libx11-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
    libjpeg-dev libpng-dev libtiff-dev ffmpeg libgl1-mesa-glx

# Run the app
streamlit run app.py
