#!/bin/bash
# Install system dependencies for dlib
apt-get update
apt-get install -y cmake build-essential libopenblas-dev liblapack-dev libx11-dev
pip install dlib==19.24.2