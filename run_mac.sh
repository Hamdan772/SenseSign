#!/bin/bash

# This script runs on the MacBook to receive the video stream.
# Optimized for /Users/hamdannishad/Desktop/SenseSign

echo "Starting UDP video receiver on MacBook..."

# Ensure we are in the correct directory
cd /Users/hamdannishad/Desktop/SenseSign || exit 1

# Ensure venv exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
# Force upgrade pip and install correct macOS tensorflow
if command -v brew >/dev/null 2>&1; then
    brew install portaudio || true
fi

python3 -m pip install --upgrade pip
python3 -m pip install -q -r requirements.txt pyautogui
# For Apple M-series chips, force the macOS tensorflow backend if standard is broken
python3 -m pip install --upgrade tensorflow-macos

python3 udp_server.py
