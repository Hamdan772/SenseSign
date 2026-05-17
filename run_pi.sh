#!/bin/bash

# This script runs on the Raspberry Pi to capture webcam feed and stream it.
# It requires the IP address of the MacBook as an argument.

if [ -z "$1" ]; then
    echo "Error: Missing target IP address."
    echo "Usage: ./run_pi.sh <MACBOOK_IP_ADDRESS>"
    echo "Example: ./run_pi.sh 192.168.1.100"
    exit 1
fi

MACBOOK_IP=$1
echo "Starting UDP video sender on Raspberry Pi, streaming to $MACBOOK_IP..."

# Use virtual environment if it exists, otherwise use system python
if [ -d "venv" ]; then
    source venv/bin/activate
    # Ensure dependencies are installed
    pip install -q opencv-python-headless numpy pyserial
fi

# Ensure serial port permissions are granted for LiDAR
if [ -e "/dev/serial0" ]; then
    sudo chmod 666 /dev/serial0 /dev/ttyS0 /dev/ttyAMA0 2>/dev/null || true
    # Also explicitly add readlink target just in case
    sudo chmod 666 $(readlink -f /dev/serial0) 2>/dev/null || true
fi

python3 udp_client.py "$MACBOOK_IP"
