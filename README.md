<div align="center">
  <img src="resources/sensesign-logo.png" alt="SenseSign Logo" width="300" />
</div>

<p align="center">
  <strong>SenseSign: Real-Time ASL Translation & Blind Mobility Assistance System</strong>
</p>

## 🌟 Overview
SenseSign is a dual-purpose accessibility tool executing locally over a low-latency UDP stream. 
It combines **Machine Learning for Sign Language Recognition (ASL)** and **IoT LiDAR integration for spatial awareness** to assist visually and hearing-impaired users. A Raspberry Pi acts as the sensory node, wearing a camera and a TF-Luna LiDAR sensor, streaming real-time environment data wirelessly to a MacBook base-station which processes computer vision and spatial alerts.

## 🚀 Key Features

* **ASL Translation Mode**: Leverages **Google MediaPipe** to interpret ASL hand signs through the video feed, speaking translated phrases out loud to ease conversational barriers.
* **Blind Mobility Mode**: Utilizes a TF-Luna LiDAR to measure absolute distances with acute precision. Actively outputs **dynamic audio alerts** generated with variable pitch and frequency depending on proximity to obstacles (e.g. high-pitched rapid beeps for critical proximity and low-pitched slow pulses for safe zones), acting as an advanced spatial sonar. 
* **Multimodal Visual Q&A (Llama Vision & Groq Whisper)**: Visually impaired users can press an on-screen button or a keyboard trigger to speak a question (e.g., "What is in front of me?"). The system captures their voice, transcribes it via Whisper, takes a snapshot of the camera feed, and queries a Vision AI model for an immediate audio response! 

## 🛠 Prerequisites

### Raspberry Pi (Sensory Node)
* Raspberry Pi (3/4/5) with Pi Camera or USB Webcam
* TF-Luna LiDAR wired to `/dev/serial0`
* Python 3.9+ 

### MacBook (Base Station)
* macOS
* Python 3.9+ (Python 3.10-3.11 recommended for MediaPipe stability)
* Functional audio speakers/headphones

## 🖥 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Hamdan772/SenseSign
   cd SenseSign
   ```

2. **API Keys Configuration**
   For the intelligent Vision Q&A assistant to function, create a `.env` file in the root of the project on your MacBook and insert your Groq API key:
   ```env
   GROQ_API_KEY=gsk_YOUR_KEY_HERE
   ```

3. **Install Dependencies**
   The shell scripts automatically initialize a Virtual Environment (`venv`) and manage package installations dynamically. Ensure your internet connection is active during the first run.

## ⚙️ Running The System

* **Step 1: Start the Receiver (MacBook Base Station)**
   Launch the processing server. It will initialize the UI and wait for the incoming UDP stream from the Pi.
   ```bash
   ./run_mac.sh
   # (Optionally run manually: source venv/bin/activate && python3 udp_server.py)
   ```

* **Step 2: Start the Sender (Raspberry Pi Node)**
   Replace the placeholder IP with the specific local IP address of your MacBook.
   ```bash
   ./run_pi.sh 192.168.1.100
   ```

## 🎮 Interface Controls

The processing UI offers interactive overlays:
* **Mode Toggle Button**: Click the UI overlay to seamlessly swap between `ASL` and `BLIND` interaction paradigms.
* **Q&A Trigger Button**: Available in Blind Mode. Clicking it (or hitting `v` on your keyboard) temporarily mutes LiDAR sonar, captures a 4-second audio inquiry from your microphone, and reads the Vision response aloud.
* **HUD Stats**: Always tracks FPS, connectivity, and distance readings dynamically.

## 🤝 Note
We aggressively stripped the obsolete Cloud and Website hosting features in favor of a strictly local, high-performance UDP infrastructure. We recommend keeping `run_mac.sh` pointing to your deployment directory.
