import os
os.environ['GLOG_minloglevel'] = '2'
import cv2 as cv
import socket
import pickle
import numpy as np
import copy
import csv
import os
import datetime
import time
import subprocess
import threading
import sys
import base64
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import mediapipe as mp
from dotenv import load_dotenv

try:
    from groq import Groq
except ImportError:
    pass

from slr.utils.args import get_args
from slr.utils.cvfpscalc import CvFpsCalc
from slr.utils.landmarks import draw_landmarks
from slr.utils.draw_debug import draw_bounding_rect, draw_hand_label
from slr.utils.pre_process import calc_bounding_rect, calc_landmark_list, pre_process_landmark
from slr.model.classifier import KeyPointClassifier

def speak_letter(letter):
    """Speak a letter using macOS 'say' command."""
    def _speak():
        try:
            subprocess.run(["say", letter], check=False)
        except Exception:
            pass
    threading.Thread(target=_speak, daemon=True).start()

# Beep Logic Background Thread
class RadarBeeper:
    def __init__(self):
        self.dist_cm = -1
        self.active = False
        self.running = True
        self.muted = False
        threading.Thread(target=self._beep_loop, daemon=True).start()
        
    def _beep_loop(self):
        fs = 44100
        while self.running:
            if self.active and self.dist_cm > 0 and not self.muted:
                if self.dist_cm < 25:
                    # Critical Proximity Zone: ultra-rapid high pitch beep
                    delay = 0.05
                    duration = 0.05
                    freq = 1200
                elif self.dist_cm <= 200:
                    # Dynamic Warning Zone: map [25, 200] cm to [0.05, 0.4] seconds
                    fraction = (self.dist_cm - 25) / (200 - 25)
                    delay = 0.05 + fraction * 0.35
                    duration = 0.05 + fraction * 0.1
                    # Map frequency from 1200 Hz down to 600 Hz
                    freq = 1200 - fraction * 600
                else:
                    # Safe Zone: slow heartbeat lower pitch beep
                    delay = 1.0
                    duration = 0.2
                    freq = 400

                try:
                    # Generate a seamless beep using sounddevice
                    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
                    wave = 0.5 * np.sin(2 * np.pi * freq * t)
                    # Apply a simple envelope to prevent clicking audio artifacts
                    envelope = np.ones_like(wave)
                    fade_len = int(0.01 * fs)
                    if fade_len > len(t) // 2: 
                        fade_len = len(t) // 2
                    if fade_len > 0:
                        fade_in = np.linspace(0, 1, fade_len)
                        fade_out = np.linspace(1, 0, fade_len)
                        envelope[:fade_len] = fade_in
                        envelope[-fade_len:] = fade_out
                    
                    sd.play(wave * envelope, samplerate=fs)
                    sd.wait() # Wait for the duration to finish naturally
                    time.sleep(max(0, delay - duration))
                except Exception as e:
                    print(f"Audio Beep Error: {e}")
                    time.sleep(delay)
            else:
                time.sleep(0.1)

# Application Context State
app_state = {'mode': 'ASL', 'trigger_qa': False, 'qa_running': False}

def click_button(event, x, y, flags, param):
    """OpenCV Mouse Event for the on-screen UI button toggles."""
    if event == cv.EVENT_LBUTTONDOWN:
        # Check if click is inside the UI button
        if 40 <= x <= 440 and 40 <= y <= 140:
            param['mode'] = 'BLIND' if param['mode'] == 'ASL' else 'ASL'
            print(f"INFO: Switched to {param['mode']} Mode")
        elif 480 <= x <= 880 and 40 <= y <= 140 and param['mode'] == 'BLIND':
            param['trigger_qa'] = True

def draw_transparent_rect(img, top_left, bottom_right, color, alpha):
    """Utility to draw a semi-transparent rectangle for better UI."""
    overlay = img.copy()
    cv.rectangle(overlay, top_left, bottom_right, color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def handle_visual_qa(frame, beeper, state):
    """Voice QA Pipeline via Groq Whisper and Llama Vision APIs"""
    print("INFO: Initializing Voice Query...")
    beeper.muted = True
    try:
        # Load API key from dot environment or environment variables
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
        
        client = Groq(api_key=groq_api_key)
        
        # Audio capturing
        subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("INFO: Recording audio for 4 seconds...")
        fs = 44100
        duration = 4.0
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait() # Block until 4s is done
        
        # Save exact moment's audio locally to pass to Groq
        wav_write("query.wav", fs, recording)
        subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Step 2: Speech-to-Text via Whisper (STT)
        print("INFO: Calling Groq Whisper API...")
        with open("query.wav", "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=("query.wav", f.read()),
                model="whisper-large-v3-turbo"
            )
            
        query_text = transcription.text.strip()
        print(f"INFO: Recognized Speech: '{query_text}'")
        
        if not query_text:
            raise ValueError("No speech caught in 4 seconds.")

        # Step 3: Base64 Encode the exact momentary frame
        _, buffer = cv.imencode('.jpg', frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Step 4: Vision Execution
        print("INFO: Running Multimodal Vision inference...")
        sys_prompt = "You are a helpful, conversational physical assistant for a blind user. Answer very concisely and conversationally in one brief sentence. Do not use formatting, markdown, or lists—just raw text for Text-To-Speech execution."
        
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Discovered active Llama-4 Multimodal endpoint
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{sys_prompt}\n\nQuestion: {query_text}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        
        answer = completion.choices[0].message.content.strip()
        print(f"INFO: Vision Engine Returns: '{answer}'")
        
        # Step 5: Speak out to the user loudly
        subprocess.run(["say", answer])
        
    except Exception as e:
        print(f"ERROR in Voice QA Pipeline: {e}")
        subprocess.run(["say", "Sorry, I encountered an error processing your vision request."])
    finally:
        beeper.muted = False
        state['qa_running'] = False


def main():
    print("INFO: Initializing SenseSign Server")
    load_dotenv()
    args = get_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 5000))
    max_length = 65540

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    keypoint_classifier = KeyPointClassifier()
    with open("slr/model/label.csv", encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        labels = [row[0] for row in key_points]

    cv_fps = CvFpsCalc(buffer_len=10)
    
    last_letter = ""
    letter_start = None
    spoken = False
    
    beeper = RadarBeeper()
    
    cv.namedWindow("SenseSign", cv.WINDOW_NORMAL)
    cv.resizeWindow("SenseSign", 1280, 960) # Ensure window spawns large on laptop screens
    cv.setMouseCallback("SenseSign", click_button, param=app_state)

    print("INFO: Ready and listening for packets...")
    dist_cm = -1
    
    translation_history = []

    while True:
        try:
            data, address = sock.recvfrom(max_length)
            if len(data) < 100:
                frame_info = pickle.loads(data)
                if frame_info:
                    nums_of_packs = frame_info.get("packs", 1)
                    dist_cm = frame_info.get("distance", -1)
                    
                    buffer = b""
                    for i in range(nums_of_packs):
                        data, _ = sock.recvfrom(max_length)
                        buffer += data

                    frame_bytes = np.frombuffer(buffer, dtype=np.uint8)
                    image = cv.imdecode(frame_bytes, cv.IMREAD_COLOR)
                    if image is None: continue

                    fps = cv_fps.get()
                    key = cv.waitKey(1)
                    if key == 27: break
                    
                    image = cv.resize(image, (1280, 960))
                    image = cv.flip(image, 1)
                    debug_image = copy.deepcopy(image)
                    
                    beeper.dist_cm = dist_cm
                    beeper.active = (app_state['mode'] == 'BLIND')

                    if app_state['mode'] == 'ASL':
                        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)
                        
                        if results.multi_hand_landmarks:
                            for hl, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                                brect = calc_bounding_rect(debug_image, hl)
                                landmark_list = calc_landmark_list(debug_image, hl)
                                pre_processed = pre_process_landmark(landmark_list)
                                
                                sign_id, conf = keypoint_classifier(pre_processed)
                                if sign_id != 25:
                                    text = labels[sign_id]
                                    
                                    if text == last_letter:
                                        if letter_start and not spoken and (time.time() - letter_start >= 2.0):
                                            speak_letter(text)
                                            spoken = True
                                            translation_history.append(text)
                                            if len(translation_history) > 5:
                                                translation_history.pop(0)
                                    else:
                                        last_letter = text
                                        letter_start = time.time()
                                        spoken = False
                                    
                                    # Draw Live Detection Box
                                    # Add smooth anti-aliased lines and a cleaner background layout
                                    draw_transparent_rect(debug_image, (20, 170), (640, 320), (20, 20, 20), 0.75)
                                    cv.putText(debug_image, f"Sign: {text}", (40, 280), 
                                               cv.FONT_HERSHEY_DUPLEX, 2.6, (0, 255, 0), 4, cv.LINE_AA)
                                    cv.putText(debug_image, f"Conf: {conf*100:.1f}%", (420, 274), 
                                               cv.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2, cv.LINE_AA)
                                else:
                                    last_letter = ""
                                    
                                debug_image = draw_bounding_rect(debug_image, True, brect, outline_color=(100, 255, 100))
                                debug_image = draw_landmarks(debug_image, landmark_list)
                                debug_image = draw_hand_label(debug_image, brect, hd)

                        draw_transparent_rect(debug_image, (0, 840), (1280, 960), (15, 15, 15), 0.85)
                        cv.putText(debug_image, "History: " + " ".join(translation_history), 
                                   (40, 910), cv.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2, cv.LINE_AA)

                    elif app_state['mode'] == 'BLIND':
                        draw_transparent_rect(debug_image, (0, 0), (1280, 960), (10, 10, 30), 0.6)
                        
                        if dist_cm > 0:
                            if dist_cm < 25:
                                flash_color = (0, 0, 255) # Red
                                alert_msg = "CRITICAL PROXIMITY"
                            elif dist_cm <= 100:
                                flash_color = (0, 165, 255) # Orange
                                alert_msg = "OBSTACLE NEARING"
                            else:
                                flash_color = (0, 255, 0) # Green
                                alert_msg = "PATH CLEAR"

                            cv.rectangle(debug_image, (100, 300), (1180, 700), flash_color, 12)
                            cv.putText(debug_image, f"{dist_cm} cm", (420, 520), 
                                       cv.FONT_HERSHEY_DUPLEX, 5.0, flash_color, 8, cv.LINE_AA)
                            
                            # Center the alert msg depending on text size approximation
                            msg_x = 270 if alert_msg == "CRITICAL PROXIMITY" else (320 if alert_msg == "OBSTACLE NEARING" else 420)
                            cv.putText(debug_image, alert_msg, (msg_x, 630), 
                                       cv.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 2, cv.LINE_AA)

                        # Q&A UI Logic
                        qa_flash = int(time.time() * 4) % 2 == 0
                        if app_state['qa_running']:
                            btn_qa_color = (0, 0, 150) if qa_flash else (0, 0, 200) # Crimson flash (BGR)
                            qa_text = "LISTENING..."
                        else:
                            btn_qa_color = (200, 100, 0) # Cobalt Blue (BGR)
                            qa_text = "ASK Q&A [V]"
                            
                        cv.rectangle(debug_image, (480, 40), (880, 140), (200, 200, 200), 4)
                        draw_transparent_rect(debug_image, (480, 40), (880, 140), btn_qa_color, 0.8)
                        
                        # Adjust text centering based on text length
                        text_x = 530 if app_state['qa_running'] else 530
                        cv.putText(debug_image, qa_text, (text_x, 104), cv.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)
                        
                        # Handle Trigger execution (from UI click or 'v' key)
                        if (app_state.get('trigger_qa') or key == ord('v')) and not app_state.get('qa_running'):
                            app_state['trigger_qa'] = False
                            app_state['qa_running'] = True
                            
                            # Spin off standard UI execution thread, pass frame strictly
                            threading.Thread(
                                target=handle_visual_qa, 
                                args=(image.copy(), beeper, app_state), 
                                daemon=True
                            ).start()

                    # --- Global Render UI Elements ---
                    # 1. Floating Mode Toggle Button
                    if app_state['mode'] == 'ASL':
                        btn_color = (50, 50, 50)
                        outline_color = (200, 200, 200)
                    else:
                        btn_color = (30, 30, 30)
                        outline_color = (0, 215, 255) # Amber/Gold accent in BGR
                        
                    cv.rectangle(debug_image, (40, 40), (440, 140), outline_color, 4)
                    draw_transparent_rect(debug_image, (40, 40), (440, 140), btn_color, 0.8)
                    btn_text = "MODE: ASL" if app_state['mode'] == 'ASL' else "MODE: BLIND"
                    cv.putText(debug_image, btn_text, (90, 104), cv.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv.LINE_AA)

                    # 2. Top-Right Global Specs (System Telemetry Hub)
                    draw_transparent_rect(debug_image, (900, 40), (1240, 180), (10, 10, 10), 0.7)
                    cv.putText(debug_image, f"FPS: {fps:0.1f}" if type(fps) != str else f"FPS: {fps}", (920, 96), cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
                    
                    lidar_status = "Connected" if dist_cm > 0 else "Offline / Blocked"
                    lidar_color = (0, 255, 0) if dist_cm > 0 else (0, 0, 255)
                    cv.putText(debug_image, f"LiDAR: {lidar_status}", (920, 150), cv.FONT_HERSHEY_DUPLEX, 0.8, lidar_color, 2, cv.LINE_AA)
                    
                    cv.imshow("SenseSign", debug_image)

        except KeyboardInterrupt:
            print("\nINFO: Shutting down safely...")
            break
        except Exception as e:
            pass

    beeper.running = False
    sd.stop()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
