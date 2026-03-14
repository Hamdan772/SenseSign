"""
receiver.py

Pure-Python receiver + viewer for SenseSign.
Runs on the laptop alongside web_app.py. Displays the remote camera feed
in an OpenCV window with hand landmarks and detected letters drawn on screen.

Usage:
    python receiver.py
    python receiver.py --port 5000

The laptop must also be running web_app.py (the Flask server).
On the Pi, run: python sender.py --server http://LAPTOP_IP:5000
"""

import argparse
import time

import cv2 as cv
import numpy as np
import requests

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_hand(frame, hand, w, h):
    """Draw landmarks, bounding box, and label on the frame."""
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in hand["landmarks"]]

    # Connections
    for a, b in CONNECTIONS:
        cv.line(frame, pts[a], pts[b], (180, 120, 255), 2, cv.LINE_AA)

    # Keypoints
    for x, y in pts:
        cv.circle(frame, (x, y), 4, (255, 255, 255), -1, cv.LINE_AA)
        cv.circle(frame, (x, y), 4, (0, 0, 0), 1, cv.LINE_AA)

    # Bounding box
    bb = hand.get("bbox", [])
    if len(bb) == 4:
        pad = 12
        cv.rectangle(frame, (bb[0] - pad, bb[1] - pad),
                      (bb[2] + pad, bb[3] + pad), (238, 211, 34), 2)

    # Label
    sign = hand.get("sign", "")
    if sign:
        label = f"{sign} ({hand.get('label', '')})"
        x0 = bb[0] - 12 if bb else pts[0][0]
        y0 = (bb[1] - 20) if bb else (pts[0][1] - 20)
        cv.putText(frame, label, (x0, y0), cv.FONT_HERSHEY_SIMPLEX,
                   0.7, (180, 120, 255), 2, cv.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="SenseSign Python Receiver / Viewer")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port of the local web_app.py server (default 5000)")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    stream_url = f"{base}/remote_stream"
    result_url = f"{base}/remote_result"

    print("SenseSign Receiver")
    print(f"  Server  : {base}")
    print(f"  Stream  : {stream_url}")
    print(f"  Press Q or ESC to quit\n")

    # Sentence builder state
    sentence = ""
    last_letter = ""
    hold_start = None
    letter_added = False
    HOLD_MS = 1500

    # Open MJPEG stream
    cap = cv.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Waiting for remote camera stream...")
        while not cap.isOpened():
            time.sleep(1)
            cap = cv.VideoCapture(stream_url)
        print("Connected!")

    session = requests.Session()
    last_poll = 0
    result = {"letter": "", "confidence": 0, "hands": [], "active": False}

    while True:
        ret, frame = cap.read()
        if not ret:
            # Stream may have dropped, retry
            time.sleep(0.1)
            cap.release()
            cap = cv.VideoCapture(stream_url)
            continue

        h, w = frame.shape[:2]

        # Poll detection results every ~150ms
        now = time.time()
        if now - last_poll >= 0.15:
            last_poll = now
            try:
                r = session.get(result_url, timeout=2)
                if r.ok:
                    result = r.json()
            except Exception:
                pass

        # Draw hands
        for hand in result.get("hands", []):
            draw_hand(frame, hand, w, h)

        # Detection info overlay
        letter = result.get("letter", "")
        conf = result.get("confidence", 0)

        # Hold-to-lock logic
        if letter and letter == last_letter:
            if hold_start and not letter_added:
                elapsed = (time.time() - hold_start) * 1000
                if elapsed >= HOLD_MS:
                    sentence += last_letter
                    letter_added = True
        else:
            last_letter = letter
            hold_start = time.time() if letter else None
            letter_added = False

        # Draw HUD
        # Dark bar at top
        cv.rectangle(frame, (0, 0), (w, 50), (12, 13, 20), -1)
        if letter:
            cv.putText(frame, f"Sign: {letter}  ({conf}%)", (15, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (180, 120, 255), 2, cv.LINE_AA)
        else:
            cv.putText(frame, "No sign detected", (15, 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 114), 2, cv.LINE_AA)

        # Hold progress bar
        if last_letter and hold_start and not letter_added:
            elapsed = (time.time() - hold_start) * 1000
            pct = min(elapsed / HOLD_MS, 1.0)
            bar_w = int(w * pct)
            cv.rectangle(frame, (0, 50), (bar_w, 54), (238, 211, 34), -1)

        # Sentence bar at bottom
        cv.rectangle(frame, (0, h - 40), (w, h), (12, 13, 20), -1)
        display_sent = sentence if sentence else "Hold sign 1.5s to build sentence..."
        color = (240, 240, 246) if sentence else (90, 90, 114)
        cv.putText(frame, display_sent, (15, h - 12),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv.LINE_AA)

        cv.imshow("SenseSign Receiver", frame)

        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):  # ESC or Q
            break
        elif key == ord(" "):  # Space
            sentence += " "
        elif key == 8:  # Backspace
            sentence = sentence[:-1]
        elif key == ord("x"):  # Clear
            sentence = ""

    cap.release()
    cv.destroyAllWindows()
    if sentence:
        print(f"\nFinal sentence: {sentence}")


if __name__ == "__main__":
    main()
