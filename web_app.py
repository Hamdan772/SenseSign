"""
web_app.py

Flask-based web interface for the ASL Sign Language Translator.
The browser captures webcam frames via getUserMedia and sends them
to the server for hand-detection + classification. This allows
anyone with the link to use their own camera.
"""

import base64
import copy
import csv
import itertools
import threading
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from slr.model.classifier import KeyPointClassifier

app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Configuration ───────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55  # lowered for better recall

# ── MediaPipe Tasks hand detector ──────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

_detector_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="slr/model/hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.5,   # lower = detect hands more reliably
    min_tracking_confidence=0.4,
    min_hand_presence_confidence=0.5,
)
detector = HandLandmarker.create_from_options(_detector_options)

# ── TFLite classifier + labels ─────────────────────────────────
classifier = KeyPointClassifier()
with open("slr/model/label.csv", encoding="utf-8-sig") as f:
    labels = [row[0] for row in csv.reader(f)]


# ── Helpers ─────────────────────────────────────────────────────
def _landmark_list(landmarks, w, h):
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks
    ]


def _bounding_rect(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _pre_process(landmark_list):
    pts = copy.deepcopy(landmark_list)
    bx, by = pts[0]
    for p in pts:
        p[0] -= bx
        p[1] -= by
    flat = list(itertools.chain.from_iterable(pts))
    mx = max(map(abs, flat))
    if mx == 0:
        return flat
    return [v / mx for v in flat]


def _draw_landmarks(image, pts):
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        cv.line(image, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)
    for x, y in pts:
        cv.circle(image, (x, y), 4, (255, 255, 255), -1)
        cv.circle(image, (x, y), 4, (0, 0, 0), 1)
    return image


# ── Remote feed state ───────────────────────────────────────────
_remote_lock = threading.Lock()
_remote_frame = None          # latest raw JPEG bytes from sender
_remote_result = None         # latest detection result dict
_remote_ts = 0                # timestamp of last frame


def _detect_hands(frame):
    """Run hand detection + classification on a BGR frame. Returns result dict."""
    h, w = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    letter = ""
    confidence = 0.0
    hands = []

    for hand_lms, handedness in zip(
        results.hand_landmarks, results.handedness
    ):
        pts = _landmark_list(hand_lms, w, h)
        brect = _bounding_rect(pts)
        processed = _pre_process(pts)

        hand_id, conf = classifier(
            processed, confidence_threshold=CONFIDENCE_THRESHOLD
        )

        hand_label = handedness[0].category_name
        detected_letter = ""
        if hand_id != 25:
            detected_letter = labels[hand_id]
            letter = detected_letter
            confidence = conf

        norm_pts = [[lm.x, lm.y] for lm in hand_lms]
        hands.append({
            "landmarks": norm_pts,
            "label": hand_label,
            "sign": detected_letter,
            "bbox": brect,
        })

    return {
        "letter": letter,
        "confidence": round(confidence * 100, 1),
        "hands": hands,
    }


# ── Routes ──────────────────────────────────────────────────────
@app.route("/health")
def health():
    return "ok", 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_frame():
    """Receive a JPEG frame from the browser, run detection, return results."""
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "no image"}), 400

    img_b64 = data["image"].split(",")[-1]
    img_bytes = base64.b64decode(img_b64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    return jsonify(_detect_hands(frame))


# ── Remote feed (Raspberry Pi → Laptop) ─────────────────────────
@app.route("/remote_frame", methods=["POST"])
def remote_frame():
    """Receive a raw JPEG from sender.py, process it, store results."""
    global _remote_frame, _remote_result, _remote_ts

    jpeg_bytes = request.get_data()
    if not jpeg_bytes:
        return jsonify({"error": "no data"}), 400

    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "bad image"}), 400

    result = _detect_hands(frame)

    with _remote_lock:
        _remote_frame = jpeg_bytes
        _remote_result = result
        _remote_ts = time.time()

    return jsonify({"ok": True})


@app.route("/remote_result")
def remote_result():
    """Poll latest detection result from the remote feed."""
    with _remote_lock:
        if _remote_result and (time.time() - _remote_ts < 5):
            return jsonify({"active": True, **_remote_result})
    return jsonify({"active": False, "letter": "", "confidence": 0, "hands": []})


@app.route("/remote_stream")
def remote_stream():
    """MJPEG stream of the remote camera for the web UI."""
    def generate():
        last_ts = 0
        while True:
            with _remote_lock:
                if _remote_frame and _remote_ts > last_ts:
                    frame_bytes = _remote_frame
                    last_ts = _remote_ts
                else:
                    frame_bytes = None
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
            time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    import os
    import subprocess
    import threading

    port = int(os.environ.get("PORT", 5000))

    # Start cloudflared tunnel when running locally on Windows
    cloudflared_path = os.path.join(os.path.dirname(__file__), "cloudflared", "cloudflared.exe")

    def _start_tunnel():
        try:
            proc = subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                stdin=subprocess.DEVNULL,
            )
            for line in proc.stdout:
                line = line.strip()
                if "trycloudflare.com" in line:
                    for word in line.split():
                        if "trycloudflare.com" in word:
                            url = word if word.startswith("http") else f"https://{word}"
                            print(f"\n{'='*60}", flush=True)
                            print(f"  PUBLIC URL → {url}", flush=True)
                            print(f"  Share this link with anyone!", flush=True)
                            print(f"{'='*60}\n", flush=True)
                            url_file = os.path.join(os.path.dirname(__file__), "public_url.txt")
                            with open(url_file, "w") as uf:
                                uf.write(url)
                            break
        except Exception as e:
            print(f"WARNING: Tunnel failed ({e})", flush=True)

    if os.path.exists(cloudflared_path):
        t = threading.Thread(target=_start_tunnel, daemon=True)
        t.start()
        print("INFO: Starting cloudflared tunnel...")

    print(f"INFO: Local URL  → http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
