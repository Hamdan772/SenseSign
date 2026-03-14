"""
receiver_tcp.py

Python-to-Python TCP receiver for SenseSign.
Listens for JPEG frames from sender_tcp.py, performs hand detection +
classification locally, and displays translated signs in an OpenCV window.

Usage:
    python receiver_tcp.py --port 9000
"""

import argparse
import copy
import csv
import itertools
import socket
import struct
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from slr.model.classifier import KeyPointClassifier


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

CONFIDENCE_THRESHOLD = 0.55


def recvall(sock, size):
    """Receive exactly size bytes or return None if disconnected."""
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def landmark_list(landmarks, w, h):
    return [
        [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
        for lm in landmarks
    ]


def bounding_rect(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def pre_process(pts):
    norm = copy.deepcopy(pts)
    bx, by = norm[0]
    for p in norm:
        p[0] -= bx
        p[1] -= by
    flat = list(itertools.chain.from_iterable(norm))
    mx = max(map(abs, flat))
    if mx == 0:
        return flat
    return [v / mx for v in flat]


def detect_hands(frame, detector, classifier, labels):
    h, w = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    letter = ""
    conf_pct = 0.0
    hands = []

    for hand_lms, handedness in zip(results.hand_landmarks, results.handedness):
        pts = landmark_list(hand_lms, w, h)
        brect = bounding_rect(pts)
        processed = pre_process(pts)
        hand_id, conf = classifier(processed, confidence_threshold=CONFIDENCE_THRESHOLD)

        detected = ""
        if hand_id != 25:
            detected = labels[hand_id]
            letter = detected
            conf_pct = conf * 100.0

        hands.append({
            "landmarks": [[lm.x, lm.y] for lm in hand_lms],
            "bbox": brect,
            "label": handedness[0].category_name,
            "sign": detected,
        })

    return {
        "letter": letter,
        "confidence": round(conf_pct, 1),
        "hands": hands,
    }


def draw_overlay(frame, result, sentence, hold_start, hold_ms, last_letter, letter_added):
    h, w = frame.shape[:2]

    for hand in result.get("hands", []):
        pts = [(int(x * w), int(y * h)) for x, y in hand.get("landmarks", [])]
        if len(pts) != 21:
            continue

        for a, b in CONNECTIONS:
            cv.line(frame, pts[a], pts[b], (180, 120, 255), 2, cv.LINE_AA)

        for x, y in pts:
            cv.circle(frame, (x, y), 4, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, (x, y), 4, (0, 0, 0), 1, cv.LINE_AA)

        bb = hand.get("bbox", [])
        if len(bb) == 4:
            pad = 12
            cv.rectangle(frame, (bb[0] - pad, bb[1] - pad), (bb[2] + pad, bb[3] + pad), (34, 211, 238), 2)
            if hand.get("sign"):
                cv.putText(
                    frame,
                    hand["sign"],
                    (bb[0] - pad, bb[1] - pad - 8),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv.LINE_AA,
                )

    cv.rectangle(frame, (0, 0), (w, 52), (12, 13, 20), -1)
    letter = result.get("letter", "")
    conf = result.get("confidence", 0)
    if letter:
        cv.putText(frame, f"Sign: {letter} ({conf}%)", (14, 34), cv.FONT_HERSHEY_SIMPLEX, 0.85, (180, 120, 255), 2, cv.LINE_AA)
    else:
        cv.putText(frame, "No sign detected", (14, 34), cv.FONT_HERSHEY_SIMPLEX, 0.72, (90, 90, 114), 2, cv.LINE_AA)

    if last_letter and hold_start and not letter_added:
        elapsed = (time.time() - hold_start) * 1000
        pct = min(elapsed / hold_ms, 1.0)
        bar_w = int(w * pct)
        cv.rectangle(frame, (0, 52), (bar_w, 56), (34, 211, 238), -1)

    cv.rectangle(frame, (0, h - 40), (w, h), (12, 13, 20), -1)
    sentence_text = sentence if sentence else "Hold sign 1.5s to append letter"
    color = (240, 240, 246) if sentence else (90, 90, 114)
    cv.putText(frame, sentence_text, (14, h - 14), cv.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="SenseSign TCP receiver with ASL translation")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9000, help="Bind port (default 9000)")
    args = parser.parse_args()

    with open("slr/model/label.csv", encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="slr/model/hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.4,
        min_hand_presence_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)
    classifier = KeyPointClassifier()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)

    print("SenseSign TCP Receiver")
    print(f"Listening on {args.host}:{args.port}")
    print("Press Q or ESC to quit window")

    hold_ms = 1500
    sentence = ""
    last_letter = ""
    hold_start = None
    letter_added = False

    try:
        while True:
            conn, addr = srv.accept()
            print(f"Connected: {addr[0]}:{addr[1]}")
            conn.settimeout(10)
            with conn:
                while True:
                    header = recvall(conn, 4)
                    if header is None:
                        print("Sender disconnected.")
                        break

                    (size,) = struct.unpack(">I", header)
                    payload = recvall(conn, size)
                    if payload is None:
                        print("Sender disconnected.")
                        break

                    arr = np.frombuffer(payload, dtype=np.uint8)
                    frame = cv.imdecode(arr, cv.IMREAD_COLOR)
                    if frame is None:
                        continue

                    result = detect_hands(frame, detector, classifier, labels)
                    letter = result.get("letter", "")

                    if letter and letter == last_letter:
                        if hold_start and not letter_added:
                            elapsed = (time.time() - hold_start) * 1000
                            if elapsed >= hold_ms:
                                sentence += last_letter
                                letter_added = True
                    else:
                        last_letter = letter
                        hold_start = time.time() if letter else None
                        letter_added = False

                    draw_overlay(frame, result, sentence, hold_start, hold_ms, last_letter, letter_added)
                    cv.imshow("SenseSign TCP Receiver", frame)

                    key = cv.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        return
                    if key == ord(" "):
                        sentence += " "
                    elif key == 8:
                        sentence = sentence[:-1]
                    elif key == ord("x"):
                        sentence = ""

    finally:
        srv.close()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
