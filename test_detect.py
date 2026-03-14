"""Quick diagnostic: capture webcam frames and test detection pipeline."""
import copy
import csv
import itertools
import time

import cv2 as cv
import mediapipe as mp
import numpy as np

from slr.model.classifier import KeyPointClassifier

# Load detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
opts = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="slr/model/hand_landmarker.task"),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.4,
    min_hand_presence_confidence=0.5,
)
det = HandLandmarker.create_from_options(opts)

clf = KeyPointClassifier()
with open("slr/model/label.csv", encoding="utf-8-sig") as f:
    labels = [row[0] for row in csv.reader(f)]

print(f"Labels ({len(labels)}): {labels}")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

time.sleep(1)
print("Show your hand to the camera...\n")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: capture failed")
        continue

    h, w = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = det.detect(mp_image)

    n = len(results.hand_landmarks)
    print(f"Frame {i}: {w}x{h}, hands={n}")

    for j, (hand_lms, handedness) in enumerate(
        zip(results.hand_landmarks, results.handedness)
    ):
        pts = [
            [min(int(lm.x * w), w - 1), min(int(lm.y * h), h - 1)]
            for lm in hand_lms
        ]
        pts_c = copy.deepcopy(pts)
        bx, by = pts_c[0]
        for p in pts_c:
            p[0] -= bx
            p[1] -= by
        flat = list(itertools.chain.from_iterable(pts_c))
        mx = max(map(abs, flat))
        if mx > 0:
            flat = [v / mx for v in flat]

        hand_id, conf = clf(flat, confidence_threshold=0.55)
        side = handedness[0].category_name
        if hand_id < len(labels):
            ltr = labels[hand_id]
        else:
            ltr = "?"
        status = ltr if hand_id != 25 else "(below threshold)"
        print(f"  Hand {j}: {side}, idx={hand_id}, conf={conf:.3f}, result={status}")

    time.sleep(0.5)

cap.release()
print("\nDone.")
