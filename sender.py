"""
sender.py

Video sender for Raspberry Pi (or any remote camera).
Captures webcam frames and streams them over the network to the
SenseSign server for ASL detection.

Usage:
    python sender.py --server http://LAPTOP_IP:5000

The laptop must be running web_app.py. This script sends raw JPEG
frames to the /remote_frame endpoint at ~15 fps.
"""

import argparse
import sys
import time

import cv2
import requests


def main():
    parser = argparse.ArgumentParser(description="SenseSign remote camera sender")
    parser.add_argument(
        "--server",
        required=True,
        help="Server URL, e.g. http://192.168.1.100:5000",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--fps", type=int, default=15, help="Target send rate (default 15)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100 (default 80)")
    args = parser.parse_args()

    server_url = args.server.rstrip("/") + "/remote_frame"
    frame_interval = 1.0 / args.fps

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: Could not open camera", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"SenseSign Sender")
    print(f"  Camera : {args.camera}")
    print(f"  Server : {server_url}")
    print(f"  Target : {args.fps} fps  |  {args.width}x{args.height}  |  JPEG q{args.quality}")
    print(f"  Press Ctrl+C to stop\n")

    session = requests.Session()
    sent = 0
    errors = 0
    t0 = time.time()

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("WARNING: Frame capture failed, retrying...")
                time.sleep(0.1)
                continue

            # Encode as JPEG
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue

            try:
                resp = session.post(
                    server_url,
                    data=buf.tobytes(),
                    headers={"Content-Type": "image/jpeg"},
                    timeout=5,
                )
                if resp.status_code == 200:
                    sent += 1
                else:
                    errors += 1
            except requests.RequestException:
                errors += 1

            # Print stats every 2 seconds
            elapsed = time.time() - t0
            if elapsed >= 2.0:
                fps = sent / elapsed if elapsed > 0 else 0
                print(f"  Sent: {sent} frames  |  {fps:.1f} fps  |  Errors: {errors}", end="\r")
                sent = 0
                errors = 0
                t0 = time.time()

            # Rate limiting
            sleep_time = frame_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
