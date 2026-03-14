"""
sender_tcp.py

Python-to-Python TCP video sender for Raspberry Pi (or any camera host).
Sends JPEG frames to a laptop receiver over a raw TCP socket.

Usage:
    python sender_tcp.py --host 192.168.1.100 --port 9000
"""

import argparse
import socket
import struct
import sys
import time

import cv2


def connect_with_retry(host, port, retry_delay=2.0):
    """Connect to the receiver, retrying until available."""
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            print(f"Waiting for receiver {host}:{port} ({exc})")
            time.sleep(retry_delay)


def main():
    parser = argparse.ArgumentParser(description="SenseSign TCP camera sender")
    parser.add_argument("--host", required=True, help="Laptop receiver IP")
    parser.add_argument("--port", type=int, default=9000, help="Receiver port (default 9000)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    parser.add_argument("--fps", type=int, default=15, help="Target send rate (default 15)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality 1-100")
    args = parser.parse_args()

    frame_interval = 1.0 / max(args.fps, 1)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print("SenseSign TCP Sender")
    print(f"  Camera : {args.camera}")
    print(f"  Target : {args.host}:{args.port}")
    print(f"  Stream : {args.fps} fps | {args.width}x{args.height} | JPEG q{args.quality}")

    sock = connect_with_retry(args.host, args.port)
    print("Connected to receiver. Streaming...")

    sent = 0
    dropped = 0
    stat_t0 = time.time()

    try:
        while True:
            loop_start = time.time()

            ok, frame = cap.read()
            if not ok:
                dropped += 1
                time.sleep(0.05)
                continue

            enc_ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not enc_ok:
                dropped += 1
                continue

            payload = buf.tobytes()
            header = struct.pack(">I", len(payload))

            try:
                sock.sendall(header)
                sock.sendall(payload)
                sent += 1
            except OSError:
                print("Connection lost. Reconnecting...")
                try:
                    sock.close()
                except OSError:
                    pass
                sock = connect_with_retry(args.host, args.port)
                continue

            elapsed = time.time() - stat_t0
            if elapsed >= 2.0:
                fps = sent / elapsed if elapsed > 0 else 0.0
                print(f"Sent: {sent} | {fps:.1f} fps | Dropped: {dropped}", end="\r")
                sent = 0
                dropped = 0
                stat_t0 = time.time()

            sleep_time = frame_interval - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        try:
            sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    main()
