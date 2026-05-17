import cv2
import socket
import math
import pickle
import sys
import serial
import time
import collections

max_length = 65000
host = sys.argv[1]
port = 5000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Setup TF-Luna (using robust initialization)
try:
    ser = serial.Serial(
        port        = "/dev/serial0",
        baudrate    = 115200,
        bytesize    = serial.EIGHTBITS,
        parity      = serial.PARITY_NONE,
        stopbits    = serial.STOPBITS_ONE,
        timeout     = 0.01,
        write_timeout = 0.1,
    )
    # Send command to force-start the LiDAR output
    cmd = [0x5A, 0x05, 0x07, 0x01]
    cmd.append(sum(cmd) & 0xFF)
    ser.write(bytes(cmd))
    time.sleep(0.1)
    
    # Send command to set 50hz frequency (Balance of accuracy and speed)
    cmd = [0x5A, 0x06, 0x03, 50, 0x00]
    cmd.append(sum(cmd) & 0xFF)
    ser.write(bytes(cmd))
    time.sleep(0.1)
    
    ser.reset_input_buffer()
    has_lidar = True
except Exception as e:
    print(f"Lidar warning: {e}")
    has_lidar = False

def read_tfluna():
    if not has_lidar:
        return -1
    try:
        if ser.in_waiting >= 9:
            # Read all available bytes
            raw_data = ser.read(ser.in_waiting)
            
            # Find the most recent frame marker (0x59 0x59)
            idx = raw_data.rfind(b'\x59\x59')
            if idx != -1 and len(raw_data) >= idx + 9:
                payload = raw_data[idx+2 : idx+9]
                dist_raw = payload[0] | (payload[1] << 8)
                checksum = payload[6]
                
                # Checksum verify
                expected = (0x59 + 0x59 + sum(payload[:6])) & 0xFF
                if checksum == expected:
                    return dist_raw
    except Exception as e:
        print(f"Lidar error: {e}")
    return -1

class PrecisionFilter:
    def __init__(self, n=5):
        self._n = n
        self._buf = collections.deque(maxlen=n)
        self._last = None

    def update(self, value):
        self._buf.append(value)
        # Apply smoothing average
        self._last = int(sum(self._buf) / len(self._buf))
        return self._last

dist_filter = PrecisionFilter(n=4) # Buffer last 4 frames for stability

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
ret, frame = cap.read()

dist_cm = -1

while ret:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    retval, buffer = cv2.imencode(".jpg", frame, encode_param)
    
    # Read distance
    new_dist = read_tfluna()
    # Check for valid TF-Luna range (20cm to 800cm)
    if new_dist != -1 and 20 <= new_dist <= 800:
        # Apply smoothing filter for higher accuracy
        dist_cm = dist_filter.update(new_dist)

    if retval:
        buffer = buffer.tobytes()
        buffer_size = len(buffer)
        num_of_packs = 1
        if buffer_size > max_length:
            num_of_packs = math.ceil(buffer_size/max_length)

        frame_info = {"packs": num_of_packs, "distance": dist_cm}
        sock.sendto(pickle.dumps(frame_info), (host, port))
        
        left = 0
        right = max_length

        for i in range(num_of_packs):
            data = buffer[left:right]
            left = right
            right += max_length
            sock.sendto(data, (host, port))
    
    ret, frame = cap.read()

print("done")
