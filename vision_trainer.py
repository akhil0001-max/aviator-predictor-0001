import cv2
import pytesseract
import time
import re
import numpy as np
from datetime import datetime
from collections import deque

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Update if needed
ocr_config = '--psm 6'

MAX_HISTORY = 100
multiplier_history = deque(maxlen=MAX_HISTORY)

def extract_multiplier(text):
    match = re.search(r'(\\d+\\.\\d+)x', text)
    return float(match.group(1)) if match else None

def start_camera_training():
    cap = cv2.VideoCapture(0)
    print("[INFO] Camera started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=ocr_config)
        multiplier = extract_multiplier(text)

        if multiplier:
            timestamp = datetime.now().strftime('%H:%M:%S')
            multiplier_history.append((timestamp, multiplier))
            print(f"[{timestamp}] Multiplier Detected: {multiplier}x")

        cv2.imshow('AI Training Camera Feed', resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n[INFO] Collected multipliers:")
    for t, m in multiplier_history:
        print(f"{t} - {m}x")
