import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

model = joblib.load("hand_gesture_model.joblib")

blue = (255, 0, 0)
red = (0, 0, 255)
pink = (255, 0, 255)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

device = AudioUtilities.GetSpeakers()
interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, _ = volume.GetVolumeRange()
current_vol = volume.GetMasterVolumeLevel()

cap = cv2.VideoCapture(0)
prev_cx, prev_cy = None, None
gesture = "Waiting..."
last_seen_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        last_seen_time = time.time()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            features = []
            x_list, y_list = [], []
            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                features.extend([lm.x, lm.y])

            cx, cy = np.mean(x_list), np.mean(y_list)

            pred = model.predict([features])[0]
            gesture_map = {
                2: "Stop Gesture",
                3: "Thumbs Up",
                4: "Thumbs Down"
            }

            if pred in gesture_map:
                gesture = gesture_map[pred]

                if gesture == "Thumbs Up":
                    current_vol = min(current_vol + 0.5, max_vol)
                    volume.SetMasterVolumeLevel(current_vol, None)
                elif gesture == "Thumbs Down":
                    current_vol = max(current_vol - 0.5, min_vol)
                    volume.SetMasterVolumeLevel(current_vol, None)
            else:
                # Swipe detection
                if prev_cx is not None:
                    dx = cx - prev_cx
                    if abs(dx) > 0.02:
                        gesture = "Right Swipe" if dx > 0 else "Left Swipe"

            prev_cx, prev_cy = cx, cy
    else:
        if time.time() - last_seen_time > 0.5:
            gesture = "Waiting..."
            prev_cx, prev_cy = None, None

    cv2.putText(frame, f"Gesture: {gesture}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, pink, 3)
    cv2.imshow("Hybrid Gesture with Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()


