import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput.keyboard import Key, Controller as KeyboardController


blue = (255, 0, 0)
red = (0, 0, 255)
pink = (255, 0, 255)

try:
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    min_vol_db, max_vol_db, _ = volume.GetVolumeRange()
    print("Volume control successfully initialized.")
except Exception as e:
    print(f"Warning: Unable to initialize volume control (pycaw): {e}")
    print("Volume control functionality will be disabled.")
    volume = None

keyboard = KeyboardController()
print("Keyboard control (pynput) successfully initialized.")

# Load trained model
model = joblib.load("hand_gesture_model.joblib")
print(f"\nModel successfully loaded from 'hand_gesture_model.joblib'.")

# --- 3. Initialize MediaPipe Hands detector ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.85, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# --- 4. Define complete gesture mapping (all 5 gestures) ---
label_map = {
    0: "Right Swipe",
    1: "Left Swipe",
    2: "Stop Gesture",
    3: "Thumbs Up",
    4: "Thumbs Down"
}
print("\nComplete gesture mapping:")
for label, name in label_map.items():
    print(f"  {label}: {name}")

# --- 5. Camera setup and state variables ---
cap = cv2.VideoCapture(0)
prev_cx, prev_cy = None, None
gesture = "Waiting..."
last_seen_time = time.time()
last_gesture_time = {g: 0 for g in label_map.values()}
cooldown_time = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # --- Initialize variable at the beginning of loop with a default value ---
    predicted_gesture_name = "No Hand Detected"

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

            # --- Gesture recognition with hybrid approach ---
            current_time = time.time()

            # First predict static gestures with the model
            pred = model.predict([features])[0]

            # Static gesture mapping
            static_gesture_map = {
                2: "Stop Gesture",
                3: "Thumbs Up",
                4: "Thumbs Down"
            }

            if pred in static_gesture_map:
                predicted_gesture_name = static_gesture_map[pred]
            else:
                # If not a static gesture, use rule-based approach for Swipe
                if prev_cx is not None:
                    dx = cx - prev_cx
                    if abs(dx) > 0.02:
                        predicted_gesture_name = "Right Swipe" if dx > 0 else "Left Swipe"

            # --- Execute system operations using a single model ---
            if predicted_gesture_name != "No Hand Detected" and current_time - last_gesture_time.get(
                    predicted_gesture_name, 0) > cooldown_time:

                if predicted_gesture_name == "Right Swipe":
                    print(f"\nGesture detected: {predicted_gesture_name} (Forward video)")
                    keyboard.press(Key.right)
                    keyboard.release(Key.right)
                elif predicted_gesture_name == "Left Swipe":
                    print(f"\nGesture detected: {predicted_gesture_name} (Rewind video)")
                    keyboard.press(Key.left)
                    keyboard.release(Key.left)
                elif predicted_gesture_name == "Thumbs Up":
                    if volume:
                        print(f"\nGesture detected: {predicted_gesture_name} (Increase volume)")
                        current_vol_db = volume.GetMasterVolumeLevel()
                        new_vol_db = min(max_vol_db, current_vol_db + 4.0)
                        volume.SetMasterVolumeLevel(new_vol_db, None)
                elif predicted_gesture_name == "Thumbs Down":
                    if volume:
                        print(f"\nGesture detected: {predicted_gesture_name} (Decrease volume)")
                        current_vol_db = volume.GetMasterVolumeLevel()
                        new_vol_db = max(min_vol_db, current_vol_db - 4.0)
                        volume.SetMasterVolumeLevel(new_vol_db, None)
                elif predicted_gesture_name == "Stop Gesture":
                    print(f"\nGesture detected: {predicted_gesture_name} (Play/Pause video)")
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)

                gesture = predicted_gesture_name
                last_gesture_time[predicted_gesture_name] = current_time

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