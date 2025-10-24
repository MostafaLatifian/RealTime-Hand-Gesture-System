import cv2
import mediapipe as mp
import os
import csv

dataset_path = r"D:\Term 6\Mechatronics Fundamental\archive\train\train"

label_map = {
    "Right Swipe": 0,
    "Left Swipe": 1,
    "Stop Gesture": 2,
    "Thumbs Up": 3,
    "Thumbs Down": 4
}

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

output_csv = "features_train.csv"

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)

    header = []
    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])
    header.append("label")
    writer.writerow(header)

    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)

        gesture_name = None
        for key in label_map.keys():
            if key in folder:
                gesture_name = key
                break
        if gesture_name is None:
            continue

        label = label_map[gesture_name]

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])
                    row.append(label)
                    writer.writerow(row)

print(" The task finished successfully. ")
