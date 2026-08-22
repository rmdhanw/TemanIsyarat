import os
import cv2
import csv
import numpy as np

import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

DATASET_DIR = 'dataset/SIBI'
CSV_PATH = 'dataset_sibi_landmarks.csv'

header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

print("Start Extraction Process!")

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for dir_name in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, dir_name)
        
        if not os.path.isdir(class_path):
            continue

        print(f"Mengekstrak kelas: {dir_name}...")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = []
                    
                    for lm in hand_landmarks.landmark:
                        landmark_list.append([lm.x, lm.y, lm.z])

                    base_x, base_y, base_z = landmark_list[0]
                    normalized_list = []
                    for lm in landmark_list:
                        normalized_list.append([lm[0] - base_x, lm[1] - base_y, lm[2] - base_z])

                    flat_list = list(np.concatenate(normalized_list))
                    max_val = max(list(map(abs, flat_list)))

                    if max_val > 0:
                        flat_list = [val / max_val for val in flat_list]

                    row = [dir_name] + flat_list
                    writer.writerow(row)

hands.close()
print(f"\nDone Extraction! Saved to {CSV_PATH}")