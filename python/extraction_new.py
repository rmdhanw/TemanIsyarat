import os
import cv2
import csv
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

DATASET_DIR = 'dataset/SIBI'
CSV_PATH = 'dataset_sibi_distances.csv'

header = ['label']
for i in range(21):
    for j in range(i+1, 21):
        header.append(f'dist_{i}_{j}')

print("Memulai proses ekstraksi Pairwise Distance...")

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for dir_name in sorted(os.listdir(DATASET_DIR)):
        class_path = os.path.join(DATASET_DIR, dir_name)
        if not os.path.isdir(class_path): continue

        print(f"Mengekstrak kelas: {dir_name}...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    
                    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    
                    distances = []
                    for i in range(21):
                        for j in range(i+1, 21):
                            dist = np.linalg.norm(pts[i] - pts[j])
                            distances.append(dist)

                    max_d = max(distances)
                    if max_d > 0:
                        distances = [d / max_d for d in distances]

                    writer.writerow([dir_name] + distances)

hands.close()
print(f"\nEkstraksi selesai! Data tersimpan di {CSV_PATH}")