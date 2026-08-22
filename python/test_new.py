import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = 'sibi_model_distance.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pairwise Distance Extraction
            pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
            distances = []
            
            for i in range(21):
                for j in range(i+1, 21):
                    dist = np.linalg.norm(pts[i] - pts[j])
                    distances.append(dist)

            max_d = max(distances)
            if max_d > 0:
                distances = [d / max_d for d in distances]

            input_data = np.array([distances], dtype=np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_index = np.argmax(output_data)
            confidence = output_data[0][predicted_index]

            if confidence > 0.6:
                text = f"Isyarat: {LABELS[predicted_index]} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sistem Uji SIBI (Metode Jarak)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

hands.close()
cap.release()
cv2.destroyAllWindows()