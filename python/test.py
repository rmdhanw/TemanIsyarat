import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

MODEL_PATH = 'sibi_model.tflite'
print(f"Memuat model dari {MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#Define Labels
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Open Cam
cap = cv2.VideoCapture(0)
print("Press 'q' for exit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error Camera!")
        break

    frame = cv2.flip(frame, 1)
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Hand Detect
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
            

            input_data = np.array([flat_list], dtype=np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            predicted_index = np.argmax(output_data)
            confidence = output_data[0][predicted_index]

            if confidence > 0.6:
                predicted_char = LABELS[predicted_index]
                text = f"Isyarat: {predicted_char} ({confidence*100:.1f}%)"
                
                cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sistem Uji SIBI Real-time', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()