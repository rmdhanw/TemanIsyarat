import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_PATH = 'dataset_sibi_landmarks.csv'
print(f"Membaca data dari {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

X = df.drop('label', axis=1).values
y = df['label'].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
NUM_CLASSES = len(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Arsitektur menerima 210 input fitur jarak
model = Sequential([
    Dense(256, activation='relu', input_shape=(210,)), 
    Dropout(0.2),
    Dense(128, activation='relu'),                    
    Dropout(0.2),
    Dense(64, activation='relu'),                     
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Memulai proses training model (Metode Pairwise Distance)...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAkurasi pada data testing: {accuracy * 100:.2f}%")

MODEL_SAVE_PATH = 'sibi_model_distance.h5'
model.save(MODEL_SAVE_PATH)