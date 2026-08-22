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
print(f"Jumlah kelas terdeteksi: {NUM_CLASSES} ({encoder.classes_})")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Memulai proses training model...")
history = model.fit(
    X_train, y_train,
    epochs=50, 
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nAkurasi pada data testing: {accuracy * 100:.2f}%")

MODEL_SAVE_PATH = 'sibi_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"Model berhasil disimpan di {MODEL_SAVE_PATH}")

print("\nMapping Label (Ingat urutan ini untuk aplikasi Anda nanti!):")
for i, label in enumerate(encoder.classes_):
    print(f"{i} : {label}")