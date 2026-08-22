import tensorflow as tf

print("Memuat model H5...")
model = tf.keras.models.load_model('sibi_model_new.h5')

print("Mengonversi ke TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open('sibi_model_new.tflite', 'wb') as f:
    f.write(tflite_model)

print("Done: sibi_model.tflite")