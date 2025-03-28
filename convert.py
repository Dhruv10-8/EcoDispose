import tensorflow as tf

# ✅ Path to the .h5 model
model_path = "C:\\Users\\dhruv\\Downloads\\DEVPOST\\h1_two\\waste_classification_model.h5"

# ✅ Load the Keras model
model = tf.keras.models.load_model(model_path)

# ✅ Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# ✅ Save the new TFLite model
tflite_model_path = "C:\\Users\\dhruv\\Downloads\\DEVPOST\\h1_two\\waste_classification_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"✅ Model successfully converted and saved to {tflite_model_path}")
