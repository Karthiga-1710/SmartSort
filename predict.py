import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Class labels (auto from training)
class_names = ['Cardboard', 'food Organics', 'Glass']  # adjust if you have more folders

# Load a test image
img_path = "smart-waste-cv/Glass/Glass_1.jpg"  # change path to any image you want
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediction
pred = model.predict(img_array)
class_index = np.argmax(pred)

print("Predicted class:", class_names[class_index])