import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import sys

MODEL_PATH = "skin_model.h5"
IMG_SIZE = 224

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# MUST match training class order
class_names = [
    "acne",
    "contact_dermatitis",
    "eczema",
    "fungal",
    "psoriasis",
    "urticaria",
    "warts"
]

def predict_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
