from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import uuid
from tensorflow.keras.utils import load_img, img_to_array
def entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))



app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

IMG_SIZE = 224

# ✅ Load native Keras model (matches TF 2.13)
model = tf.keras.models.load_model("skin_model.h5")

class_names = [
    "acne",
    "contact_dermatitis",
    "eczema",
    "fungal",
    "psoriasis",
    "urticaria",
    "warts"
]

disease_info = {
    "Normal Skin": {
    "description": "No visible signs of skin disease detected.",
    "precautions": "Maintain good hygiene, stay hydrated, and follow a healthy skincare routine."
},

    "acne": {
        "description": "A common skin condition caused by clogged pores.",
        "precautions": "Avoid oily cosmetics, cleanse skin regularly."
    },
    "contact_dermatitis": {
        "description": "Skin inflammation due to irritants or allergens.",
        "precautions": "Avoid irritants, use soothing creams."
    },
    "eczema": {
        "description": "Chronic inflammatory skin condition causing itching.",
        "precautions": "Moisturize often, avoid allergens."
    },
    "fungal": {
        "description": "Skin infection caused by fungi.",
        "precautions": "Keep skin dry, use antifungal creams."
    },
    "psoriasis": {
        "description": "Autoimmune condition causing skin scaling.",
        "precautions": "Manage stress, moisturize, consult doctor."
    },
    "urticaria": {
        "description": "Allergic skin reaction causing itchy welts.",
        "precautions": "Avoid allergens, take antihistamines."
    },
    "warts": {
        "description": "Viral infection causing skin growths.",
        "precautions": "Avoid touching, maintain hygiene."
    }
}

history = []

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = confidence = severity = info = image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = model.predict(img_array)[0]

        max_prob = np.max(preds)
        confidence = round(float(max_prob) * 100, 2)

        uncertainty = entropy(preds)

        THRESHOLD = 0.50
        ENTROPY_LIMIT = 1.2

        if max_prob < THRESHOLD or uncertainty > ENTROPY_LIMIT:
            prediction = "Normal Skin"
            severity = "None"
            info = {
                "description": "No clear skin disease detected.",
                "precautions": "Skin appears healthy. Maintain regular skincare."
            }
        else:
            prediction = class_names[np.argmax(preds)]

            severity = (
                "Mild" if confidence < 60 else
                "Moderate" if confidence < 80 else
                "Severe"
            )

        info = disease_info[prediction]

        image_path = filepath

        history.insert(0, {
                "disease": prediction,
                "confidence": confidence,
                "severity": severity
            })
        history[:] = history[:5]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        info=info,
        history=history,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
