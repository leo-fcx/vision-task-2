import tensorflow as tf
import cv2
import numpy as np
import os

MODEL_PATH = os.path.abspath("classifier/model_subclases_cellphones.keras")

# Cargar modelo entrenado
model_subclases = tf.keras.models.load_model(MODEL_PATH)

# Etiquetas según tu dataset de subclases
subclase_labels = {
    0: "iPhone",
    1: "Samsung",
    2: "Xiaomi"
}

IMG_SIZE = (224, 224)

def preprocess_crop(crop_img):
    img = cv2.resize(crop_img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def classify_subclass(crop_img):
    input_img = preprocess_crop(crop_img)
    preds = model_subclases.predict(input_img)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    label = subclase_labels.get(class_idx, "Desconocido")
    return label, confidence
