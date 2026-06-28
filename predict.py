import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array


def predict_image(model, image_path):
    img = load_img(image_path, target_size = (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    prediction = model.predict(img_array)
    score = prediction[0][0]

    if score < 0.5 :
        print(f"Prediction: CAT (confidence: {(1 - score) * 100:.2f}%)")

    else:
        print(f"Prediction : DOG (confident: {score * 100:.2f}%)")

