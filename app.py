# User uploads image on index.html
#         ↓
# index.html sends image to Flask (POST /predict)
#         ↓
# Flask receives image
#         ↓
# Loads it, resizes it, converts to numbers
#         ↓
# model.predict() → score
#         ↓
# Flask sends back {"result": "CAT (confidence: 99%)"}
#         ↓
# index.html displays the result on screen


import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

app = Flask(__name____)                                 # Create web server
CORS(app)                                               # gives HTML page permission to send requests to Flask.

model = load_model("cat_dog_model.h5")                  # Load train model

@app.route("/predict", methods = ['POST'])              # when "who am i?" is clicked in website, flask receive it in here.
def predict():
    file = request.files["file"]                        # SAYS : give me the file the user uploaded
    img = load_img(file, target_size=(224, 224))        # resize
    img_array = img_to_array(img)                       # convert to numbers
    img_array = np.expand_dims(img_array, axis=0)       # add batch dimension

    prediction = model.predict(img_array)
    score = prediction[0][0]


    if score < 0.5:
        result = f"CAT (confidence: {(1 - score) * 100:.2f}%)"
    else:
        result = f"DOG (confidence: {score * 100:.2f}%)"

    
    # sends the prediction back to  website as JSON.

    return jsonify({"result": result})  

if __name__ == "__main__":
    app.run(debug = True)
