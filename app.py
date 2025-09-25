from flask import Flask, request, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

STATIC_DIR = os.path.join(app.root_path, 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

# Load model with Keras
model = tf.keras.models.load_model("model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction_text="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text="No image selected")

    # Save file temporarily
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))  # adjust size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    result = "Mask Detected 😷" if prediction[0][0] > 0.5 else "No Mask 🙅‍♂️"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
