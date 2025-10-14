from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
import cv2
import os
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your ML model
mdl = tf.keras.models.load_model('./model.h5')
print("✅ Model loaded successfully!")
try:
    print(mdl.summary())
except Exception:
    pass


# Define the new dimensions
width = 150
height = 150
dimensions = (width, height)


def tranfomation(image):
    """Process image and return prediction result with score."""
    # Resize the image using bicubic interpolation
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)

    # normalizing the image and resizing it
    normalized_image = resized_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)
    
    # predict (get raw score between 0 and 1)
    pred_score = float(mdl.predict(normalized_image)[0][0])
    print(f"Raw prediction score: {pred_score}")  # Debug print
    
    # Use 0.5 as threshold for binary classification
    is_disease = pred_score >= 0.5
    result = "Disease" if is_disease else "No Disease"
    
    return result, pred_score

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure file part exists
        if 'file' not in request.files:
            return jsonify({'error': 'no file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'no selected file'}), 400

        # Save the uploaded file
        if file and file.filename:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            # Read the saved image using cv2 (same as your original code)
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({'error': 'could not read saved image'}), 400

            # Use your existing transformation function
            result, score = tranfomation(image)
            
            # Print debug info
            print(f"Prediction for {filepath}:")
            print(f"Score: {score:.4f}")
            print(f"Result: {result}")
            
            return jsonify({
                'prediction': 1 if result == "Disease" else 0,
                'result': result,
                'score': float(score),
                'filepath': filepath
            })
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
