from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from flask_cors import CORS
import cv2

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load your ML model
mdl = tf.keras.models.load_model('./model.h5')
print("✅ Model loaded successfully!")
print(mdl.summary())


# Define the new dimensions
width = 150
height = 150
dimensions = (width, height)

image1 = r'C:\CropProject\test_case\img1.jpg'
image = cv2.imread(image1)


def tranfomation(image):
    
    # Resize the image using bicubic interpolation
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)

    # normalizing the image and resozing it
    normalized_image= resized_image / 255.0

    print(normalized_image)
    normalized_image = np.expand_dims(normalized_image,axis=0)
    # predict
    predict = mdl.predict(normalized_image)[0][0]
    print(predict)

    result = "Disease" if predict == 1 else "No Disease"
    return result

final_result = tranfomation(image)
print(final_result)
'''''
# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']  # Get uploaded file
        img = image.load_img(file, target_size=(128, 128))  # Resize for model
        img_array = image.img_to_array(img) / 255.0         # Normalize
        img_array = np.expand_dims(img_array, axis=0)       # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])

        # Return result as JSON
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
'''