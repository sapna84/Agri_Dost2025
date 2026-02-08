from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import sys
import time
import socket
import urllib.request
import json

# Optional heavy ML imports; keep app runnable if ML deps missing
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
except Exception:
    tf = None
    image = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

# Configure logging to stdout for clearer startup feedback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout
)

# Serve frontend from frontend/ and provide /predict on port 5001
app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Try to load model if TensorFlow is available
mdl = None
if tf is not None:
    logging.info("TensorFlow detected, attempting to load model at ./model.h5")
    try:
        t0 = time.time()
        mdl = tf.keras.models.load_model('./model.h5')
        logging.info("✅ Model loaded successfully in %.2f seconds", time.time() - t0)
        # skip printing model.summary() which can be slow and noisy
    except Exception as e:
        logging.exception("⚠️ Could not load model at startup (continuing without it): %s", e)

# Image settings
width = 150
height = 150
dimensions = (width, height)


@app.route('/')
def index():
    return app.send_static_file('home.html')


@app.route('/<path:path>')
def serve_frontend(path):
    try:
        return app.send_static_file(path)
    except Exception:
        return app.send_static_file('home.html')


def tranfomation(image):
    if cv2 is None or np is None or mdl is None:
        raise RuntimeError('ML dependencies or model not available')

    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)
    normalized_image = resized_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)

    pred_score = float(mdl.predict(normalized_image)[0][0])
    is_disease = pred_score >= 0.5
    result = 'Disease' if is_disease else 'No Disease'
    return result, pred_score


@app.route('/predict', methods=['POST'])
def predict():
    if mdl is None:
        return jsonify({'error': 'model not available'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'no file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no selected file'}), 400

    uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    filepath = os.path.join(uploads_dir, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath) if cv2 is not None else None
    if img is None:
        return jsonify({'error': 'could not read saved image'}), 400

    try:
        result, score = tranfomation(img)
        return jsonify({'prediction': 1 if result == 'Disease' else 0, 'result': result, 'score': float(score)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# New: simple health endpoint to quickly check server + model status
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'pid': os.getpid(),
        'model_loaded': mdl is not None
    }), 200


# Log incoming requests (method + path) to help diagnose frontend failures
@app.before_request
def log_request():
    logging.info("Incoming request: %s %s", request.method, request.path)


# Provide /status endpoint used by frontend (historically requested)
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok', 'model_loaded': mdl is not None}), 200


# Provide a simple placeholder for /api/weather so frontend requests stop 404ing
@app.route('/api/weather', methods=['GET'])
def weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    # return a harmless placeholder; frontend can be updated to call a real weather API later
    return jsonify({
        'lat': lat,
        'lon': lon,
        'weather': None,
        'note': 'placeholder - real weather service not configured'
    }), 200


def _is_port_free(host: str, port: int) -> bool:
    """Return True if the port can be bound (i.e. appears free)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        try:
            s.close()
        except Exception:
            pass
        return False


# added imports for the CLI check
import urllib.request
import json

# Replace the existing __main__ block with a CLI-check-aware block.
# If run as: python app.py check  -> probe /health and exit
if __name__ == '__main__':
    # quick CLI health-check mode
    if 'check' in sys.argv:
        HOST = os.environ.get('CROP_HOST', '127.0.0.1')
        PORT = os.environ.get('CROP_PORT', '5001')
        url = f'http://{HOST}:{PORT}/health'
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode('utf-8')
                print(body)
                sys.exit(0)
        except Exception as e:
            print('ERROR contacting', url, '-', str(e))
            sys.exit(2)

    # normal server start (unchanged behavior)
    HOST = os.environ.get('CROP_HOST', '0.0.0.0')
    PORT = int(os.environ.get('CROP_PORT', '5001'))
    port_free = _is_port_free(HOST, PORT)
    logging.info("PID=%s; port %s:%d free=%s", os.getpid(), HOST, PORT, port_free)
    if not port_free:
        logging.warning("Port %d appears to be in use — stop the conflicting process or change PORT before running.", PORT)
    logging.info("Starting Flask app on http://%s:%d (debug=False, reloader disabled)", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
