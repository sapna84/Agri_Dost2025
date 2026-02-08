from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import time
import re
import html as _html

# Make heavy ML imports optional so the Flask app can start even when
# TensorFlow/numpy/opencv are not installed. The /predict endpoint will
# return a 503 until the model is available.
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

# Optional requests used by lightweight proxy endpoints
try:
    import requests
except Exception:
    requests = None

# Live market fetch configuration
LIVE_MARKET_ENABLED = True
MARKET_CACHE_TTL = 30 * 60  # seconds
_MARKET_CACHE = {'ts': 0, 'data': None}

# data.gov.in integration (configurable via env)
DATA_GOV_API_URL = os.getenv('DATA_GOV_API_URL')  # e.g. https://api.data.gov.in/resource/<id>
DATA_GOV_API_KEY = os.getenv('DATA_GOV_API_KEY')  # optional

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Allow frontend to connect

# Ensure uploads directory exists
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve frontend files
@app.route('/')
def index():
    return app.send_static_file('home.html')

@app.route('/<path:path>')
def serve_frontend(path):
    try:
        return app.send_static_file(path)
    except:
        return app.send_static_file('home.html')

# Load your ML model if TensorFlow is available. Otherwise leave mdl=None
mdl = None
if tf is not None:
    try:
        mdl = tf.keras.models.load_model('./model.h5')
        print("✅ Model loaded successfully!")
        try:
            print(mdl.summary())
        except Exception:
            pass
    except Exception as e:
        print("⚠️ Could not load model at startup (continuing without it):", e)

# Check for a problematic final-layer softmax with a single unit.
# If the model ends with softmax and output size == 1, softmax will always
# return 1.0. In that case we'll compute the sigmoid of the final linear
# output instead by using the penultimate layer output and the final
# Dense layer weights (W,b). This avoids retraining the model.
penultimate_model = None
final_W = None
final_b = None
use_sigmoid_fix = False
if mdl is not None:
    try:
        final_layer = mdl.layers[-1]
        out_units = final_layer.output_shape[-1]
        # Try multiple heuristics to detect a softmax-over-single-unit misconfiguration
        cfg = None
        try:
            cfg = final_layer.get_config()
        except Exception:
            cfg = {}
        cfg_act = cfg.get('activation')
        layer_class = final_layer.__class__.__name__.lower()
        is_softmax_like = False
        if cfg_act and 'softmax' in str(cfg_act).lower():
            is_softmax_like = True
        # fallback: check activation attribute or layer class name
        act = getattr(final_layer, 'activation', None)
        try:
            act_name = act.__name__
        except Exception:
            act_name = str(act).lower() if act is not None else None
        if act_name and 'softmax' in act_name:
            is_softmax_like = True
        if 'softmax' in layer_class:
            is_softmax_like = True

        print(f"Final layer class: {layer_class}, activation(config)={cfg_act}, activation(attr)={act_name}, out_units: {out_units}")
        if out_units == 1 and is_softmax_like:
            # build penultimate model
            penultimate_model = tf.keras.Model(inputs=mdl.input, outputs=mdl.layers[-2].output)
            final_W, final_b = final_layer.get_weights()
            use_sigmoid_fix = True
            print('⚠️ Detected final softmax with 1 unit — applying sigmoid-fix for predictions.')
    except Exception:
        pass


# Define the new dimensions
width = 150
height = 150
dimensions = (width, height)


def tranfomation(image):
    """Process image and return prediction result with score."""
    # Defensive: ensure required ML deps and model exist
    if cv2 is None or np is None or mdl is None:
        raise RuntimeError("ML dependencies or model not available")

    # Resize the image using bicubic interpolation
    resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_CUBIC)

    # normalizing the image and resizing it
    normalized_image = resized_image / 255.0
    normalized_image = np.expand_dims(normalized_image, axis=0)
    
    # Lazy-detect the softmax-with-single-unit issue at prediction time
    # (handles cases where layer metadata may not be available at import time)
    global penultimate_model, final_W, final_b, use_sigmoid_fix
    if not use_sigmoid_fix and penultimate_model is None and mdl is not None:
        try:
            final_layer = mdl.layers[-1]
            cfg = {}
            try:
                cfg = final_layer.get_config()
            except Exception:
                cfg = {}
            cfg_act = cfg.get('activation')
            layer_class = final_layer.__class__.__name__.lower()
            act = getattr(final_layer, 'activation', None)
            try:
                act_name = act.__name__
            except Exception:
                act_name = str(act).lower() if act is not None else None
            out_units = None
            try:
                out_units = final_layer.output_shape[-1]
            except Exception:
                # fallback: try to read units from layer config
                try:
                    out_units = int(cfg.get('units')) if cfg.get('units') is not None else None
                except Exception:
                    out_units = None
            is_softmax_like = False
            if cfg_act and 'softmax' in str(cfg_act).lower():
                is_softmax_like = True
            if act_name and 'softmax' in act_name:
                is_softmax_like = True
            if 'softmax' in layer_class:
                is_softmax_like = True
            print(f"[detection] Final layer class: {layer_class}, activation(config)={cfg_act}, activation(attr)={act_name}, out_units: {out_units}")
            if out_units == 1 and is_softmax_like:
                penultimate_model = tf.keras.Model(inputs=mdl.input, outputs=mdl.layers[-2].output)
                final_W, final_b = final_layer.get_weights()
                use_sigmoid_fix = True
                print('⚠️ Detected final softmax with 1 unit — applying sigmoid-fix for predictions.')
        except Exception as e:
            print('detection error:', e)

    # If we detected the softmax-with-single-unit issue, compute the
    # sigmoid of the final linear output using the penultimate outputs
    # and the final Dense weights. Otherwise, use the model's predict()
    if use_sigmoid_fix and penultimate_model is not None and final_W is not None:
        pen_out = penultimate_model.predict(normalized_image)
        # pen_out shape: (1, features), final_W shape: (features, 1)
        z = np.dot(pen_out, final_W) + final_b
        z = float(np.squeeze(z))
        pred_score = 1.0 / (1.0 + np.exp(-z))
        print(f"Raw linear output z: {z}")
        print(f"Sigmoid-corrected prediction score: {pred_score}")
    else:
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
        # If the model isn't loaded, return 503 so callers know predictions
        # are not available yet.
        if mdl is None:
            return jsonify({'error': 'model not available'}), 503
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


# Simple backend endpoint that proxies Open-Meteo current weather (no API key required).
@app.route('/api/weather', methods=['GET'])
def api_weather():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    if not lat or not lon:
        return jsonify({'error': 'lat and lon query parameters required (e.g. ?lat=26.9&lon=75.8)'}), 400
    if requests is None:
        return jsonify({'error': 'server missing requests library'}), 503
    try:
        url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto'
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        cur = data.get('current_weather', {})
        simplified = {
            'temperature': cur.get('temperature'),
            'windspeed': cur.get('windspeed'),
            'winddirection': cur.get('winddirection'),
            'weathercode': cur.get('weathercode'),
            'time': cur.get('time')
        }
        return jsonify({'source': 'open-meteo', 'data': simplified})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Simple mock market prices endpoint. Returns filtered sample data.
SAMPLE_MARKET_DATA = [
    { 'crop':'Wheat', 'state':'Punjab', 'market':'Ludhiana', 'price':2300, 'change':'+1.2%', 'date':'Today' },
    { 'crop':'Wheat', 'state':'Haryana', 'market':'Hisar', 'price':2250, 'change':'+0.8%', 'date':'Today' },
    { 'crop':'Rice', 'state':'West Bengal', 'market':'Kolkata', 'price':3050, 'change':'+1.0%', 'date':'Today' },
    { 'crop':'Onion', 'state':'Maharashtra', 'market':'Nashik', 'price':2000, 'change':'+3.2%', 'date':'Today' },
    { 'crop':'Potato', 'state':'Uttar Pradesh', 'market':'Kanpur', 'price':1100, 'change':'+0.5%', 'date':'Today' }
]


@app.route('/api/market-prices', methods=['GET'])
def api_market_prices():
    crop = (request.args.get('crop') or '').strip()
    state = (request.args.get('state') or '').strip()
    # simple case-insensitive filter
    def match(item):
        ok = True
        if crop:
            ok = ok and item['crop'].lower() == crop.lower()
        if state:
            ok = ok and item['state'].lower() == state.lower()
        return ok

    filtered = [i for i in SAMPLE_MARKET_DATA if match(i)]
    return jsonify({'data': filtered, 'count': len(filtered)})


def _parse_first_table(html_text):
    """Naive HTML table parser: returns list of rows (each a list of cell texts).
    This is intentionally small and forgiving; it's a fallback parser for Agmarknet pages.
    """
    m = re.search(r"<table[\s\S]*?</table>", html_text, re.IGNORECASE)
    if not m:
        return []
    table = m.group(0)
    rows = re.findall(r"<tr[\s\S]*?</tr>", table, re.IGNORECASE)
    result = []
    for r in rows:
        cols = re.findall(r"<t[dh][\s\S]*?</t[dh]>", r, re.IGNORECASE)
        cells = []
        for c in cols:
            # strip tags
            txt = re.sub(r"<.*?>", "", c, flags=re.S).strip()
            txt = _html.unescape(txt)
            if txt:
                cells.append(txt)
        if cells:
            result.append(cells)
    return result


def fetch_agmarknet_prices():
    """Attempt to fetch and parse Agmarknet latest price table.
    Returns list of dicts with keys roughly matching our SAMPLE_MARKET_DATA.
    If anything fails, raises Exception.
    NOTE: Agmarknet does not provide a stable JSON API here; this is a best-effort scraper.
    """
    if requests is None:
        raise RuntimeError('requests library not available')

    url = 'https://agmarknet.gov.in/PriceAndArrivals/PriceListByCommodity.aspx'
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    tbls = _parse_first_table(resp.text)
    if not tbls or len(tbls) < 2:
        raise RuntimeError('no table found on agmarknet page')

    header = tbls[0]
    rows = tbls[1:]
    out = []
    # Try to map common columns heuristically
    for r in rows:
        try:
            # common Agmarknet tables often have columns: Commodity, Market, Min, Max, Modal Price, Date
            # We'll try to fill our schema: crop, state (unknown), market, price, change, date
            crop = r[0] if len(r) > 0 else ''
            market = r[1] if len(r) > 1 else ''
            # try to find a numeric price in later columns
            price = None
            for cell in r[2:]:
                num = re.sub(r'[^0-9.]', '', cell)
                if num:
                    try:
                        price = int(float(num))
                        break
                    except Exception:
                        continue
            date = r[-1] if len(r) >= 1 else ''
            out.append({'crop': crop or 'Unknown', 'state': '', 'market': market or 'Unknown', 'price': price or 0, 'change': '', 'date': date})
        except Exception:
            continue
    if not out:
        raise RuntimeError('parsed table but no usable rows')
    return out


@app.route('/api/market-prices/live', methods=['GET'])
def api_market_prices_live():
    """Return live market prices (best-effort scraping of Agmarknet). Falls back to sample data.
    Query params: crop, state
    """
    crop = (request.args.get('crop') or '').strip()
    state = (request.args.get('state') or '').strip()

    # cache
    now = time.time()
    if _MARKET_CACHE['data'] and (now - _MARKET_CACHE['ts'] < MARKET_CACHE_TTL):
        data = _MARKET_CACHE['data']
    else:
        try:
            if LIVE_MARKET_ENABLED:
                data = fetch_agmarknet_prices()
            else:
                data = SAMPLE_MARKET_DATA
        except Exception as e:
            # fallback
            data = SAMPLE_MARKET_DATA
        _MARKET_CACHE['data'] = data
        _MARKET_CACHE['ts'] = now

    if crop:
        data = [d for d in data if d.get('crop','').lower() == crop.lower()]
    if state:
        data = [d for d in data if d.get('state','').lower() == state.lower()]

    return jsonify({'data': data, 'count': len(data), 'source': 'agmarknet' if _MARKET_CACHE['data'] and _MARKET_CACHE['data'] is not SAMPLE_MARKET_DATA else 'sample'})


def fetch_data_gov_prices(params=None):
    """Fetch from a configured data.gov.in API endpoint. The exact JSON schema
    varies per dataset; this function attempts to normalize to our schema.
    Requires DATA_GOV_API_URL to be set in environment.
    """
    if not DATA_GOV_API_URL:
        raise RuntimeError('DATA_GOV_API_URL not configured')
    if requests is None:
        raise RuntimeError('requests library not available')

    # build params
    query = {}
    if DATA_GOV_API_KEY:
        # some data.gov.in APIs accept 'api-key' param
        query['api-key'] = DATA_GOV_API_KEY
    if params:
        query.update(params)

    resp = requests.get(DATA_GOV_API_URL, params=query, timeout=10)
    resp.raise_for_status()
    j = resp.json()
    # data.gov.in common response shape: { 'records': [...] } or 'result': {...}
    records = None
    if isinstance(j, dict):
        records = j.get('records') or j.get('result') or j.get('data') or j.get('results')
    if records is None:
        # try to interpret top-level list
        if isinstance(j, list):
            records = j
    if not records:
        raise RuntimeError('no records found in data.gov response')

    out = []
    # try to map common fields
    for r in records:
        # try several field names for commodity, market, price, state, date
        crop = r.get('commodity') or r.get('commodity_name') or r.get('commodity_name_en') or r.get('crop') or r.get('Commodity') or ''
        market = r.get('market') or r.get('market_name') or r.get('Market') or ''
        state = r.get('state') or r.get('state_name') or r.get('State') or ''
        price = None
        for key in ['modal_price', 'price', 'min_price', 'max_price', 'modal_price_per_qtl', 'price_per_qtl']:
            if key in r and r.get(key) is not None:
                try:
                    price = int(float(r.get(key)))
                    break
                except Exception:
                    continue
        date = r.get('date') or r.get('arrival_date') or r.get('report_date') or ''
        out.append({'crop': crop or 'Unknown', 'state': state or '', 'market': market or '', 'price': price or 0, 'change': '', 'date': date})

    if not out:
        raise RuntimeError('no usable records parsed from data.gov response')
    return out


@app.route('/api/market-prices/data', methods=['GET'])
def api_market_prices_data():
    """Proxy to a configured data.gov.in dataset. If DATA_GOV_API_URL is not set
    this returns the sample data with a note. Query params (crop/state) are forwarded.
    """
    crop = (request.args.get('crop') or '').strip()
    state = (request.args.get('state') or '').strip()

    if not DATA_GOV_API_URL:
        # not configured: return fallback
        data = SAMPLE_MARKET_DATA
        if crop:
            data = [d for d in data if d.get('crop','').lower() == crop.lower()]
        if state:
            data = [d for d in data if d.get('state','').lower() == state.lower()]
        return jsonify({'data': data, 'count': len(data), 'source': 'sample', 'note': 'DATA_GOV_API_URL not configured'})

    # build params to forward
    params = {}
    if crop:
        params['commodity'] = crop
    if state:
        params['state'] = state

    try:
        data = fetch_data_gov_prices(params=params)
    except Exception as e:
        # fallback
        data = SAMPLE_MARKET_DATA
        return jsonify({'data': data, 'count': len(data), 'source': 'sample', 'error': str(e)})

    return jsonify({'data': data, 'count': len(data), 'source': 'data.gov'})


@app.route('/status', methods=['GET'])
def status():
    """Return model readiness and basic info for frontend checks."""
    info = {
        'model_loaded': mdl is not None,
        'use_sigmoid_fix': bool(use_sigmoid_fix),
        'input_shape': None,
    }
    try:
        if mdl is not None:
            info['input_shape'] = getattr(mdl, 'input_shape', None)
    except Exception:
        info['input_shape'] = None
    return jsonify(info)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
