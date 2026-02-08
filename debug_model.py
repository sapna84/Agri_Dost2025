"""Diagnostic script to load the Keras model and print raw outputs for images.

Usage:
  .venv311\Scripts\python.exe debug_model.py uploads\img1.jpg test_case\img1.jpg
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print('Usage: debug_model.py <image1> [image2 ...]')
        return

    imgs = sys.argv[1:]

    try:
        import tensorflow as tf
    except Exception as e:
        print('ERROR: TensorFlow not available:', e)
        return

    try:
        import numpy as np
    except Exception as e:
        print('ERROR: numpy not available:', e)
        return

    try:
        import cv2
    except Exception:
        cv2 = None

    # Try to find model in backend/ first, then repo root
    model_candidates = [os.path.join('backend', 'model.h5'), 'model.h5']
    model_path = None
    for p in model_candidates:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print('No model file found. Looked for:', model_candidates)
        return

    print('Loading model from:', model_path)
    model = tf.keras.models.load_model(model_path)
    print('\nModel summary:')
    try:
        model.summary()
    except Exception:
        print('(model.summary() failed)')

    # Get input shape
    try:
        input_shape = model.input_shape
    except Exception:
        input_shape = None
    print('Detected input_shape =', input_shape)

    # Determine expected target size (height, width)
    target_h = target_w = None
    if input_shape and len(input_shape) >= 3:
        # Common format: (None, H, W, C)
        if input_shape[1] is not None and input_shape[2] is not None:
            target_h = int(input_shape[1]); target_w = int(input_shape[2])

    if target_h is None:
        # default fallback
        target_h, target_w = 150, 150

    print(f'Using target size: ({target_h},{target_w})')

    for img_path in imgs:
        print('\n---')
        print('Image:', img_path)
        if not os.path.exists(img_path):
            print('  NOT FOUND')
            continue

        # Load image
        if cv2 is not None:
            img = cv2.imread(img_path)
            if img is None:
                print('  cv2 failed to read')
                continue
            # convert BGR->RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (target_w, target_h))
            x = img_resized.astype('float32') / 255.0
        else:
            from PIL import Image
            im = Image.open(img_path).convert('RGB')
            im = im.resize((target_w, target_h))
            x = np.array(im).astype('float32') / 255.0

        # Add batch dim
        x = np.expand_dims(x, axis=0)
        print('  input array shape:', x.shape, 'dtype:', x.dtype)

        # Run prediction
        pred = model.predict(x)
        print('  raw model output shape:', getattr(pred, 'shape', type(pred)))
        print('  raw model output (first item):', pred[0])

        # Interpret common cases
        try:
            if pred.shape[-1] == 1:
                score = float(pred[0][0])
                print('  interpreted score (sigmoid/logit->prob):', score)
                print('  decision (score>=0.5):', 'Disease' if score >= 0.5 else 'No Disease')
            else:
                import numpy as _np
                cls = int(_np.argmax(pred, axis=1)[0])
                print('  predicted class index:', cls)
        except Exception as e:
            print('  could not interpret output:', e)

if __name__ == '__main__':
    main()
