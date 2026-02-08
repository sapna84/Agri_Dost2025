from backend.app import tranfomation
import cv2

paths = [r'uploads\img1.jpg', r'test_case\img1.jpg']
for p in paths:
    print('\n===', p)
    img = cv2.imread(p)
    if img is None:
        print('Could not read', p)
        continue
    try:
        res, score = tranfomation(img)
        print('Result:', res, 'Score:', score)
    except Exception as e:
        print('Error during tranfomation:', e)

# Inspect penultimate outputs if available
import backend.app as b
if b.penultimate_model is not None:
    import numpy as np
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        resized = cv2.resize(img, (150,150))
        x = resized.astype('float32')/255.0
        x = np.expand_dims(x,0)
        pen = b.penultimate_model.predict(x)
        print('penultimate shape for', p, pen.shape)
        print('penultimate (first 8):', pen.flatten()[:8])
