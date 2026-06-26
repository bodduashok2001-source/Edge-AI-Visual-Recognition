import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

# Load model
model = MobileNetV2(weights="imagenet")

# Load image
img = image.load_img("test.jpg", target_size=(224, 224))

# Convert to array
x = image.img_to_array(img)

# Add batch dimension
x = np.expand_dims(x, axis=0)

# Preprocess
x = preprocess_input(x)

# Predict
preds = model.predict(x)

# Decode
results = decode_predictions(preds, top=3)[0]

for _, label, confidence in results:
    print(f"{label}: {confidence:.4f}")