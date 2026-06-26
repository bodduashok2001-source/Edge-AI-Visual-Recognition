from tensorflow.keras.applications import MobileNetV2

print("Loading model...")

model = MobileNetV2(weights='imagenet')

print("Model loaded successfully")