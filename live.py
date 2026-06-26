import cv2
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

print("Loading MobileNetV2...")
model = MobileNetV2(weights="imagenet")
print("Model Loaded!")

# Use your Iriun camera index
cap = cv2.VideoCapture(1)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run prediction every 20 frames
    if frame_count % 20 == 0:

        img = cv2.resize(frame, (224, 224))

        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)

        result = decode_predictions(preds, top=1)[0][0]

        label = result[1]
        confidence = result[2] * 100

        text = f"{label}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Edge AI Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()