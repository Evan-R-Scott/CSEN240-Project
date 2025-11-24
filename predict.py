import sys
import cv2
import numpy as np
import tensorflow as tf

CATEGORIES = ["Normal","Osteopenia", "Osteoporosis"]

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]

# Load model
model = tf.keras.models.load_model("trained_model.keras")

# Preprocess image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype("float32") / 255
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)
print("Prediction:", CATEGORIES[np.argmax(pred)])
