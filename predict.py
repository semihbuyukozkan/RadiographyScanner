import numpy as np
import tensorflow as tf
import cv2

def load_image(file_path):
    try:
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    image = load_image(image_path)
    if image is not None:
        prediction = model.predict(image)
        class_idx = np.argmax(prediction, axis=1)[0]
        classes = ["COVID-19", "NORMAL", "PNEUMONIA"]
        print(f"Prediction: {classes[class_idx]}")
    else:
        print("Error: Could not load the image.")

if __name__ == "__main__":
    model_path = "models/covid19_detector.h5"
    image_path = "test_images/covid19_1.jpg"
    predict_image(model_path, image_path)
