import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import numpy as np
import tensorflow as tf
import cv2


model_path = "models/covid19_detector.h5"

def load_image(file_path):
    try:
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        messagebox.showerror("Error", f"Error loading image: {e}")
        return None

def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    image = load_image(image_path)
    if image is not None:
        prediction = model.predict(image)
        class_idx = np.argmax(prediction, axis=1)[0]
        classes = ["COVID-19", "NORMAL", "PNEUMONIA"]
        return classes[class_idx]
    else:
        return None

def browse_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(model_path, file_path)
        if result:
            result_label.config(text=f"Prediction: {result}")
        else:
            result_label.config(text="Error in prediction")

def test_accuracy():
    result = subprocess.run(["python", "accuracy_test.py"], capture_output=True, text=True)
    if result.returncode == 0:
        accuracy_output = result.stdout.strip()
        accuracy_label.config(text=accuracy_output)
    else:
        messagebox.showerror("Error", f"Error in accuracy test: {result.stderr}")


root = tk.Tk()
root.title("COVID-19 Detection")


root.geometry("400x400")
root.configure(bg="#f0f0f0")


title_label = tk.Label(root, text="COVID-19 Detection", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
title_label.pack(pady=20)


frame = tk.Frame(root, padx=10, pady=10, bg="#f0f0f0")
frame.pack(pady=10)


browse_button = tk.Button(frame, text="Browse Image", command=browse_image, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
browse_button.pack(pady=10)


result_label_frame = tk.Frame(root, bg="#f0f0f0")
result_label_frame.pack(pady=10)
result_label = tk.Label(result_label_frame, text="Prediction will be shown here", font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
result_label.pack(pady=10)


accuracy_button = tk.Button(frame, text="Test Accuracy", command=test_accuracy, font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
accuracy_button.pack(pady=10)


accuracy_label_frame = tk.Frame(root, bg="#f0f0f0")
accuracy_label_frame.pack(pady=10)
accuracy_label = tk.Label(accuracy_label_frame, text="Accuracy will be shown here", font=("Helvetica", 12), bg="#f0f0f0", fg="#333")
accuracy_label.pack(pady=10)

root.mainloop()
