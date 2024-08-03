import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_image(file_path):
    try:
        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (224, 224))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        return img_array
    except Exception as e:
        return None

def load_data(data_dir, max_files_per_category=1000):
    categories = ["COVID-19", "NORMAL", "PNEUMONIA"]
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        with ThreadPoolExecutor() as executor:
            files = os.listdir(path)[:max_files_per_category]
            futures = [executor.submit(load_image, os.path.join(path, img)) for img in files]
            for future in as_completed(futures):
                img_array = future.result()
                if img_array is not None:
                    data.append(img_array)
                    labels.append(class_num)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)

def preprocess_data(data, labels):
    data = data / 255.0
    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    data, labels = load_data("data", max_files_per_category=1000)
    X_train, X_test, y_train, y_test = preprocess_data(data, labels)
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)