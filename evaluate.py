import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_data():
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    return X_test, y_test

def evaluate_model(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes))
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.show()

if __name__ == "__main__":
    X_test, y_test = load_data()
    evaluate_model('models/covid19_detector.h5', X_test, y_test)