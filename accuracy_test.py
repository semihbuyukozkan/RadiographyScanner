import numpy as np
import tensorflow as tf

def load_data():
    X_test = np.load('models/X_test.npy')
    y_test = np.load('models/y_test.npy')
    return X_test, y_test

def evaluate_accuracy(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy

if __name__ == "__main__":
    X_test, y_test = load_data()
    model_path = 'models/covid19_detector.h5'
    accuracy = evaluate_accuracy(model_path, X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")