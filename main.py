# main.py
import tensorflow as tf
import os
import subprocess

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPUs available: ", gpus)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs available. The code will run on CPU.")

def run_preprocessing():
    result = subprocess.run(["python", "preprocessing.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in preprocessing step:")
        print(result.stderr)
        exit(1)
    else:
        print("Preprocessing completed successfully.")

def run_model_training():
    result = subprocess.run(["python", "model.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in model training step:")
        print(result.stderr)
        exit(1)
    else:
        print("Model training completed successfully.")

def run_evaluation():
    result = subprocess.run(["python", "evaluate.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in evaluation step:")
        print(result.stderr)
        exit(1)
    else:
        print("Evaluation completed successfully.")

def run_prediction(image_path):
    result = subprocess.run(["python", "predict.py", "models/covid19_detector.h5", image_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in prediction step:")
        print(result.stderr)
        exit(1)
    else:
        print("Prediction completed successfully.")
        print(result.stdout)

if __name__ == "__main__":
    check_gpu()
    run_preprocessing()
    run_model_training()
    run_evaluation()
    run_prediction("test fotoğrafı dizini")
