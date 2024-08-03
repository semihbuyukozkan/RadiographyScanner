# RadiographyScanner

RadiographyScanner is a machine learning project designed to detect COVID-19, NORMAL, and PNEUMONIA cases from radiography images using a Convolutional Neural Network (CNN) model. The project leverages transfer learning with the VGG16 architecture to achieve high accuracy in image classification tasks.

## Features
- Load and preprocess radiography images.
- Train a CNN model using transfer learning with VGG16.
- Evaluate model performance on test data.
- Predict the class of new radiography images through a user-friendly GUI.

## Dataset
The dataset used in this project is the COVID-19 Radiography Database, which contains radiography images categorized into three classes: COVID-19, NORMAL, and PNEUMONIA. The dataset is publicly available and can be accessed [here](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).

## Technologies and Libraries Used
- **Python**: The primary programming language used for the project.
- **TensorFlow**: A powerful library for building and training deep learning models.
- **Keras**: An API within TensorFlow used for building and training neural networks.
- **OpenCV**: A library for image processing tasks.
- **NumPy**: A library for numerical computations.
- **Matplotlib**: A library for plotting graphs and visualizations.
- **Seaborn**: A library for statistical data visualization.
- **Scikit-learn**: A library for machine learning and evaluation metrics.
- **tkinter**: A standard Python library for creating graphical user interfaces.

## Techniques Used
- **Transfer Learning**: Utilized the VGG16 architecture pre-trained on the ImageNet dataset. Transfer learning allows leveraging the learned features from a large dataset to improve the performance on our specific task.
- **Data Preprocessing**: Involved resizing images, normalizing pixel values, and converting grayscale images to RGB.
- **Model Evaluation**: Used confusion matrices and classification reports to evaluate the model's performance on the test set.
- **GUI for Prediction**: Implemented a user-friendly GUI using tkinter to allow users to upload radiography images and view prediction results.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/RadiographyScanner.git
   cd RadiographyScanner

2. Download the model and data files from Google Drive and add them to the project directory. ([google drive link:https://drive.google.com/drive/u/1/folders/13gI-HLGJKKXfMI-BNfD0QVouSJ2YEj_F](https://drive.google.com/drive/folders/13gI-HLGJKKXfMI-BNfD0QVouSJ2YEj_F?usp=sharing))
3. Install required packages from requirements.txt
4. run gui.py
