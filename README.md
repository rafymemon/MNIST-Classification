# MNIST-Classification
MNIST Digit Classification using Keras
This project demonstrates a simple neural network implemented with Keras to classify handwritten digits from the MNIST dataset.

Project Overview
This repository contains a Jupyter Notebook (Colab notebook) that walks through the process of building and training a basic feedforward neural network to classify digits (0-9) from the MNIST dataset. The steps include data loading, preprocessing, model definition, training, evaluation, and making predictions on individual test images.

Technologies Used
Python 3.x
TensorFlow / Keras
NumPy
Matplotlib
Scikit-learn
Setup
To run this notebook, you'll need to have Python installed along with the required libraries. You can install them using pip:

pip install tensorflow numpy matplotlib scikit-learn

If you are using Google Colab, most of these libraries are pre-installed. You just need to run the cells sequentially.


Model Architecture
The neural network consists of a Sequential model with the following layers:

Flatten Layer: Converts the 28x28 pixel input images into a 1D array of 784 pixels.
Dense Layer (128 neurons): A fully connected hidden layer with ReLU activation.
Dense Layer (32 neurons): Another fully connected hidden layer with ReLU activation.
Dense Layer (10 neurons): The output layer with Softmax activation, corresponding to the 10 possible digit classes (0-9).
The model is compiled with sparse_categorical_crossentropy as the loss function and Adam optimizer, tracking accuracy as a metric.

Results
After training for 25 epochs, the model achieved the following performance:

Training Accuracy: Approximately 97.00%
Validation Accuracy: Approximately 96.22%
Test Accuracy: 96.38%
Visualizations
The notebook includes plots to visualize:

Training & Validation Loss: Shows how the loss decreased over epochs for both training and validation sets.
Training & Validation Accuracy: Illustrates the improvement in accuracy over epochs for both training and validation sets.
Sample Image Predictions: Displays a few test images and their corresponding predicted labels by the trained model.
