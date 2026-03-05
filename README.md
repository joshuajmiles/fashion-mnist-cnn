# Fashion MNIST CNN Image Classifier

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify clothing images from the Fashion-MNIST dataset.

## **Overview** 

Fashion-MNIST is a dataset of 70,000 grayscale images of clothing items across 10 categories including shirts, sneakers, bags, and coats.

This project demonstrates how to:

- Load and preprocess image datasets
- Build a convolutional neural network in PyTorch
- Train and evaluate a model
- Perform image classification

## **Technologies Used**

- Python
- PyTorch
- Torchvision
- Matplotlib

## **Model Architecture**

The CNN includes:

- Two convolutional layers
- ReLU activation functions
- Max pooling layers
- Fully connected layers for classification

## **Results** 

The model achieves approximately **~90% accuracy** on the test dataset.

## **How to Run**

Install dependencies:

pip install torch torchvision matplotlib

Run the training script:

python cnn_fashion_mnist.py
