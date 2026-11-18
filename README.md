# CNN-Based Mathematical Symbol Recognition on FPGA

A Convolutional Neural Network (CNN) project for recognizing mathematical symbols from synthetic image data, with the objective of implementing the trained model on an Intel Cyclone V FPGA.

## Project Overview

This project focuses on developing a CNN-based mathematical symbol recognition system. The model is trained using synthetically generated images and designed to classify five mathematical symbols.

The initial development is carried out using Python and PyTorch. The trained model is then exported and quantized to prepare its parameters for hardware implementation using Verilog HDL.

The long-term goal is to perform CNN inference on an FPGA platform.

## Objectives

- Develop a CNN model for mathematical symbol recognition.
- Generate a synthetic dataset containing five mathematical symbol classes.
- Train and evaluate the CNN model using PyTorch.
- Export trained model parameters.
- Apply fixed-point quantization to the model weights and biases.
- Implement the CNN inference architecture using Verilog HDL.
- Integrate and test the design on an Intel Cyclone V FPGA.

## Recognized Symbols

The model currently supports five mathematical symbol classes:

| Class | Symbol |
|---|---|
| Addition | + |
| Subtraction | − |
| Multiplication | × |
| Division | ÷ |
| Equal | = |

## Dataset

The dataset is synthetically generated using Python image processing tools.

**Dataset specifications:**

- Image dimensions: 32 × 32 pixels
- Image format: Grayscale
- Number of classes: 5
- Training images: 400 images per class
- Testing images: 100 images per class
- Total images: 2,500

| Dataset | Images per Class | Total Images |
|---|---:|---:|
| Training | 400 | 2,000 |
| Testing | 100 | 500 |
| **Total** | **500** | **2,500** |

The dataset is divided into separate training and testing directories.

## CNN Architecture

The CNN model consists of two convolutional layers, ReLU activation functions, two max-pooling layers, and one fully connected layer.

| Layer | Configuration | Output Shape |
|---|---|---|
| Input | Grayscale image | 32 × 32 × 1 |
| Conv1 | 4 filters, 3 × 3 | 30 × 30 × 4 |
| ReLU | Activation | 30 × 30 × 4 |
| MaxPool1 | 2 × 2 | 15 × 15 × 4 |
| Conv2 | 8 filters, 3 × 3 | 13 × 13 × 8 |
| ReLU | Activation | 13 × 13 × 8 |
| MaxPool2 | 2 × 2 | 6 × 6 × 8 |
| Flatten | 288 features | 288 |
| Fully Connected | 288 → 5 | 5 |

## Model Training and Evaluation

The CNN model is developed and trained using Python and PyTorch.

**Training configuration:**

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32
- Training epochs: 20
- Loss function: Cross Entropy Loss

**Evaluation results:**

| Metric | Result |
|---|---:|
| Total test images | 500 |
| Correct predictions | 488 |
| Incorrect predictions | 12 |
| Test accuracy | **97.60%** |

The evaluation results represent the software model performance on the generated synthetic test dataset.

## Model Quantization

The trained model parameters are exported and quantized using a Q8.8 fixed-point