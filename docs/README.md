
# Mathematical Symbol Recognition Using CNN on FPGA

## 1. Project Overview

This project implements a Convolutional Neural Network (CNN)
for recognizing handwritten mathematical symbols using
Intel Cyclone V FPGA.

The system classifies input images into five mathematical
symbol categories using a CNN model.

The CNN algorithm is developed and trained using Python.
The trained model parameters are then used for hardware
inference implementation using Verilog HDL.

## 2. Objectives

- Develop a CNN model for mathematical symbol recognition.
- Generate and prepare a mathematical symbol dataset.
- Train and evaluate the CNN model using Python.
- Implement CNN inference using Verilog HDL.
- Simulate and evaluate the hardware implementation.
- Target the Intel Cyclone V FPGA platform.

## 3. Symbol Classes

The system is designed to recognize five classes:

| Class | Symbol | Description |
|---|---|---|
| 0 | + | Addition |
| 1 | - | Subtraction |
| 2 | x | Multiplication |
| 3 | / | Division |
| 4 | = | Equal |

## 4. Dataset Specification

| Parameter | Specification |
|---|---|
| Image Size | 32 x 32 pixels |
| Color Format | Grayscale |
| Pixel Depth | 8-bit |
| Number of Classes | 5 |
| Dataset Source | Generated using Python |

The dataset will contain training and testing images
for each mathematical symbol class.

## 5. CNN Architecture

The proposed CNN architecture consists of:

1. Input Image (32 x 32 x 1)
2. Convolution Layer 1 (3 x 3, 4 filters)
3. ReLU Activation
4. Max Pooling (2 x 2)
5. Convolution Layer 2 (3 x 3, 8 filters)
6. ReLU Activation
7. Max Pooling (2 x 2)
8. Flatten Layer (288 features)
9. Fully Connected Layer (5 outputs)

## 6. Development Tools

- Python
- NumPy
- Matplotlib
- PyTorch
- Verilog HDL
- ModelSim
- Intel Quartus Prime

## 7. Project Workflow

1. Dataset generation
2. Image preprocessing
3. CNN model development
4. Model training and evaluation
5. Weight and bias extraction
6. Verilog hardware implementation
7. Hardware simulation
8. FPGA compilation and evaluation

## 8. Repository Structure

```text
CNN-FPGA-Math-Symbol-Recognition/
|
|-- dataset/
|-- algorithm/
|-- verilog/
|-- quartus/
|-- docs/
|-- README.md
```

## 9. Project Status

Currently in the initial development stage.

Dataset generation, CNN development, and hardware
implementation are planned as the next development steps.

## 10. Author

Electrical Engineering Student

CNN FPGA Project - 2026
