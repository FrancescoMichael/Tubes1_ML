# Feedforward Neural Network (FFNN) Implementation from Scratch

## Overview
This project implements a **Feedforward Neural Network (FFNN) from scratch** as part of the IF3270 Machine Learning course. The implementation is tested on the MNIST dataset to demonstrate handwritten digit recognition capabilities without using high-level deep learning frameworks.

## Objectives
- Implement a fully connected Feedforward Neural Network (FFNN) from scratch.
- Train and evaluate the model on a given dataset.
- Compare performance with scikit-learn's `MLPClassifier`.
- Visualize the neural network architecture and training progress.

## Dataset: MNIST
The MNIST database (Modified National Institute of Standards and Technology database) contains:
- 70,000 handwritten digit images (0-9)
- 60,000 training samples and 10,000 test samples
- 28×28 pixel grayscale images
- One of the most benchmark datasets in machine learning

## Requirements
### Libraries
- Python 3.x
- NumPy
- scikit-learn
- Matplotlib
- Pandas
- NetworkX (for visualization)

### Installation
```bash
pip install numpy scikit-learn matplotlib pandas networkx
```

## Project Structure
- `ann.py`: Contains the custom implementation of the FFNN.
- `visualizer.py`: Provides visualization tools for the neural network.
- `main.ipynb`: Jupyter notebook for running experiments, comparing results, and visualizing outputs.

## Usage
1. Open and run `main.ipynb` to execute the FFNN implementation.
2. The notebook includes:
   - Data loading and preprocessing.
   - Training the custom FFNN.
   - Comparison with scikit-learn's `MLPClassifier`.
   - Visualization of the neural network architecture.

## Group Members
| NIM      | Name                     | Contribution                     |
|----------|--------------------------|---------------------------------------|
| 13522002 | Ariel Herfrison          |         |
| 13522024 | Kristo Anugrah           |          |
| 13522038 | Francesco Michael Kusuma  |  |