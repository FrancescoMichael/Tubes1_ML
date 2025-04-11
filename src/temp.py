import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

def save_mnist_to_csv():
    # Load MNIST dataset from TensorFlow
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Flatten 28x28 images into 784 features & normalize pixel values
    X = np.vstack([X_train, X_test]).reshape(-1, 28*28) / 255.0
    y = np.concatenate([y_train, y_test])

    # Convert to pandas DataFrame
    df = pd.DataFrame(X)
    df.insert(0, "label", y)  # Add labels as first column

    # Save as CSV file
    df.to_csv("mnist.csv", index=False)
    print("MNIST dataset saved as 'mnist.csv'.")

save_mnist_to_csv()
