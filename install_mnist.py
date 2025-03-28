import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

def save_mnist_to_csv(filename="mnist.csv"):
    """Load MNIST, normalize, encode labels, and save as CSV"""
    print("Downloading MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
    
    # Normalize pixel values
    X = X / 255.0  
    
    # Convert to DataFrame
    df = pd.DataFrame(X)
    df.insert(0, "label", y)  # Insert labels as the first column

    # Save as CSV
    df.to_csv(filename, index=False)
    print(f"MNIST dataset saved as {filename}")

# Call the function to save the file
save_mnist_to_csv()
