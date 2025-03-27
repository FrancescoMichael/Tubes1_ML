from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from ann import ANNScratch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def load_data():
    data = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
    X, y = data
    X = X / 255.0

    df = pd.DataFrame(X)
    df.insert(0, "label", y)  # Menambahkan label sebagai kolom pertama
    
    df.to_csv("mnist.csv", index=False)
    print("Dataset telah disimpan sebagai 'mnist.csv'.")
    return X, y

def load_mnist_from_csv():
    print("Loading...")
    # df = pd.read_csv("mnist.csv")
    df = pd.read_csv("train.csv")
    y = df["price_range"].values
    X = df.drop(columns=["price_range"]).values
    return X, y

def preprocess_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_sklearn_mlp(config, X_train, X_test, y_train, y_test):

    reg_lambda = 1e-4
    if config['regularization'] == 'l2':
        reg_lambda = config['reg_lambda']

    mlp = MLPClassifier(hidden_layer_sizes=config['neurons'], max_iter=config['epochs'], alpha=reg_lambda,
                        solver='sgd', verbose=10, random_state=42, learning_rate_init=config['learning_rate'])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    plt.figure(2)
    print(mlp.loss_curve_)
    plt.plot([i+1 for i in range(len(mlp.loss_curve_))], mlp.loss_curve_)
    plt.draw()

def get_user_model_config(input_dim):
    n_layer = int(input("Jumlah layer: "))
    
    n_neurons = [input_dim]
    activations = []
    
    activation_options = {
        "1": "linear", "2": "relu", "3": "sigmoid", "4": "tanh", 
        "5": "softmax", "6": "softplus", "7": "leaky_relu", "8": "mish"
    }
    
    for i in range(n_layer):
        n_neurons.append(int(input(f"Jumlah neuron layer-{i+1}: ")))
        act_choice = input("1. Linear\n2. ReLU\n3. Sigmoid\n4. Tanh\n5. Softmax\n6. Softplus\n7. Leaky ReLU\n8. Mish\nFungsi aktivasi: ")
        activations.append(activation_options.get(act_choice, "relu"))

    act_choice = input("1. Linear\n2. ReLU\n3. Sigmoid\n4. Tanh\n5. Softmax\n6. Softplus\n7. Leaky ReLU\n8. Mish\nFungsi aktivasi output layer: ")
    activations.append(activation_options.get(act_choice, "relu"))
    
    loss_options = {"1": "mse", "2": "binary_cross_entropy", "3": "categorical_cross_entropy"}
    loss = loss_options.get(input("1. MSE\n2. Binary Cross-Entropy\n3. Categorical Cross-Entropy\nFungsi loss: "), "mse")
    
    batch_size = int(input("Batch size: "))
    learning_rate = float(input("Learning rate: "))
    n_epoch = int(input("Jumlah epoch: "))
    verbose = bool(int(input("Verbose (0/1): ")))
    
    regularization = input("Regularization (None/L1/L2): ").lower()
    reg_lambda = float(input("Regularization Lambda: ")) if regularization in ["l1", "l2"] else 0
    
    initialization_options = {"1": "zero", "2": "uniform", "3": "normal", "4": "xavier", "5": "he"}
    initialization = initialization_options.get(input("1. Zero\n2. Uniform\n3. Normal\n4. Xavier\n5. He\nInitialisasi: "), "xavier")
    
    return {
        "neurons": n_neurons, "activations": activations, "epochs": n_epoch, "loss": loss,
        "learning_rate": learning_rate, "batch_size": batch_size, "verbose": verbose,
        "regularization": regularization, "reg_lambda": reg_lambda, "initialization": initialization
    }

def train_custom_ann(config, X, y):
    model = ANNScratch(**config)
    model.fit(X, y)

if __name__ == "__main__":
    # X, y = load_data()
    X, y = load_mnist_from_csv()
    print("X: \n", X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = preprocess_data(X, y_encoded)
    
    # X_train = X
    # y_train = y

    print("X train: ", X_train.shape)
    print("y train: ", y_train.shape)

    config = get_user_model_config(X.shape[1])

    # MLP
    print("Training scikit-learn MLP...")
    train_sklearn_mlp(config, X_train, X_test, y_train, y_test)
    
    # FFNN
    # X_train, y_train = np.array([[0.05, 0.1]]), np.array([[0.01, 0.99]])
    print("\nTraining custom ANN...")
    train_custom_ann(config, X_train, y_train)

    plt.show()

    time.sleep(10)