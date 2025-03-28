from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from ann import ANNScratch
from visualizer import ANNVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback

def load_data():
    """Load MNIST data and normalize pixel values"""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
    return X / 255.0, y

def load_mnist_from_csv():
    print("Loading...")
    df = pd.read_csv("mnist.csv")
    y = df["label"].values

    y_transformed = np.zeros((len(y), len(np.unique(y))))
    print(y_transformed)
    labels = y - 1
    for i, label in enumerate(labels):
        y_transformed[i, :] = 0 
        y_transformed[i, label] = 1 

    X = df.drop(columns=["label"]).values
    
    return X, y_transformed

def preprocess_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_sklearn_mlp(config, X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(
        hidden_layer_sizes=config['neurons'],
        max_iter=config['epochs'],
        alpha=config['reg_lambda'] if config['regularization'] == 'l2' else 1e-4,
        # solver='sgd',
        verbose=10,
        random_state=42,
        learning_rate_init=config['learning_rate'],
        n_iter_no_change=config['epochs'],
        early_stopping=False
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    return mlp.loss_curve_

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

def train_custom_ann(config, X_train, y_train):
    model = ANNScratch(**config)
    model.fit(X_train, y_train)
    model.save_model('saved_models/my_ann_model.pkl')
    return model.loss_x, model.loss_y

def plot_results(mlp_loss, custom_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_loss, label='Scikit-learn MLP', color='blue')
    plt.plot(custom_loss, label='Custom ANN', color='red')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    model = None
    try:
        # X, y = load_mnist_from_csv()

        X, y = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        use_saved = input("\nLoad existing model? (y/n): ").lower() == 'y'
        custom_loss = []
        
        if use_saved:
            try:
                model_path = input("Enter model path (default: saved_models/my_ann_model.pkl): ") or "saved_models/my_ann_model.pkl"
                model = ANNScratch.load_model(model_path)
                print("Loaded existing model successfully!")
                
                if model.neurons[0] != X_train.shape[1]:
                    print(f"Error: Model expects input dimension {model.neurons[0]} but data has {X_train.shape[1]}")
                    print("Cannot continue with loaded model - creating new model instead")
                    use_saved = False
                    model = None
                    raise ValueError("Input dimension mismatch")
                
                # continue_train = input("Continue training this model? (y/n): ").lower() == 'y'
                # if continue_train:
                #     try:
                #         epochs = int(input("Epochs to train: "))
                #         model.epochs = epochs
                #         model.fit(X_train, y_train)
                #         custom_loss = model.loss_y
                #     except ValueError as e:
                #         print("Invalid number of epochs - using default epochs")
                #         model.fit(X_train, y_train)
                #         custom_loss = model.loss_y

                # Make predictions with the loaded model
                print("\nMaking predictions with loaded model...")
                y_pred = model.predict(X_test)
                
                # Evaluate predictions
                if model.loss == "binary_cross_entropy":
                    y_pred_class = (y_pred > 0.5).astype(int)
                    accuracy = np.mean(y_pred_class == y_test)
                    print(f"Test Accuracy: {accuracy:.4f}")
                elif model.loss == "categorical_cross_entropy":
                    y_pred_class = np.argmax(y_pred, axis=1)
                    y_test_class = np.argmax(y_test, axis=1)
                    accuracy = np.mean(y_pred_class == y_test_class)
                    print(f"Test Accuracy: {accuracy:.4f}")
                else: 
                    mse = np.mean((y_pred - y_test)**2)
                    print(f"Test MSE: {mse:.4f}")
                
                # Show some example predictions
                print("\nSample predictions:")
                # predict here
            except Exception as e:
                print(f"Error loading/continuing model: {e}")
                print("Proceeding with new model training...")
                use_saved = False
                model = None
        
        if not use_saved:
            try:
                config = get_user_model_config(X.shape[1])
                
                print("\nTraining scikit-learn MLP...")
                mlp_loss = train_sklearn_mlp(config, X_train, X_test, y_train, y_test)
                
                print("\nTraining custom ANN...")
                model = ANNScratch(**config)
                model.fit(X_train, y_train)

                
                custom_epochs, custom_loss = range(len(model.loss_y)), model.loss_y
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                return
            except Exception as e:
                print(f"\nError during training: {e}")
                raise
        else:
            print("\nTraining scikit-learn MLP for comparison...")
            config = {
                'neurons': model.neurons,
                'epochs': model.epochs,
                'learning_rate': model.learning_rate,
                'regularization': model.regularization,
                'reg_lambda': model.reg_lambda
            }
            mlp_loss = train_sklearn_mlp(config, X_train, X_test, y_train, y_test)
        
        plot_results(mlp_loss, custom_loss)

        if model is not None:
            visualize = input("\nVisualize model details? (y/n): ").lower() == 'y'
            if visualize:
                visualizer = ANNVisualizer(model)
                
                # Network architecture
                if input("Show network architecture? (y/n): ").lower() == 'y':
                    visualizer.visualize_network()
                
                # Weight distributions
                if input("Show weight distributions? (y/n): ").lower() == 'y':
                    layers = input("Enter layers to show (comma separated, leave empty for all): ")
                    layers = [int(l) for l in layers.split(',')] if layers else None
                    visualizer.plot_weight_distribution(layers=layers)
                
                # Gradient distributions
                if hasattr(model, 'weight_gradients'):
                    if input("Show gradient distributions? (y/n): ").lower() == 'y':
                        layers = input("Enter layers to show (comma separated, leave empty for all): ")
                        layers = [int(l) for l in layers.split(',')] if layers else None
                        visualizer.plot_gradient_distribution(layers=layers)
                
                # Neuron-specific inspection
                if input("Inspect specific neuron? (y/n): ").lower() == 'y':
                    try:
                        layer = int(input("Enter layer index: "))
                        neuron = int(input("Enter neuron index: "))
                        visualizer.plot_neuron_weights(layer, neuron)
                        if hasattr(model, 'weight_gradients'):
                            visualizer.plot_neuron_gradients(layer, neuron)
                    except (ValueError, IndexError) as e:
                        print(f"Invalid neuron specification: {e}")
        
        if model is not None:
            save_model = input("\nSave this model? (y/n): ").lower() == 'y'
            if save_model:
                model_path = input("Enter save path (default: saved_models/my_ann_model.pkl): ") or "saved_models/my_ann_model.pkl"
                try:
                    model.save_model(model_path)
                    print("Model saved successfully!")
                except Exception as e:
                    print(f"Error saving model: {e}")
        else:
            print("\nNo model available to save.")
                
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
