from ann import *

if __name__ == "__main__":
    n_layer = int(input("Jumlah layer: "))

    n_neurons = []
    activations = []
    loss = ""
    for i in range(n_layer):
        n_neuron_in_layer = int(input(f"Jumlah neuron layer-{i+1}: "))
        n_neurons.append(n_neuron_in_layer)

        activation_in_layer = input(f"1. Linear\
                                    \n2. ReLU\
                                    \n3. Sigmoid\
                                    \n4. Hyperbolic Tangent (tanh)\
                                    \n5. Softmax\
                                    \n6. Softplus\
                                    \n7. Leaky ReLU\
                                    \n8. Mish\
                                    \nFungsi aktivasi di layer-{i+1}: ").lower()
        
        match (activation_in_layer):
            case "1":
                activation_in_layer = "linear"
            case "2":
                activation_in_layer = "relu"
            case "3":
                activation_in_layer = "sigmoid"
            case "4":
                activation_in_layer = "tanh"
            case "5":
                activation_in_layer = "softmax"
            case "6":
                activation_in_layer = "softplus"
            case "7":
                activation_in_layer = "leaky_relu"
            case "8":
                activation_in_layer = "mish"
            case _:
                raise ValueError(f"Unsupported activation function: {activation_in_layer}")
        
        activations.append(activation_in_layer)

    loss = input(f"1. MSE\
                    \n2. Binary Cross-Entropy\
                    \n3. Categorical Cross-Entropy\
                    \nFungsi loss: ").lower()
    
    match (loss):
        case "1":
            loss = "mse"
        case "2":
            loss = "binary_cross_entropy"
        case "3":
            loss = "categorical_cross_entropy"
        case _:
            raise ValueError(f"Unsupported loss function: {loss}")
    
    batch_size = int(input(f"Batch size: "))
    learning_rate = float(input(f"Learning rate: "))
    n_epoch = int(input("Jumlah epoch: "))
    verbose = input("Verbose (0/1): ")

    if (verbose != "0" and verbose != "1"):
        raise ValueError(f"False input - Verbose: {verbose}")
    
    regularization = input("Regularization (None/L1/L2): ").lower()

    if (regularization != "none" and regularization != "l1" and regularization != "l2"):
        raise ValueError(f"False input - Regularization: {regularization}")
    
    reg_lambda = 0
    if (regularization != "none"):
        reg_lambda = float(input("Regularization strength (Regularization Lambda): "))
    
    rms_norm = input("RMS Norm (True/False): ").lower()

    if (rms_norm != "true" and rms_norm != "false"):
        raise ValueError(f"False input - RMS Norm: {rms_norm}")
    
    initialization = input(f"1. Zero\
                           \n2. Uniform\
                           \n3. Normal\
                           \n4. Xavier\
                           \n5. He\
                           \nInitialisasi: ")
    
    match (initialization):
        case "1":
            initialization = "zero"
        case "2":
            initialization = "uniform"
        case "3":
            initialization = "normal"
        case "4":
            initialization = "xavier"
        case "5":
            initialization = "he"
        case _:
            raise ValueError(f"Unsupported initialization: {initialization}")
    
    model = ANNScratch(
        neurons=n_neurons,
        activations=activations,
        epochs=n_epoch,
        loss=loss,
        learning_rate=learning_rate,
        batch_size=batch_size,
        verbose=verbose,
        regularization=regularization,
        reg_lambda=reg_lambda,
        initialization=initialization
    )

    X = np.random.randn(1000, 10)
    y = np.random.randn(1000, 1)

    model.fit(X, y)

    # DUMMY
    # model = ANNScratch(
    #     neurons=[10, 50, 150, 50, 1],
    #     activations=["relu", "tanh", "softplus", "sigmoid"],
    #     epochs=1000,
    #     loss="mse",
    #     learning_rate=1e-4,
    #     batch_size=32,
    #     verbose=1,
    #     # regularization="l2",
    #     reg_lambda=0.01
    # )

