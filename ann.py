import numpy as np
import matplotlib.pyplot as plt

class ANNScratch:
    def __init__(self, neurons, activations, epochs, loss, learning_rate, initialization = "normal", batch_size = 32, verbose = 1, regularization = None, reg_lambda = 0.01, epsilon = 1e-8, alpha = 0.01, rms_norm = None):
        self.neurons = neurons
        self.activations = activations
        self.epochs = epochs
        self.loss = loss
        self.learning_rate = learning_rate
        self.initialization = initialization
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.epsilon = epsilon
        self.alpha = alpha
        self.rms_norm = rms_norm
        self.weights = []
        self.biases = []
        self.initialize_weights()
    
    # array of (array of (one neuron to all next neuron))
    def initialize_weights(self):
        weight = []
        bias = []
        for i in range(len(self.neurons) - 1):
            input_dim = self.neurons[i]
            output_dim = self.neurons[i+1]

            print("Input dim: ", input_dim)
            print("Output dim: ", output_dim)
            if self.initialization == "zero":
                weight = np.zeros((input_dim, output_dim))
                bias = np.zeros((1, output_dim))
            elif self.initialization == "uniform":
                weight = np.random.uniform(-1, 1, (input_dim, output_dim))
                bias = np.random.uniform(-1, 1, (1, output_dim))
            elif self.initialization == "normal":
                weight = np.random.randn(input_dim, output_dim) * 0.1
                bias = np.random.randn(1, output_dim) * 0.1
            elif self.initialization == "xavier":
                scale = np.sqrt(2.0 / (input_dim + output_dim))
                weight = np.random.randn(input_dim, output_dim) * scale
                bias = np.random.randn(1, output_dim) * scale
            elif self.initialization == "he":
                scale = np.sqrt(2.0 / input_dim)
                weight = np.random.randn(input_dim, output_dim) * scale
                bias = np.random.randn(1, output_dim) * scale
            else:
                raise ValueError(f"Unsupported initialization method: {self.initialization}")
            
            # print(f"Weight {i}: ", weight)
            # print(f"Weight {i}: ", weight)

            self.weights.append(weight)
            self.biases.append(bias)
        
        # print("Initial weight: ", self.weights)
        # print("Initial weight: ", self.weights)

    def initialize_output_weights(self, y_dim):
        n_layer = len(self.neurons)

        input_dim = self.neurons[n_layer-1]
        output_dim = y_dim

        print("Input dim: ", input_dim)
        print("Output dim: ", output_dim)
        if self.initialization == "zero":
            weight = np.zeros((input_dim, output_dim))
            bias = np.zeros((1, output_dim))
        elif self.initialization == "uniform":
            weight = np.random.uniform(-1, 1, (input_dim, output_dim))
            bias = np.random.uniform(-1, 1, (1, output_dim))
        elif self.initialization == "normal":
            weight = np.random.randn(input_dim, output_dim) * 0.1
            bias = np.random.randn(1, output_dim) * 0.1
        elif self.initialization == "xavier":
            scale = np.sqrt(2.0 / (input_dim + output_dim))
            weight = np.random.randn(input_dim, output_dim) * scale
            bias = np.random.randn(1, output_dim) * scale
        elif self.initialization == "he":
            scale = np.sqrt(2.0 / input_dim)
            weight = np.random.randn(input_dim, output_dim) * scale
            bias = np.random.randn(1, output_dim) * scale
        else:
            raise ValueError(f"Unsupported initialization method: {self.initialization}")

        self.weights.append(weight)
        self.biases.append(bias)
        
        # print("Final weight: ", self.weights)

    def safe_exp(self, x, thr=700, inf=np.float64(1e18)):
        x_cop = np.copy(x)

        overflow = x_cop > thr
        underflow = x_cop < -thr
        
        result = np.zeros_like(x_cop, dtype=np.float64)
        
        safe_mask = ~(overflow | underflow)
        result[safe_mask] = np.exp(x_cop[safe_mask])
        
        result[overflow] = inf
        result[underflow] = 0.0
        
        return result

    def activation(self, x, func):
        if func == "linear":
            return x
        elif func == "relu":
            return np.maximum(0, x)
        elif func == "sigmoid":
            return 1 / (1 + self.safe_exp(-x))
        elif func == "tanh":
            return np.tanh(x)
        # TODO
        elif func == "softmax": 
            exp_x = self.safe_exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif func == "softplus":
            return np.log(1 + self.safe_exp(x))
        elif func == "leaky_relu":
            return np.where(x > 0, x, self.alpha * x)
        elif func == "mish":
            return x * np.tanh(np.log(1 + self.safe_exp(x)))
        else:
            raise ValueError(f"Unsupported activation function: {func}")
    
    def activation_derivative(self, x, func):
        if func == "linear":
            return np.ones_like(x)
        elif func == "relu":
            return np.where(x > 0, 1, 0)
        elif func == "sigmoid":
            return x * (1 - x)
        elif func == "tanh":
            return 1 - np.tanh(x) ** 2
        # TODO
        elif func == "softmax": # not yet
            return x * (1 - x)
        elif func == "softplus":
            return 1 / (1 + self.safe_exp(-x))
        elif func == "leaky_relu":
            return np.where(x > 0, 1, self.alpha)
        elif func == "mish":
            return np.tanh(np.log(1 + self.safe_exp(x))) + x * (1 - np.tanh(np.log(1 + self.safe_exp(x))) ** 2)
        else:
            raise ValueError(f"Unsupported activation function: {func}")
    
    def loss_function(self, y_actual, y_predicted):
        if self.loss == "mse":
            # print("Y actual", y_actual.shape)
            # print("Y predicted", y_predicted.shape)
            return np.mean((y_actual - y_predicted) ** 2)
        elif self.loss == "binary_cross_entropy":
            return -np.mean(y_actual * np.log(y_predicted + self.epsilon) + (1 - y_actual) * np.log(1 - y_predicted + self.epsilon))
        elif self.loss == "categorical_cross_entropy":
            return -np.mean(np.sum(y_actual * np.log(y_predicted + self.epsilon)))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def loss_gradient(self, y_actual, y_predicted):
        if self.loss == "mse":
            return -2 * (y_actual - y_predicted) / y_actual.shape[1]
        elif self.loss == "binary_cross_entropy":
            return (y_predicted - y_actual) / (y_predicted * (1 - y_predicted) + self.epsilon)
        elif self.loss == "categorical_cross_entropy":
            return y_predicted - y_actual
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def regularization_loss(self):
        if self.regularization == "l1":
            return self.reg_lambda * sum(np.sum(np.abs(w)) for w in self.weights)
        elif self.regularization == "l2":
            return self.reg_lambda * sum(np.sum(w**2) for w in self.weights)
        else:
            raise ValueError(f"Unsupported regularization method: {self.regularization}")
        
    def compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_function(y, y_pred) + self.regularization_loss()
    
    def apply_rms_norm(self, X):
        rms = np.sqrt(np.mean(X ** 2, axis = 1, keepdims = True) + self.epsilon)
        return X / rms

    def backward(self, X, y):
        gradients_w = [np.zeros_like(w) for w in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        y_predicted = self.predict(X) # forward

        if np.isnan(np.sum(X)) or np.isnan(np.sum(y)):
            print("WARNING: Input data contains NaN")
            assert False
        
        # Validate predictions
        if np.isnan(np.sum(y_predicted)):
            print("WARNING: Prediction contains NaN")
            assert False

        # Ensure y_predicted is properly shaped for binary classification
        # if y_predicted.shape[1] == 1:
        #     y_predicted = y_predicted.squeeze()  # Convert from (n_samples, 1) to (n_samples,)

        # reshape to be consistent with output shape
        # print("Y_PREDICTED", y_predicted)
        # print("Y", y)
        delta = self.loss_gradient(y, y_predicted)
        # print("DELTA", delta)

        for i in reversed(range(len(self.weights))):

            output = self.layer_outputs[i]
            input_data = self.layer_inputs[i]

            activation_derivative = self.activation_derivative(output, self.activations[i]) 

            # print("Activ der", activation_derivative)
            assert delta.shape == activation_derivative.shape

            delta = np.multiply(delta, activation_derivative)

            # print("Multiplied delta", delta)

            gradients_w[i] = np.matmul(input_data.T, delta) 
            gradients_b[i] = np.sum(delta, axis=0, keepdims=True) 

            if i > 0:
                delta = np.matmul(delta, self.weights[i].T)

        if self.regularization == "l1":
            for i in range(len(self.weights)):
                gradients_w[i] += self.reg_lambda * np.sign(self.weights[i])
        elif self.regularization == "l2":
            for i in range(len(self.weights)):
                gradients_w[i] += self.reg_lambda * 2 * self.weights[i]

        # print("Gradient W", gradients_w)
        # print("Gradient B", gradients_b)
        gradients_w = [np.clip(grad, -30, 30) for grad in gradients_w]
        gradients_b = [np.clip(grad, -30, 30) for grad in gradients_b]
        return gradients_w, gradients_b
        
    def update_weights(self, gradients_w, gradients_b):
        # TODO
        # for i in range(len(self.weights)):
        #     self.weights[i] -= self.learning_rate * gradients_w[i]
        #     self.biases[i] -= self.learning_rate * gradients_b[i]
        # print("Before update:")
        # original_weights = [np.copy(w) for w in self.weights]
        
        for i in range(len(self.weights)):
            # print(f"Layer {i} weight change: {np.mean(gradients_w[i])}")
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
        
        # # print("After update:")
        # for i, (orig, updated) in enumerate(zip(original_weights, self.weights)):
        #     weight_change = np.mean(np.abs(updated - orig))
        #     # print(f"Layer {i} total weight change: {weight_change}")
        #     if weight_change == 0:
        #         print(f"WARNING: Layer {i} weights did not change!")
        #         # assert False

    def fit(self, X, y):
        # TODO
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.initialize_output_weights(y.shape[1])

        self.loss_x = []
        self.loss_y = []

        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = X[batch_start:batch_end, :]
                y_batch = y[batch_start:batch_end]

                gradients_w, gradients_b = self.backward(X_batch, y_batch)
                self.update_weights(gradients_w, gradients_b)
            
            if self.verbose: # and epoch % 10 == 0:
                print("Weight size: ", len(self.weights[0]))
                print("Weight: ", self.weights)
                print("Bias size: ", len(self.biases))
                # print("Bias: ", self.biases)

                y_predicted = self.predict(X)

                loss = self.loss_function(y, y_predicted)
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")
                # print(f"Epoch {epoch}/{self.epochs}")

                self.loss_x.append(epoch)
                self.loss_y.append(loss)

        plt.figure(1)
        plt.plot(self.loss_x, self.loss_y)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.draw()

    def predict(self, X): # forward
        self.layer_outputs = []
        self.layer_inputs = []
        input_data = X
        cur_sample = X.shape[0]
        for i in range(len(self.weights)):
            
            self.layer_inputs.append(input_data)
            z = np.matmul(input_data, self.weights[i]) + np.tile(self.biases[i], (cur_sample, 1))

            if (self.rms_norm == "true"):
                input_data = self.apply_rms_norm(input_data)
                
            input_data = self.activation(z, self.activations[i]) #output
            self.layer_outputs.append(input_data)
            # print("Input_data shape", input_data.shape)
        # print("Result fit", input_data)
        row_sums = input_data.sum(axis=1)
        normalized_arr = np.divide(input_data, row_sums[:, np.newaxis], 
                                out=np.zeros_like(input_data, dtype=float), 
                                where=row_sums[:, np.newaxis]!=0)
        return normalized_arr