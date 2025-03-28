import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from visualizer import ANNVisualizer

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

        self.weight_gradients = []
    
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
        
            self.weights.append(weight)
            self.biases.append(bias)

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
        elif func == "softmax":
            shiftx = x - np.max(x)
            exps = self.safe_exp(shiftx)
            return exps / np.sum(exps)
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
            return np.mean((y_actual - y_predicted) ** 2)
        elif self.loss == "binary_cross_entropy":
            return -np.mean(y_actual * np.log(y_predicted + self.epsilon) + (1 - y_actual) * np.log(1 - y_predicted + self.epsilon))
        elif self.loss == "categorical_cross_entropy":
            return -np.mean(np.sum(y_actual * np.log(y_predicted + self.epsilon), axis=1))
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
        
        y_predicted = self.predict(X)

        delta = self.loss_gradient(y, y_predicted)

        for i in reversed(range(len(self.weights))):

            output = self.layer_outputs[i]
            input_data = self.layer_inputs[i]

            activation_derivative = self.activation_derivative(output, self.activations[i]) 

            assert delta.shape == activation_derivative.shape

            delta = np.multiply(delta, activation_derivative)

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

        gradients_w = [np.clip(grad, -1, 1) for grad in gradients_w]
        gradients_b = [np.clip(grad, -1, 1) for grad in gradients_b]

        self.weight_gradients = [gw.copy() for gw in gradients_w]
        self.bias_gradients = [gb.copy() for gb in gradients_b]
        
        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        print(y)

        print(y.shape)

        self.initialize_output_weights(y.shape[1])
        self.neurons.append(y.shape[1])

        self.loss_x = []
        self.loss_y = []

        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            for batch_start in range(0, n_samples, self.batch_size):
                batch_end = batch_start + self.batch_size
                X_batch = X[batch_start:batch_end, :]
                y_batch = y[batch_start:batch_end]

                gradients_w, gradients_b = self.backward(X_batch, y_batch)
                self.update_weights(gradients_w, gradients_b)
            
            if self.verbose:
                # print("Weight size: ", len(self.weights[0]))
                # print("Weight: ", self.weights)
                # print("Bias size: ", len(self.biases))
                # print("Bias: ", self.biases)

                y_predicted = self.predict(X)

                loss = self.loss_function(y, y_predicted)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

                self.loss_x.append(epoch)
                self.loss_y.append(loss)

    def predict(self, X):
        self.layer_outputs = []
        self.layer_inputs = []
        input_data = X
        cur_sample = X.shape[0]
        for i in range(len(self.weights)):
            
            self.layer_inputs.append(input_data)
            z = np.matmul(input_data, self.weights[i]) + np.tile(self.biases[i], (cur_sample, 1))

            if (self.rms_norm == "true"):
                input_data = self.apply_rms_norm(input_data)
                
            input_data = self.activation(z, self.activations[i]) 
            self.layer_outputs.append(input_data)

        # normalize output
        row_sums = input_data.sum(axis=1)
        normalized_arr = np.divide(input_data, row_sums[:, np.newaxis], 
                                out=np.zeros_like(input_data, dtype=float), 
                                where=row_sums[:, np.newaxis]!=0)
        return normalized_arr
    
    def get_model_state(self):
        state = {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'neurons': self.neurons.copy(),
            'activations': self.activations.copy(),
            'hyperparameters': {
                'epochs': self.epochs,
                'loss': self.loss,
                'learning_rate': self.learning_rate,
                'initialization': self.initialization,
                'batch_size': self.batch_size,
                'regularization': self.regularization,
                'reg_lambda': self.reg_lambda,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'rms_norm': self.rms_norm
            }
        }
        
        if hasattr(self, 'weight_gradients'):
            state['weight_gradients'] = [wg.copy() for wg in self.weight_gradients]
        if hasattr(self, 'bias_gradients'):
            state['bias_gradients'] = [bg.copy() for bg in self.bias_gradients]
        if hasattr(self, 'layer_outputs'):
            state['layer_outputs'] = [out.copy() for out in self.layer_outputs]
        if hasattr(self, 'layer_inputs'):
            state['layer_inputs'] = [inp.copy() for inp in self.layer_inputs]
        
        return state

    def save_model(self, filepath):
        try:
            model_state = self.get_model_state()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath, input_dim=None, output_dim=None):
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            if input_dim and model_state['neurons'][0] != input_dim:
                raise ValueError(f"Model expects {model_state['neurons'][0]} input features, got {input_dim}")
            if output_dim and model_state['neurons'][-1] != output_dim:
                raise ValueError(f"Model expects {model_state['neurons'][-1]} outputs, got {output_dim}")
            
            model = cls(
                neurons=model_state['neurons'],
                activations=model_state['activations'],
                epochs=model_state['hyperparameters']['epochs'],
                loss=model_state['hyperparameters']['loss'],
                learning_rate=model_state['hyperparameters']['learning_rate'],
                initialization=model_state['hyperparameters']['initialization'],
                batch_size=model_state['hyperparameters']['batch_size'],
                regularization=model_state['hyperparameters']['regularization'],
                reg_lambda=model_state['hyperparameters']['reg_lambda'],
                epsilon=model_state['hyperparameters']['epsilon'],
                alpha=model_state['hyperparameters']['alpha'],
                rms_norm=model_state['hyperparameters']['rms_norm']
            )
            
            model.weights = [w.copy() for w in model_state['weights']]
            model.biases = [b.copy() for b in model_state['biases']]
            
            if 'weight_gradients' in model_state:
                model.weight_gradients = [wg.copy() for wg in model_state['weight_gradients']]
            if 'bias_gradients' in model_state:
                model.bias_gradients = [bg.copy() for bg in model_state['bias_gradients']]
            if 'layer_outputs' in model_state:
                model.layer_outputs = [out.copy() for out in model_state['layer_outputs']]
            if 'layer_inputs' in model_state:
                model.layer_inputs = [inp.copy() for inp in model_state['layer_inputs']]
            
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def visualize(self):
        visualizer = ANNVisualizer(self)
        
        print("\nNetwork Architecture Visualization:")
        visualizer.visualize_network()
        
        print("\nWeight Distributions:")
        visualizer.plot_weight_distribution()
        
        if hasattr(self, 'weight_gradients'):
            print("\nGradient Distributions:")
            visualizer.plot_gradient_distribution()