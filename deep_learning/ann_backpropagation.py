"""
Artificial Neural Network with Backpropagation Algorithm Implementation
This module implements a feed-forward neural network with backpropagation from scratch
"""

import numpy as np
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(output: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid (output is already sigmoid(x))"""
        return output * (1 - output)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(output: np.ndarray) -> np.ndarray:
        """Derivative of tanh (output is already tanh(x))"""
        return 1 - output ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(output: np.ndarray) -> np.ndarray:
        """Derivative of ReLU (output is already relu(x))"""
        return (output > 0).astype(float)
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation function"""
        return x
    
    @staticmethod
    def linear_derivative(output: np.ndarray) -> np.ndarray:
        """Derivative of linear activation"""
        return np.ones_like(output)


class NeuralNetwork:
    """Feed-forward Neural Network with Backpropagation"""
    
    def __init__(self, 
                 layer_sizes: List[int],
                 activation: str = 'sigmoid',
                 learning_rate: float = 0.1,
                 momentum: float = 0.0):
        """
        Initialize neural network
        
        Args:
            layer_sizes: List of layer sizes, e.g., [2, 4, 1] for 2 input, 4 hidden, 1 output
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            learning_rate: Learning rate for gradient descent
            momentum: Momentum coefficient for weight updates
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_layers = len(layer_sizes)
        
        # Set activation functions
        self._set_activation_functions()
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.weight_velocities = []  # For momentum
        self.bias_velocities = []
        
        self._initialize_parameters()
        
        # Store activations for backprop
        self.activations = []
        self.z_values = []
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def _set_activation_functions(self):
        """Set activation function and its derivative"""
        if self.activation_name == 'sigmoid':
            self.activation = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif self.activation_name == 'tanh':
            self.activation = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
        elif self.activation_name == 'relu':
            self.activation = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif self.activation_name == 'linear':
            self.activation = ActivationFunctions.linear
            self.activation_derivative = ActivationFunctions.linear_derivative
        else:
            raise ValueError(f"Unknown activation function: {self.activation_name}")
    
    def _initialize_parameters(self):
        """Initialize weights and biases using He initialization"""
        np.random.seed(42)
        
        for i in range(self.num_layers - 1):
            # He initialization for better convergence
            if self.activation_name == 'relu':
                std = np.sqrt(2.0 / self.layer_sizes[i])
            else:
                std = np.sqrt(1.0 / self.layer_sizes[i])
            
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * std
            b = np.zeros((1, self.layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
            
            # Initialize velocities for momentum
            self.weight_velocities.append(np.zeros_like(w))
            self.bias_velocities.append(np.zeros_like(b))
    
    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network
        
        Args:
            X: Input data of shape (batch_size, input_features)
            
        Returns:
            Output predictions
        """
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i in range(self.num_layers - 1):
            # Compute z = a @ w + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function (except output layer uses sigmoid for binary classification)
            if i == self.num_layers - 2:
                # Output layer always uses sigmoid for binary classification
                a = ActivationFunctions.sigmoid(z)
            else:
                a = self.activation(z)
            
            self.activations.append(a)
            current_input = a
        
        return current_input
    
    def backward_propagation(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """
        Backward propagation (backpropagation algorithm)
        
        Args:
            X: Input data
            y: Target values
            output: Network output from forward pass
        """
        m = X.shape[0]  # batch size
        
        # Initialize delta for output layer
        delta = (output - y) * ActivationFunctions.sigmoid_derivative(output)
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases with momentum
            if self.momentum > 0:
                self.weight_velocities[i] = (self.momentum * self.weight_velocities[i] - 
                                           self.learning_rate * dW)
                self.bias_velocities[i] = (self.momentum * self.bias_velocities[i] - 
                                         self.learning_rate * db)
                
                self.weights[i] += self.weight_velocities[i]
                self.biases[i] += self.bias_velocities[i]
            else:
                self.weights[i] -= self.learning_rate * dW
                self.biases[i] -= self.learning_rate * db
            
            # Compute delta for next layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                # Apply activation derivative
                if i - 1 == self.num_layers - 2:
                    delta *= ActivationFunctions.sigmoid_derivative(self.activations[i])
                else:
                    delta *= self.activation_derivative(self.activations[i])
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              batch_size: int = None, verbose: bool = True):
        """
        Train the neural network
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Batch size for training (None = full batch)
            verbose: Print training progress
        """
        if batch_size is None:
            batch_size = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward_propagation(X_batch)
                
                # Compute loss
                loss = self._compute_loss(y_batch, output)
                epoch_loss += loss
                
                # Compute accuracy
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y_batch)
                epoch_accuracy += accuracy
                
                num_batches += 1
                
                # Backward pass
                self.backward_propagation(X_batch, y_batch, output)
            
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(epoch_accuracy)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.6f}, Accuracy: {epoch_accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on input data"""
        output = self.forward_propagation(X)
        return (output > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability on input data"""
        return self.forward_propagation(X)
    
    @staticmethod
    def _compute_loss(y: np.ndarray, output: np.ndarray) -> float:
        """Compute binary cross-entropy loss"""
        m = y.shape[0]
        # Clip to avoid log(0)
        output = np.clip(output, 1e-15, 1 - 1e-15)
        loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        return loss
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate on test data
        
        Returns:
            loss, accuracy
        """
        output = self.forward_propagation(X)
        loss = self._compute_loss(y, output)
        predictions = (output > 0.5).astype(int)
        accuracy = np.mean(predictions == y)
        return loss, accuracy


def create_xor_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create XOR dataset for testing"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return X, y, X, y  # No separate test set for XOR


def create_circles_dataset(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create binary classification dataset (circles)"""
    np.random.seed(42)
    
    # Generate samples from two circles
    n_per_class = n_samples // 2
    
    # Class 0: inner circle
    r0 = np.random.uniform(0, 0.5, n_per_class)
    theta0 = np.random.uniform(0, 2*np.pi, n_per_class)
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    y0 = np.zeros((n_per_class, 1))
    
    # Class 1: outer circle
    r1 = np.random.uniform(0.7, 1.0, n_per_class)
    theta1 = np.random.uniform(0, 2*np.pi, n_per_class)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.ones((n_per_class, 1))
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test


def create_linearly_separable_dataset(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create linearly separable binary classification dataset"""
    np.random.seed(42)
    
    n_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1)
    X0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1, -1])
    y0 = np.zeros((n_per_class, 1))
    
    # Class 1: centered at (1, 1)
    X1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([1, 1])
    y1 = np.ones((n_per_class, 1))
    
    # Combine
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, y_train, X_test, y_test
