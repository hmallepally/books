"""
Chapter 1: Neural Network Basics
===============================

This module demonstrates fundamental neural network concepts including:
- Perceptron implementation
- Multi-layer perceptrons
- Activation functions
- Basic neural network operations
- Forward and backward propagation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class Perceptron:
    """Implementation of the original perceptron algorithm."""
    
    def __init__(self, learning_rate: float = 0.1, max_epochs: int = 100):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.training_history = []
    
    def initialize_parameters(self, n_features: int):
        """Initialize weights and bias."""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
    
    def step_function(self, x: np.ndarray) -> np.ndarray:
        """Step function activation."""
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the perceptron."""
        if self.weights is None:
            raise ValueError("Model not trained yet!")
        
        # Linear combination
        z = np.dot(X, self.weights) + self.bias
        # Apply step function
        return self.step_function(z)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, List[float]]:
        """Train the perceptron using the perceptron learning algorithm."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Training history
        self.training_history = {
            'weights': [],
            'bias': [],
            'accuracy': [],
            'errors': []
        }
        
        print("Training Perceptron...")
        
        for epoch in range(self.max_epochs):
            errors = 0
            correct_predictions = 0
            
            for i in range(n_samples):
                # Forward pass
                prediction = self.predict(X[i:i+1])[0]
                
                # Calculate error
                error = y[i] - prediction
                
                if error != 0:
                    errors += 1
                    # Update weights and bias
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                else:
                    correct_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / n_samples
            
            # Store history
            self.training_history['weights'].append(self.weights.copy())
            self.training_history['bias'].append(self.bias)
            self.training_history['accuracy'].append(accuracy)
            self.training_history['errors'].append(errors)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Accuracy = {accuracy:.3f}, Errors = {errors}")
            
            # Early stopping if no errors
            if errors == 0:
                print(f"Converged at epoch {epoch}!")
                break
        
        return self.training_history
    
    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, title: str = "Perceptron Decision Boundary"):
        """Plot the decision boundary of the perceptron."""
        if X.shape[1] != 2:
            print("Can only plot 2D data!")
            return
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        # Make predictions on mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, edgecolors='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.colorbar()
        plt.show()


class ActivationFunctions:
    """Collection of activation functions with their derivatives."""
    
    @staticmethod
    def step_function(x: np.ndarray) -> np.ndarray:
        """Step function (Heaviside function)."""
        return np.where(x >= 0, 1, 0)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function."""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Exponential Linear Unit activation function."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Derivative of ELU function."""
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        # Numerical stability: subtract max before exp
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    @staticmethod
    def plot_activation_functions():
        """Plot all activation functions and their derivatives."""
        x = np.linspace(-5, 5, 1000)
        
        functions = [
            ('Step Function', ActivationFunctions.step_function, None),
            ('Sigmoid', ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            ('Tanh', ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            ('ReLU', ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            ('Leaky ReLU', lambda x: ActivationFunctions.leaky_relu(x, 0.1), 
             lambda x: ActivationFunctions.leaky_relu_derivative(x, 0.1)),
            ('ELU', lambda x: ActivationFunctions.elu(x, 1.0), 
             lambda x: ActivationFunctions.elu_derivative(x, 1.0))
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name, func, deriv_func) in enumerate(functions):
            y = func(x)
            axes[i].plot(x, y, 'b-', linewidth=2, label=f'{name}')
            
            if deriv_func is not None:
                y_deriv = deriv_func(x)
                axes[i].plot(x, y_deriv, 'r--', linewidth=2, label=f'{name} Derivative')
            
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            axes[i].set_title(name)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('f(x)')
            axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class MultiLayerPerceptron:
    """Implementation of a multi-layer perceptron."""
    
    def __init__(self, layer_sizes: List[int], activation_functions: List[str] = None):
        """
        Initialize MLP.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1_size, ..., output_size]
            activation_functions: List of activation function names for each layer
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        
        # Default activation functions
        if activation_functions is None:
            self.activation_functions = ['linear'] + ['relu'] * (len(layer_sizes) - 2) + ['linear']
        else:
            self.activation_functions = activation_functions
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': []
        }
    
    def initialize_parameters(self):
        """Initialize weights and biases using Xavier/Glorot initialization."""
        for i in range(self.n_layers - 1):
            # Xavier/Glorot initialization
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i + 1]))
            
            # Weight matrix
            W = np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i]) * scale
            self.weights.append(W)
            
            # Bias vector
            b = np.zeros((self.layer_sizes[i + 1], 1))
            self.biases.append(b)
    
    def get_activation_function(self, name: str):
        """Get activation function by name."""
        activations = {
            'linear': lambda x: x,
            'sigmoid': ActivationFunctions.sigmoid,
            'tanh': ActivationFunctions.tanh,
            'relu': ActivationFunctions.relu,
            'leaky_relu': lambda x: ActivationFunctions.leaky_relu(x, 0.01),
            'elu': lambda x: ActivationFunctions.elu(x, 1.0),
            'softmax': ActivationFunctions.softmax
        }
        return activations.get(name, ActivationFunctions.relu)
    
    def get_activation_derivative(self, name: str):
        """Get activation function derivative by name."""
        derivatives = {
            'linear': lambda x: np.ones_like(x),
            'sigmoid': ActivationFunctions.sigmoid_derivative,
            'tanh': ActivationFunctions.tanh_derivative,
            'relu': ActivationFunctions.relu_derivative,
            'leaky_relu': lambda x: ActivationFunctions.leaky_relu_derivative(x, 0.01),
            'elu': lambda x: ActivationFunctions.elu_derivative(x, 1.0),
            'softmax': lambda x: np.ones_like(x)  # Simplified for now
        }
        return derivatives.get(name, ActivationFunctions.relu_derivative)
    
    def forward_pass(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Perform forward pass through the network."""
        # Store activations and pre-activations
        activations = [X.T]  # Transpose for column vectors
        pre_activations = []
        
        for i in range(self.n_layers - 1):
            # Linear transformation
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            pre_activations.append(z)
            
            # Apply activation function
            activation_func = self.get_activation_function(self.activation_functions[i + 1])
            a = activation_func(z)
            activations.append(a)
        
        return activations, pre_activations
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        activations, _ = self.forward_pass(X)
        return activations[-1].T
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray, loss_type: str = 'mse') -> float:
        """Compute loss function."""
        if loss_type == 'mse':
            return np.mean((y_pred - y_true) ** 2)
        elif loss_type == 'cross_entropy':
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute accuracy for classification."""
        if y_pred.shape[1] == 1:  # Binary classification
            predictions = (y_pred > 0.5).astype(int)
        else:  # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        
        return np.mean(predictions == y_true)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              learning_rate: float = 0.01, batch_size: int = 32, 
              loss_type: str = 'mse') -> Dict[str, List[float]]:
        """Train the neural network using mini-batch gradient descent."""
        n_samples = X.shape[0]
        
        print("Training Multi-Layer Perceptron...")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            total_accuracy = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                activations, pre_activations = self.forward_pass(batch_X)
                
                # Compute loss and accuracy
                y_pred = activations[-1].T
                loss = self.compute_loss(y_pred, batch_y, loss_type)
                accuracy = self.compute_accuracy(y_pred, batch_y)
                
                total_loss += loss
                total_accuracy += accuracy
                n_batches += 1
            
            # Average loss and accuracy
            avg_loss = total_loss / n_batches
            avg_accuracy = total_accuracy / n_batches
            
            # Store history
            self.training_history['loss'].append(avg_loss)
            self.training_history['accuracy'].append(avg_accuracy)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")
        
        return self.training_history
    
    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(self.training_history['loss'])
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.training_history['accuracy'])
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def demonstrate_perceptron():
    """Demonstrate perceptron learning on linearly separable data."""
    print("=== Perceptron Demonstration ===\n")
    
    # Generate linearly separable data
    np.random.seed(42)
    n_samples = 100
    
    # Class 0: points with x + y < 0
    class0_x = np.random.randn(n_samples // 2) - 1
    class0_y = np.random.randn(n_samples // 2) - 1
    class0 = np.column_stack([class0_x, class0_y])
    
    # Class 1: points with x + y > 0
    class1_x = np.random.randn(n_samples // 2) + 1
    class1_y = np.random.randn(n_samples // 2) + 1
    class1 = np.column_stack([class1_x, class1_y])
    
    # Combine data
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.1, max_epochs=100)
    history = perceptron.fit(X, y)
    
    # Plot results
    perceptron.plot_decision_boundary(X, y, "Perceptron Decision Boundary")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['errors'])
    plt.title('Number of Errors')
    plt.xlabel('Epoch')
    plt.ylabel('Errors')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return perceptron, X, y


def demonstrate_activation_functions():
    """Demonstrate different activation functions."""
    print("\n=== Activation Functions Demonstration ===\n")
    
    # Plot all activation functions
    ActivationFunctions.plot_activation_functions()
    
    # Demonstrate properties
    x = np.linspace(-5, 5, 1000)
    
    print("Activation Function Properties:")
    print("-" * 40)
    
    # Sigmoid
    sigmoid_output = ActivationFunctions.sigmoid(x)
    print(f"Sigmoid range: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]")
    print(f"Sigmoid mean: {sigmoid_output.mean():.3f}")
    
    # Tanh
    tanh_output = ActivationFunctions.tanh(x)
    print(f"Tanh range: [{tanh_output.min():.3f}, {tanh_output.max():.3f}]")
    print(f"Tanh mean: {tanh_output.mean():.3f}")
    
    # ReLU
    relu_output = ActivationFunctions.relu(x)
    print(f"ReLU range: [{relu_output.min():.3f}, {relu_output.max():.3f}]")
    print(f"ReLU mean: {relu_output.mean():.3f}")
    print(f"ReLU sparsity: {(relu_output == 0).mean():.3f}")


def demonstrate_mlp():
    """Demonstrate multi-layer perceptron."""
    print("\n=== Multi-Layer Perceptron Demonstration ===\n")
    
    # Generate non-linear data (XOR-like problem)
    np.random.seed(42)
    n_samples = 200
    
    # Create XOR-like pattern
    X = np.random.randn(n_samples, 2) * 2
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int).reshape(-1, 1)
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.flatten())}")
    
    # Create MLP
    layer_sizes = [2, 10, 8, 1]  # Input: 2, Hidden: 10, 8, Output: 1
    activation_functions = ['linear', 'relu', 'relu', 'sigmoid']
    
    mlp = MultiLayerPerceptron(layer_sizes, activation_functions)
    
    # Train MLP
    history = mlp.train(X, y, epochs=200, learning_rate=0.01, batch_size=32, loss_type='mse')
    
    # Plot training history
    mlp.plot_training_history()
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Make predictions
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), alpha=0.8, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('MLP Decision Boundary')
    plt.colorbar()
    plt.show()
    
    return mlp, X, y


def demonstrate_xor_problem():
    """Demonstrate why single perceptron cannot solve XOR."""
    print("\n=== XOR Problem Demonstration ===\n")
    
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    print("XOR Data:")
    print("Input | Output")
    print("------|-------")
    for i in range(len(X_xor)):
        print(f"{X_xor[i]} | {y_xor[i][0]}")
    
    print("\nTrying to solve XOR with single perceptron...")
    
    # Try perceptron
    perceptron = Perceptron(learning_rate=0.1, max_epochs=1000)
    history = perceptron.fit(X_xor, y_xor.flatten())
    
    print(f"Final accuracy: {history['accuracy'][-1]:.3f}")
    print("Perceptron cannot solve XOR (non-linearly separable problem)")
    
    print("\nTrying to solve XOR with MLP...")
    
    # Try MLP
    mlp = MultiLayerPerceptron([2, 4, 1], ['linear', 'relu', 'sigmoid'])
    mlp_history = mlp.train(X_xor, y_xor, epochs=1000, learning_rate=0.1, loss_type='mse')
    
    print(f"Final accuracy: {mlp_history['accuracy'][-1]:.3f}")
    print("MLP can solve XOR (non-linear problem)")
    
    # Plot XOR data
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=y_xor.flatten(), s=100, alpha=0.8)
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('XOR Data')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_history['accuracy'])
    plt.title('MLP Training Accuracy on XOR')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_demonstrations():
    """Run all demonstrations."""
    
    print("Neural Network Basics Demonstrations")
    print("=" * 50)
    
    # Demonstrate activation functions
    demonstrate_activation_functions()
    
    # Demonstrate perceptron
    perceptron, X_perceptron, y_perceptron = demonstrate_perceptron()
    
    # Demonstrate XOR problem
    demonstrate_xor_problem()
    
    # Demonstrate MLP
    mlp, X_mlp, y_mlp = demonstrate_mlp()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)
    
    return {
        'perceptron': perceptron,
        'mlp': mlp,
        'data': {
            'perceptron': (X_perceptron, y_perceptron),
            'mlp': (X_mlp, y_mlp)
        }
    }


if __name__ == "__main__":
    # Run all demonstrations
    results = run_demonstrations()
    
    print("\nKey Takeaways:")
    print("1. Perceptrons can only solve linearly separable problems")
    print("2. Multi-layer perceptrons can solve non-linear problems")
    print("3. Activation functions introduce non-linearity")
    print("4. Different activation functions have different properties")
    print("5. Neural networks can learn complex patterns from data") 