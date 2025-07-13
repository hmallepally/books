"""
Chapter 1 Exercises: Neural Network Basics
=========================================

Practice exercises for fundamental neural network concepts including:
- Perceptron implementation
- Activation functions
- Multi-layer perceptrons
- Forward and backward propagation
- Basic neural network operations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional


class Chapter1Exercises:
    """Exercise problems for Chapter 1."""
    
    def __init__(self):
        pass
    
    def exercise_1_1_perceptron_implementation(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Exercise 1.1: Implement a perceptron from scratch.
        
        Requirements:
        - Implement perceptron learning algorithm
        - Use step function activation
        - Track training history
        - Return final weights, bias, and training metrics
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            Dictionary with 'weights', 'bias', 'accuracy', 'epochs_trained'
        """
        # TODO: Implement your perceptron here
        # Replace this with your implementation
        pass
    
    def exercise_1_2_activation_functions(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Exercise 1.2: Implement activation functions and their derivatives.
        
        Requirements:
        - Implement sigmoid, tanh, ReLU, Leaky ReLU, ELU
        - Implement their derivatives
        - Handle numerical stability (especially for sigmoid)
        - Return both function values and derivatives
        
        Args:
            x: Input array
            
        Returns:
            Dictionary with function names as keys and (function_value, derivative) as values
        """
        # TODO: Implement your activation functions here
        # Replace this with your implementation
        pass
    
    def exercise_1_3_mlp_forward_pass(self, X: np.ndarray, weights: List[np.ndarray], 
                                    biases: List[np.ndarray], 
                                    activations: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Exercise 1.3: Implement forward pass for multi-layer perceptron.
        
        Requirements:
        - Implement forward propagation through all layers
        - Apply appropriate activation functions
        - Handle batch processing
        - Return both activations and pre-activations
        
        Args:
            X: Input data (n_samples, n_features)
            weights: List of weight matrices
            biases: List of bias vectors
            activations: List of activation function names
            
        Returns:
            Tuple of (activations, pre_activations) for all layers
        """
        # TODO: Implement your forward pass here
        # Replace this with your implementation
        pass
    
    def exercise_1_4_loss_functions(self, y_pred: np.ndarray, y_true: np.ndarray, 
                                  loss_type: str = 'mse') -> Tuple[float, np.ndarray]:
        """
        Exercise 1.4: Implement loss functions and their gradients.
        
        Requirements:
        - Implement MSE, Binary Cross-Entropy, Categorical Cross-Entropy
        - Implement gradients for each loss function
        - Handle numerical stability
        - Return both loss value and gradient
        
        Args:
            y_pred: Predicted values
            y_true: True values
            loss_type: 'mse', 'binary_crossentropy', or 'categorical_crossentropy'
            
        Returns:
            Tuple of (loss_value, gradient)
        """
        # TODO: Implement your loss functions here
        # Replace this with your implementation
        pass
    
    def exercise_1_5_backpropagation(self, X: np.ndarray, y: np.ndarray, 
                                   weights: List[np.ndarray], biases: List[np.ndarray],
                                   activations: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Exercise 1.5: Implement backpropagation algorithm.
        
        Requirements:
        - Compute gradients for all weights and biases
        - Use chain rule for gradient computation
        - Handle different activation functions
        - Return weight and bias gradients
        
        Args:
            X: Input data
            y: Target values
            weights: Current weight matrices
            biases: Current bias vectors
            activations: Activation function names
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        # TODO: Implement your backpropagation here
        # Replace this with your implementation
        pass
    
    def exercise_1_6_weight_initialization(self, layer_sizes: List[int], 
                                        method: str = 'xavier') -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Exercise 1.6: Implement different weight initialization methods.
        
        Requirements:
        - Implement random, Xavier/Glorot, He initialization
        - Handle different activation functions
        - Ensure proper variance scaling
        - Return initialized weights and biases
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            method: 'random', 'xavier', or 'he'
            
        Returns:
            Tuple of (weights, biases)
        """
        # TODO: Implement your weight initialization here
        # Replace this with your implementation
        pass
    
    def exercise_1_7_xor_problem(self) -> Dict[str, Any]:
        """
        Exercise 1.7: Solve the XOR problem with neural networks.
        
        Requirements:
        - Create XOR dataset
        - Implement single perceptron (should fail)
        - Implement MLP (should succeed)
        - Compare results and explain why perceptron fails
        
        Returns:
            Dictionary with results and explanations
        """
        # TODO: Implement your XOR solution here
        # Replace this with your implementation
        pass
    
    def exercise_1_8_decision_boundaries(self, X: np.ndarray, y: np.ndarray, 
                                       model_weights: List[np.ndarray], 
                                       model_biases: List[np.ndarray]) -> np.ndarray:
        """
        Exercise 1.8: Visualize decision boundaries for neural networks.
        
        Requirements:
        - Create mesh grid for 2D visualization
        - Implement forward pass for predictions
        - Plot decision boundaries
        - Handle different output formats (binary, multi-class)
        
        Args:
            X: Training data (2D)
            y: Training labels
            model_weights: Trained weight matrices
            model_biases: Trained bias vectors
            
        Returns:
            Decision boundary predictions
        """
        # TODO: Implement your decision boundary visualization here
        # Replace this with your implementation
        pass


def run_exercises():
    """Run all exercises with sample data."""
    
    exercises = Chapter1Exercises()
    
    print("=== Chapter 1 Neural Network Exercises ===\n")
    
    # Exercise 1.1: Perceptron Implementation
    print("Exercise 1.1 - Perceptron Implementation:")
    print("-" * 40)
    
    # Generate linearly separable data
    np.random.seed(42)
    X_linear = np.random.randn(100, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
    
    try:
        results = exercises.exercise_1_1_perceptron_implementation(X_linear, y_linear)
        print(f"✓ Perceptron implemented")
        print(f"✓ Final accuracy: {results.get('accuracy', 0):.3f}")
        print(f"✓ Epochs trained: {results.get('epochs_trained', 0)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.2: Activation Functions
    print("Exercise 1.2 - Activation Functions:")
    print("-" * 40)
    
    x_test = np.linspace(-5, 5, 100)
    
    try:
        results = exercises.exercise_1_2_activation_functions(x_test)
        print(f"✓ Activation functions implemented")
        print(f"✓ Functions implemented: {len(results)}")
        for func_name, (func_val, deriv_val) in results.items():
            print(f"  - {func_name}: range [{func_val.min():.3f}, {func_val.max():.3f}]")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.3: MLP Forward Pass
    print("Exercise 1.3 - MLP Forward Pass:")
    print("-" * 40)
    
    # Create sample network
    X_sample = np.random.randn(10, 2)
    weights = [np.random.randn(4, 2), np.random.randn(3, 4), np.random.randn(1, 3)]
    biases = [np.random.randn(4, 1), np.random.randn(3, 1), np.random.randn(1, 1)]
    activations = ['linear', 'relu', 'relu', 'sigmoid']
    
    try:
        activations_out, pre_activations = exercises.exercise_1_3_mlp_forward_pass(
            X_sample, weights, biases, activations
        )
        print(f"✓ Forward pass implemented")
        print(f"✓ Number of layers: {len(activations_out)}")
        print(f"✓ Output shape: {activations_out[-1].shape}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.4: Loss Functions
    print("Exercise 1.4 - Loss Functions:")
    print("-" * 40)
    
    y_pred = np.array([0.8, 0.2, 0.9, 0.1])
    y_true = np.array([1, 0, 1, 0])
    
    try:
        loss_value, gradient = exercises.exercise_1_4_loss_functions(y_pred, y_true, 'binary_crossentropy')
        print(f"✓ Loss functions implemented")
        print(f"✓ Loss value: {loss_value:.4f}")
        print(f"✓ Gradient shape: {gradient.shape}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.5: Backpropagation
    print("Exercise 1.5 - Backpropagation:")
    print("-" * 40)
    
    try:
        weight_grads, bias_grads = exercises.exercise_1_5_backpropagation(
            X_sample, y_true, weights, biases, activations
        )
        print(f"✓ Backpropagation implemented")
        print(f"✓ Weight gradients: {len(weight_grads)}")
        print(f"✓ Bias gradients: {len(bias_grads)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.6: Weight Initialization
    print("Exercise 1.6 - Weight Initialization:")
    print("-" * 40)
    
    layer_sizes = [2, 4, 3, 1]
    
    try:
        init_weights, init_biases = exercises.exercise_1_6_weight_initialization(layer_sizes, 'xavier')
        print(f"✓ Weight initialization implemented")
        print(f"✓ Weights initialized: {len(init_weights)}")
        print(f"✓ Biases initialized: {len(init_biases)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.7: XOR Problem
    print("Exercise 1.7 - XOR Problem:")
    print("-" * 40)
    
    try:
        results = exercises.exercise_1_7_xor_problem()
        print(f"✓ XOR problem solved")
        print(f"✓ Perceptron success: {results.get('perceptron_success', False)}")
        print(f"✓ MLP success: {results.get('mlp_success', False)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.8: Decision Boundaries
    print("Exercise 1.8 - Decision Boundaries:")
    print("-" * 40)
    
    try:
        decision_boundary = exercises.exercise_1_8_decision_boundaries(
            X_linear, y_linear, weights, biases
        )
        print(f"✓ Decision boundary visualization implemented")
        print(f"✓ Boundary shape: {decision_boundary.shape}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")


def provide_solutions():
    """Provide solution hints and examples."""
    
    print("=== Exercise Solutions and Hints ===\n")
    
    print("Exercise 1.1 - Perceptron Implementation Hints:")
    print("- Initialize weights and bias to small random values")
    print("- Use step function for activation")
    print("- Update weights: w += learning_rate * error * x")
    print("- Update bias: b += learning_rate * error")
    print("- Stop when no errors or max epochs reached")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.2 - Activation Functions Hints:")
    print("- Sigmoid: 1/(1 + exp(-x)), derivative: sigmoid(x) * (1 - sigmoid(x))")
    print("- Tanh: (exp(x) - exp(-x))/(exp(x) + exp(-x)), derivative: 1 - tanh²(x)")
    print("- ReLU: max(0, x), derivative: 1 if x > 0 else 0")
    print("- Leaky ReLU: max(αx, x), derivative: 1 if x > 0 else α")
    print("- ELU: x if x > 0 else α(exp(x) - 1)")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.3 - MLP Forward Pass Hints:")
    print("- For each layer: z = W @ a_prev + b")
    print("- Apply activation function: a = f(z)")
    print("- Store both activations and pre-activations")
    print("- Handle batch dimension properly")
    print("- Use appropriate activation functions for each layer")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.4 - Loss Functions Hints:")
    print("- MSE: mean((y_pred - y_true)²)")
    print("- Binary CE: -mean(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))")
    print("- Categorical CE: -mean(sum(y_true * log(y_pred), axis=1))")
    print("- Add epsilon to avoid log(0)")
    print("- Gradients: ∂L/∂y_pred for each loss function")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.5 - Backpropagation Hints:")
    print("- Start with output layer error: δ = ∂L/∂a * f'(z)")
    print("- Propagate error backward: δ_prev = W.T @ δ * f'(z_prev)")
    print("- Weight gradients: ∂L/∂W = δ @ a_prev.T")
    print("- Bias gradients: ∂L/∂b = δ")
    print("- Use chain rule for gradient computation")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.6 - Weight Initialization Hints:")
    print("- Random: N(0, 0.01)")
    print("- Xavier: N(0, 2/(n_in + n_out))")
    print("- He: N(0, 2/n_in)")
    print("- Initialize biases to 0 or small constant")
    print("- Scale based on activation function")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.7 - XOR Problem Hints:")
    print("- XOR data: [[0,0], [0,1], [1,0], [1,1]] -> [0, 1, 1, 0]")
    print("- Single perceptron cannot solve XOR (non-linearly separable)")
    print("- MLP with hidden layer can solve XOR")
    print("- Use appropriate activation functions")
    print("- Compare decision boundaries")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.8 - Decision Boundaries Hints:")
    print("- Create mesh grid using np.meshgrid")
    print("- Apply forward pass to all grid points")
    print("- Reshape predictions to grid shape")
    print("- Use plt.contourf for visualization")
    print("- Handle different output formats")
    print("\n" + "-"*40 + "\n")


def demonstrate_neural_network_concepts():
    """Demonstrate key neural network concepts for reference."""
    
    print("=== Neural Network Concepts Demonstration ===\n")
    
    # Demonstrate activation functions
    x = np.linspace(-5, 5, 100)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    
    # ReLU
    relu = np.maximum(0, x)
    relu_deriv = np.where(x > 0, 1, 0)
    
    # Tanh
    tanh = np.tanh(x)
    tanh_deriv = 1 - tanh**2
    
    print("Activation Function Properties:")
    print(f"Sigmoid range: [{sigmoid.min():.3f}, {sigmoid.max():.3f}]")
    print(f"ReLU range: [{relu.min():.3f}, {relu.max():.3f}]")
    print(f"Tanh range: [{tanh.min():.3f}, {tanh.max():.3f}]")
    
    # Demonstrate forward pass
    print(f"\nForward Pass Example:")
    X = np.array([[1, 2], [3, 4]])
    W = np.array([[0.5, 0.3], [0.2, 0.8]])
    b = np.array([[0.1], [0.2]])
    
    z = np.dot(W, X.T) + b
    a = 1 / (1 + np.exp(-z))  # sigmoid
    
    print(f"Input shape: {X.shape}")
    print(f"Weight shape: {W.shape}")
    print(f"Output shape: {a.shape}")
    
    # Demonstrate XOR problem
    print(f"\nXOR Problem:")
    xor_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xor_labels = np.array([0, 1, 1, 0])
    
    print("XOR Data:")
    for i, (data, label) in enumerate(zip(xor_data, xor_labels)):
        print(f"  {data} -> {label}")


if __name__ == "__main__":
    # Run exercises
    run_exercises()
    
    # Provide solution hints
    provide_solutions()
    
    # Demonstrate concepts
    demonstrate_neural_network_concepts() 