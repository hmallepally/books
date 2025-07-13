"""
Neural Network Visualizations
============================

This module creates various diagrams and visualizations for neural network concepts.
Run this to generate images that can be used in the book documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_neural_network_diagram():
    """Create a diagram of a simple neural network."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Layer positions
    layer_positions = [1, 3, 5, 7]
    layer_sizes = [3, 4, 4, 2]
    layer_names = ['Input', 'Hidden 1', 'Hidden 2', 'Output']
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Draw layers
    for i, (pos, size, name, color) in enumerate(zip(layer_positions, layer_sizes, layer_names, colors)):
        # Draw neurons
        for j in range(size):
            y_pos = 2 - (j - (size-1)/2) * 0.8
            circle = Circle((pos, y_pos), 0.3, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            
            # Add neuron labels
            if i == 0:  # Input layer
                ax.text(pos-0.5, y_pos, f'x{j+1}', ha='right', va='center', fontsize=10, fontweight='bold')
            elif i == len(layer_positions)-1:  # Output layer
                ax.text(pos+0.5, y_pos, f'y{j+1}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Add layer labels
        ax.text(pos, 3.5, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw connections
    for i in range(len(layer_positions)-1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                y1 = 2 - (j - (layer_sizes[i]-1)/2) * 0.8
                y2 = 2 - (k - (layer_sizes[i+1]-1)/2) * 0.8
                
                arrow = FancyArrowPatch(
                    (layer_positions[i]+0.3, y1),
                    (layer_positions[i+1]-0.3, y2),
                    arrowstyle='->', mutation_scale=20, linewidth=1, alpha=0.6
                )
                ax.add_patch(arrow)
    
    # Add weights notation
    ax.text(4, 0.5, 'Weights (W)', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Neural Network Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('nn_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_activation_functions_plot():
    """Create a comprehensive plot of activation functions."""
    x = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    functions = [
        ('Step Function', lambda x: np.where(x >= 0, 1, 0), None),
        ('Sigmoid', sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
        ('Tanh', tanh, lambda x: 1 - tanh(x)**2),
        ('ReLU', relu, lambda x: np.where(x > 0, 1, 0)),
        ('Leaky ReLU', lambda x: leaky_relu(x, 0.1), lambda x: np.where(x > 0, 1, 0.1)),
        ('ELU', lambda x: elu(x, 1.0), lambda x: np.where(x > 0, 1, np.exp(x)))
    ]
    
    for i, (name, func, deriv_func) in enumerate(functions):
        y = func(x)
        axes[i].plot(x, y, 'b-', linewidth=3, label=f'{name}')
        
        if deriv_func is not None:
            y_deriv = deriv_func(x)
            axes[i].plot(x, y_deriv, 'r--', linewidth=2, label=f'{name} Derivative')
        
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=10)
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('x', fontsize=10)
        axes[i].set_ylabel('f(x)', fontsize=10)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Set y limits for better visualization
        if name == 'Step Function':
            axes[i].set_ylim(-0.1, 1.1)
        elif name in ['Sigmoid', 'Tanh']:
            axes[i].set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_training_curves():
    """Create example training curves."""
    epochs = np.arange(1, 101)
    
    # Simulate training curves
    train_loss = 2.0 * np.exp(-epochs/30) + 0.1 + 0.05 * np.random.randn(100)
    val_loss = 2.2 * np.exp(-epochs/25) + 0.15 + 0.08 * np.random.randn(100)
    train_acc = 1 - 0.8 * np.exp(-epochs/20) + 0.02 * np.random.randn(100)
    val_acc = 0.95 - 0.75 * np.exp(-epochs/18) + 0.03 * np.random.randn(100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_decision_boundaries():
    """Create decision boundary visualizations."""
    # Generate data
    np.random.seed(42)
    n_samples = 200
    
    # Linearly separable data
    X_linear = np.random.randn(n_samples, 2)
    y_linear = (X_linear[:, 0] + X_linear[:, 1] > 0).astype(int)
    
    # Non-linearly separable data (XOR-like)
    X_nonlinear = np.random.randn(n_samples, 2) * 2
    y_nonlinear = ((X_nonlinear[:, 0] > 0) ^ (X_nonlinear[:, 1] > 0)).astype(int)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear decision boundary
    ax1.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap='viridis', alpha=0.7, s=50)
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('Linearly Separable Data\n(Perceptron can solve)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Non-linear decision boundary
    ax2.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap='viridis', alpha=0.7, s=50)
    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title('Non-linearly Separable Data\n(MLP required)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('decision_boundaries.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_backpropagation_diagram():
    """Create a diagram showing backpropagation."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Layer positions
    layer_positions = [1, 3, 5, 7]
    layer_sizes = [2, 3, 3, 1]
    
    # Draw forward pass (solid lines)
    for i in range(len(layer_positions)-1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i+1]):
                y1 = 2 - (j - (layer_sizes[i]-1)/2) * 0.8
                y2 = 2 - (k - (layer_sizes[i+1]-1)/2) * 0.8
                
                arrow = FancyArrowPatch(
                    (layer_positions[i]+0.3, y1),
                    (layer_positions[i+1]-0.3, y2),
                    arrowstyle='->', mutation_scale=20, linewidth=2, color='blue', alpha=0.7
                )
                ax.add_patch(arrow)
    
    # Draw neurons
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (pos, size, color) in enumerate(zip(layer_positions, layer_sizes, colors)):
        for j in range(size):
            y_pos = 2 - (j - (size-1)/2) * 0.8
            circle = Circle((pos, y_pos), 0.3, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(circle)
    
    # Add labels
    ax.text(1, 3.5, 'Input', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(3, 3.5, 'Hidden 1', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, 3.5, 'Hidden 2', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(7, 3.5, 'Output', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add forward/backward labels
    ax.text(4, 0.5, 'Forward Pass (Blue)', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(4, 0.2, 'Backward Pass (Red)', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Draw backward pass arrows (dashed red)
    for i in range(len(layer_positions)-1, 0, -1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i-1]):
                y1 = 2 - (j - (layer_sizes[i]-1)/2) * 0.8
                y2 = 2 - (k - (layer_sizes[i-1]-1)/2) * 0.8
                
                arrow = FancyArrowPatch(
                    (layer_positions[i]-0.3, y1),
                    (layer_positions[i-1]+0.3, y2),
                    arrowstyle='->', mutation_scale=20, linewidth=2, color='red', 
                    alpha=0.7, linestyle='--'
                )
                ax.add_patch(arrow)
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Backpropagation: Forward and Backward Passes', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('backpropagation_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_cnn_diagram():
    """Create a diagram of a CNN architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Define layer positions and sizes
    layers = [
        {'name': 'Input\n(32x32x3)', 'pos': 1, 'size': 3, 'color': '#FF6B6B'},
        {'name': 'Conv1\n(30x30x16)', 'pos': 3, 'size': 4, 'color': '#4ECDC4'},
        {'name': 'Pool1\n(15x15x16)', 'pos': 5, 'size': 4, 'color': '#45B7D1'},
        {'name': 'Conv2\n(13x13x32)', 'pos': 7, 'size': 5, 'color': '#96CEB4'},
        {'name': 'Pool2\n(6x6x32)', 'pos': 9, 'size': 5, 'color': '#FFEAA7'},
        {'name': 'FC1\n(128)', 'pos': 11, 'size': 4, 'color': '#DDA0DD'},
        {'name': 'Output\n(10)', 'pos': 13, 'size': 2, 'color': '#98D8C8'}
    ]
    
    # Draw layers
    for layer in layers:
        pos = layer['pos']
        size = layer['size']
        color = layer['color']
        
        # Draw neurons
        for j in range(size):
            y_pos = 2 - (j - (size-1)/2) * 0.6
            circle = Circle((pos, y_pos), 0.25, facecolor=color, edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
        
        # Add layer labels
        ax.text(pos, 3.5, layer['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw connections
    for i in range(len(layers)-1):
        for j in range(layers[i]['size']):
            for k in range(layers[i+1]['size']):
                y1 = 2 - (j - (layers[i]['size']-1)/2) * 0.6
                y2 = 2 - (k - (layers[i+1]['size']-1)/2) * 0.6
                
                arrow = FancyArrowPatch(
                    (layers[i]['pos']+0.25, y1),
                    (layers[i+1]['pos']-0.25, y2),
                    arrowstyle='->', mutation_scale=15, linewidth=0.8, alpha=0.4
                )
                ax.add_patch(arrow)
    
    # Add special annotations
    ax.text(2, 0.5, 'Convolution\n(Feature Extraction)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(4, 0.5, 'Pooling\n(Dimensionality Reduction)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax.text(6, 0.5, 'Convolution\n(Feature Extraction)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(8, 0.5, 'Pooling\n(Dimensionality Reduction)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax.text(10, 0.5, 'Fully Connected\n(Classification)', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Convolutional Neural Network (CNN) Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('cnn_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_all_visualizations():
    """Generate all visualizations."""
    print("Generating neural network visualizations...")
    
    print("1. Creating neural network architecture diagram...")
    create_neural_network_diagram()
    
    print("2. Creating activation functions plot...")
    create_activation_functions_plot()
    
    print("3. Creating training curves...")
    create_training_curves()
    
    print("4. Creating decision boundaries...")
    create_decision_boundaries()
    
    print("5. Creating backpropagation diagram...")
    create_backpropagation_diagram()
    
    print("6. Creating CNN architecture diagram...")
    create_cnn_diagram()
    
    print("\nAll visualizations generated successfully!")
    print("Images saved as:")
    print("- nn_diagram.png")
    print("- activation_functions.png")
    print("- training_curves.png")
    print("- decision_boundaries.png")
    print("- backpropagation_diagram.png")
    print("- cnn_diagram.png")


if __name__ == "__main__":
    create_all_visualizations() 