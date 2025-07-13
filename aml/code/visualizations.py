"""
Machine Learning Visualizations
==============================

This module creates various diagrams and visualizations for machine learning concepts.
Run this to generate images that can be used in the book documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as patches
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_ml_types_diagram():
    """Create a diagram showing different types of machine learning."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define ML types and their characteristics
    ml_types = {
        'Supervised Learning': {
            'pos': (3, 7),
            'color': '#FF6B6B',
            'examples': ['Classification', 'Regression'],
            'description': 'Learn from labeled data'
        },
        'Unsupervised Learning': {
            'pos': (11, 7),
            'color': '#4ECDC4',
            'examples': ['Clustering', 'Dimensionality Reduction'],
            'description': 'Find patterns in unlabeled data'
        },
        'Reinforcement Learning': {
            'pos': (7, 3),
            'color': '#45B7D1',
            'examples': ['Q-Learning', 'Policy Gradient'],
            'description': 'Learn through interaction'
        }
    }
    
    # Draw main ML types
    for ml_type, info in ml_types.items():
        x, y = info['pos']
        color = info['color']
        
        # Main box
        rect = FancyBboxPatch(
            (x - 2, y - 0.8), 4, 1.6,
            boxstyle="round,pad=0.1", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, ml_type, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Description
        ax.text(x, y - 0.3, info['description'], ha='center', va='center', fontsize=10, style='italic')
        
        # Examples
        for i, example in enumerate(info['examples']):
            ex_x = x - 1.5 + i * 1.5
            ex_y = y - 1.5
            ex_rect = FancyBboxPatch(
                (ex_x - 0.6, ex_y - 0.2), 1.2, 0.4,
                boxstyle="round,pad=0.05", facecolor='white', 
                edgecolor=color, linewidth=1.5
            )
            ax.add_patch(ex_rect)
            ax.text(ex_x, ex_y, example, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add algorithms under each type
    algorithms = {
        'Supervised Learning': ['Linear Regression', 'Logistic Regression', 'SVM', 'Random Forest', 'Neural Networks'],
        'Unsupervised Learning': ['K-Means', 'DBSCAN', 'PCA', 't-SNE', 'Autoencoders'],
        'Reinforcement Learning': ['Q-Learning', 'SARSA', 'Deep Q-Network', 'Actor-Critic', 'Policy Gradient']
    }
    
    for ml_type, algo_list in algorithms.items():
        x, y = ml_types[ml_type]['pos']
        color = ml_types[ml_type]['color']
        
        for i, algo in enumerate(algo_list):
            algo_x = x - 2.5 + (i % 3) * 1.7
            algo_y = y - 2.5 - (i // 3) * 0.4
            algo_rect = FancyBboxPatch(
                (algo_x - 0.7, algo_y - 0.15), 1.4, 0.3,
                boxstyle="round,pad=0.02", facecolor='lightgray', 
                edgecolor=color, linewidth=1
            )
            ax.add_patch(algo_rect)
            ax.text(algo_x, algo_y, algo, ha='center', va='center', fontsize=7)
    
    # Add central "Machine Learning" title
    center_circle = Circle((7, 5), 1, facecolor='#96CEB4', edgecolor='black', linewidth=2)
    ax.add_patch(center_circle)
    ax.text(7, 5, 'Machine\nLearning', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Connect to main types
    for ml_type, info in ml_types.items():
        x, y = info['pos']
        arrow = FancyArrowPatch(
            (7, 6), (x, y - 0.8),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Types of Machine Learning', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ml_types_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_ml_workflow_diagram():
    """Create a detailed ML workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Define workflow steps with sub-steps
    workflow = [
        {
            'name': 'Problem Definition',
            'pos': (2, 8),
            'color': '#FF6B6B',
            'substeps': ['Define objective', 'Identify metrics', 'Set constraints']
        },
        {
            'name': 'Data Collection',
            'pos': (4, 8),
            'color': '#4ECDC4',
            'substeps': ['Gather data', 'Check quality', 'Document sources']
        },
        {
            'name': 'Data Preprocessing',
            'pos': (6, 8),
            'color': '#45B7D1',
            'substeps': ['Clean data', 'Handle missing', 'Feature scaling']
        },
        {
            'name': 'Exploratory Analysis',
            'pos': (8, 8),
            'color': '#96CEB4',
            'substeps': ['Visualize data', 'Check distributions', 'Find patterns']
        },
        {
            'name': 'Feature Engineering',
            'pos': (10, 8),
            'color': '#FFEAA7',
            'substeps': ['Create features', 'Select features', 'Transform data']
        },
        {
            'name': 'Model Selection',
            'pos': (12, 8),
            'color': '#DDA0DD',
            'substeps': ['Choose algorithms', 'Set parameters', 'Design pipeline']
        },
        {
            'name': 'Training',
            'pos': (14, 8),
            'color': '#98D8C8',
            'substeps': ['Split data', 'Train model', 'Validate']
        },
        {
            'name': 'Evaluation',
            'pos': (16, 8),
            'color': '#F7DC6F',
            'substeps': ['Test model', 'Measure performance', 'Analyze errors']
        },
        {
            'name': 'Deployment',
            'pos': (18, 8),
            'color': '#BB8FCE',
            'substeps': ['Deploy model', 'Monitor', 'Maintain']
        }
    ]
    
    # Draw main workflow
    for i, step in enumerate(workflow):
        x, y = step['pos']
        color = step['color']
        
        # Main step box
        rect = FancyBboxPatch(
            (x - 0.8, y - 0.3), 1.6, 0.6,
            boxstyle="round,pad=0.05", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Substeps
        for j, substep in enumerate(step['substeps']):
            sub_x = x - 0.6 + j * 0.4
            sub_y = y - 1.2
            sub_rect = FancyBboxPatch(
                (sub_x - 0.15, sub_y - 0.1), 0.3, 0.2,
                boxstyle="round,pad=0.02", facecolor='white', 
                edgecolor=color, linewidth=1
            )
            ax.add_patch(sub_rect)
            ax.text(sub_x, sub_y, substep, ha='center', va='center', fontsize=6)
    
    # Connect steps
    for i in range(len(workflow) - 1):
        x1, y1 = workflow[i]['pos']
        x2, y2 = workflow[i + 1]['pos']
        
        arrow = FancyArrowPatch(
            (x1 + 0.8, y1), (x2 - 0.8, y2),
            arrowstyle='->', mutation_scale=15, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add feedback loops
    # Evaluation to Feature Engineering
    arrow1 = FancyArrowPatch(
        (16, 7.7), (10, 7.7),
        arrowstyle='->', mutation_scale=15, linewidth=2, color='red', linestyle='--'
    )
    ax.add_patch(arrow1)
    ax.text(13, 7.4, 'Iterate', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # Evaluation to Model Selection
    arrow2 = FancyArrowPatch(
        (16, 7.5), (12, 7.5),
        arrowstyle='->', mutation_scale=15, linewidth=2, color='red', linestyle='--'
    )
    ax.add_patch(arrow2)
    
    # Add tools/libraries
    tools = [
        'Business\nRequirements', 'APIs, Databases', 'Pandas, NumPy',
        'Matplotlib,\nSeaborn', 'Scikit-learn', 'Scikit-learn',
        'Cross-validation', 'Metrics', 'Flask, Docker'
    ]
    
    for i, (step, tool) in enumerate(zip(workflow, tools)):
        x, y = step['pos']
        ax.text(x, y - 1.8, tool, ha='center', va='center', fontsize=7, style='italic')
    
    ax.set_xlim(0, 20)
    ax.set_ylim(5, 9)
    ax.axis('off')
    ax.set_title('Machine Learning Workflow', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('ml_workflow_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_algorithm_comparison():
    """Create comparison charts for different ML algorithms."""
    # Generate sample data
    np.random.seed(42)
    
    # Classification data
    X_clf, y_clf = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                     n_informative=2, random_state=42, n_clusters_per_class=1)
    
    # Regression data
    X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    
    # Clustering data
    X_clust, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Classification examples
    axes[0, 0].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Classification Data', fontweight='bold')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Regression example
    axes[0, 1].scatter(X_reg, y_reg, alpha=0.7)
    axes[0, 1].set_title('Regression Data', fontweight='bold')
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Target')
    
    # Clustering example
    axes[0, 2].scatter(X_clust[:, 0], X_clust[:, 1], alpha=0.7)
    axes[0, 2].set_title('Clustering Data', fontweight='bold')
    axes[0, 2].set_xlabel('Feature 1')
    axes[0, 2].set_ylabel('Feature 2')
    
    # Algorithm comparison table
    algorithms = ['Linear Regression', 'Logistic Regression', 'Random Forest', 'SVM', 'K-Means', 'Neural Networks']
    accuracy = [0.75, 0.82, 0.89, 0.85, 0.78, 0.91]
    training_time = [0.1, 0.2, 0.8, 0.5, 0.3, 2.0]
    interpretability = [0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
    
    # Accuracy comparison
    bars1 = axes[1, 0].bar(algorithms, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    axes[1, 0].set_title('Accuracy Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Training time comparison
    bars2 = axes[1, 1].bar(algorithms, training_time, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    axes[1, 1].set_title('Training Time (seconds)', fontweight='bold')
    axes[1, 1].set_ylabel('Time')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Interpretability comparison
    bars3 = axes[1, 2].bar(algorithms, interpretability, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    axes[1, 2].set_title('Interpretability Score', fontweight='bold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_bias_variance_tradeoff():
    """Create bias-variance tradeoff visualization."""
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    true_function = 2 * x + 1 + 0.5 * np.sin(x)
    
    # Add noise
    y_noisy = true_function + np.random.normal(0, 1, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # High bias (underfitting)
    axes[0].scatter(x, y_noisy, alpha=0.6, label='Data')
    axes[0].plot(x, 1.5 * np.ones_like(x), 'r-', linewidth=3, label='Model (High Bias)')
    axes[0].plot(x, true_function, 'g--', linewidth=2, label='True Function')
    axes[0].set_title('High Bias (Underfitting)', fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Good fit
    axes[1].scatter(x, y_noisy, alpha=0.6, label='Data')
    axes[1].plot(x, true_function, 'r-', linewidth=3, label='Model (Good Fit)')
    axes[1].set_title('Good Fit (Optimal)', fontweight='bold')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # High variance (overfitting)
    # Create a complex polynomial fit
    coeffs = np.polyfit(x, y_noisy, 15)
    poly_fit = np.polyval(coeffs, x)
    
    axes[2].scatter(x, y_noisy, alpha=0.6, label='Data')
    axes[2].plot(x, poly_fit, 'r-', linewidth=3, label='Model (High Variance)')
    axes[2].plot(x, true_function, 'g--', linewidth=2, label='True Function')
    axes[2].set_title('High Variance (Overfitting)', fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bias_variance_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_learning_curves():
    """Create learning curves showing training vs validation."""
    # Simulate learning curves
    train_sizes = np.linspace(0.1, 1.0, 20)
    
    # Underfitting case
    train_score_under = 0.6 + 0.1 * train_sizes + 0.05 * np.random.randn(20)
    val_score_under = 0.58 + 0.08 * train_sizes + 0.08 * np.random.randn(20)
    
    # Good fit case
    train_score_good = 0.7 + 0.25 * train_sizes + 0.02 * np.random.randn(20)
    val_score_good = 0.68 + 0.23 * train_sizes + 0.03 * np.random.randn(20)
    
    # Overfitting case
    train_score_over = 0.8 + 0.15 * train_sizes + 0.01 * np.random.randn(20)
    val_score_over = 0.75 + 0.1 * train_sizes + 0.05 * np.random.randn(20)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Underfitting
    axes[0].plot(train_sizes, train_score_under, 'b-', linewidth=2, label='Training Score')
    axes[0].plot(train_sizes, val_score_under, 'r-', linewidth=2, label='Validation Score')
    axes[0].set_title('Underfitting', fontweight='bold')
    axes[0].set_xlabel('Training Set Size')
    axes[0].set_ylabel('Score')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Good fit
    axes[1].plot(train_sizes, train_score_good, 'b-', linewidth=2, label='Training Score')
    axes[1].plot(train_sizes, val_score_good, 'r-', linewidth=2, label='Validation Score')
    axes[1].set_title('Good Fit', fontweight='bold')
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_ylabel('Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Overfitting
    axes[2].plot(train_sizes, train_score_over, 'b-', linewidth=2, label='Training Score')
    axes[2].plot(train_sizes, val_score_over, 'r-', linewidth=2, label='Validation Score')
    axes[2].set_title('Overfitting', fontweight='bold')
    axes[2].set_xlabel('Training Set Size')
    axes[2].set_ylabel('Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_all_visualizations():
    """Generate all machine learning visualizations."""
    print("Generating machine learning visualizations...")
    
    print("1. Creating ML types diagram...")
    create_ml_types_diagram()
    
    print("2. Creating ML workflow diagram...")
    create_ml_workflow_diagram()
    
    print("3. Creating algorithm comparison...")
    create_algorithm_comparison()
    
    print("4. Creating bias-variance tradeoff...")
    create_bias_variance_tradeoff()
    
    print("5. Creating learning curves...")
    create_learning_curves()
    
    print("\nAll machine learning visualizations generated successfully!")
    print("Images saved as:")
    print("- ml_types_diagram.png")
    print("- ml_workflow_diagram.png")
    print("- algorithm_comparison.png")
    print("- bias_variance_tradeoff.png")
    print("- learning_curves.png")


if __name__ == "__main__":
    create_all_visualizations() 