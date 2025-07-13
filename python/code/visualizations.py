"""
Python Visualizations for AI/ML/Data Science
===========================================

This module creates various diagrams and visualizations for Python concepts.
Run this to generate images that can be used in the book documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def create_python_ecosystem_diagram():
    """Create a diagram showing Python ecosystem for AI/ML/Data Science."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define categories and packages
    categories = {
        'Data Manipulation': ['NumPy', 'Pandas', 'SciPy'],
        'Machine Learning': ['Scikit-learn', 'TensorFlow', 'PyTorch'],
        'Visualization': ['Matplotlib', 'Seaborn', 'Plotly'],
        'Deep Learning': ['Keras', 'TensorFlow', 'PyTorch'],
        'NLP': ['NLTK', 'spaCy', 'Transformers'],
        'Jupyter': ['Notebook', 'Lab', 'Hub'],
        'Deployment': ['Flask', 'FastAPI', 'Docker']
    }
    
    # Colors for categories
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # Position packages in a circular layout
    center_x, center_y = 0, 0
    radius = 3
    
    for i, (category, packages) in enumerate(categories.items()):
        angle = 2 * np.pi * i / len(categories)
        cat_x = center_x + radius * np.cos(angle)
        cat_y = center_y + radius * np.sin(angle)
        
        # Draw category circle
        circle = Circle((cat_x, cat_y), 0.8, facecolor=colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(cat_x, cat_y, category, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw packages around category
        for j, package in enumerate(packages):
            pkg_angle = angle + (j - 1) * 0.3
            pkg_x = cat_x + 1.5 * np.cos(pkg_angle)
            pkg_y = cat_y + 1.5 * np.sin(pkg_angle)
            
            # Package rectangle
            rect = FancyBboxPatch(
                (pkg_x - 0.4, pkg_y - 0.2), 0.8, 0.4,
                boxstyle="round,pad=0.1", facecolor='white', 
                edgecolor=colors[i], linewidth=1.5
            )
            ax.add_patch(rect)
            ax.text(pkg_x, pkg_y, package, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Connect package to category
            arrow = FancyArrowPatch(
                (pkg_x, pkg_y), (cat_x, cat_y),
                arrowstyle='->', mutation_scale=15, linewidth=1, alpha=0.6
            )
            ax.add_patch(arrow)
    
    # Add Python logo in center
    center_circle = Circle((center_x, center_y), 0.5, facecolor='#3776AB', edgecolor='black', linewidth=2)
    ax.add_patch(center_circle)
    ax.text(center_x, center_y, 'Python', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Python Ecosystem for AI/ML/Data Science', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('python_ecosystem.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_data_structures_diagram():
    """Create a diagram showing Python data structures."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define data structures and their properties
    structures = {
        'List': {'mutable': True, 'ordered': True, 'indexed': True, 'duplicates': True},
        'Tuple': {'mutable': False, 'ordered': True, 'indexed': True, 'duplicates': True},
        'Set': {'mutable': True, 'ordered': False, 'indexed': False, 'duplicates': False},
        'Dictionary': {'mutable': True, 'ordered': True, 'indexed': False, 'duplicates': False}
    }
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Position structures
    positions = [(2, 3), (6, 3), (2, 1), (6, 1)]
    
    for i, (structure, properties) in enumerate(structures.items()):
        x, y = positions[i]
        color = colors[i]
        
        # Main structure box
        rect = FancyBboxPatch(
            (x - 1, y - 0.5), 2, 1,
            boxstyle="round,pad=0.1", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, structure, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Properties
        props = []
        if properties['mutable']:
            props.append('Mutable')
        else:
            props.append('Immutable')
        
        if properties['ordered']:
            props.append('Ordered')
        else:
            props.append('Unordered')
        
        if properties['indexed']:
            props.append('Indexed')
        else:
            props.append('Key-based')
        
        if properties['duplicates']:
            props.append('Duplicates OK')
        else:
            props.append('No Duplicates')
        
        # Property boxes
        for j, prop in enumerate(props):
            prop_x = x - 1.5 + j * 0.75
            prop_y = y - 1.2
            prop_rect = FancyBboxPatch(
                (prop_x - 0.3, prop_y - 0.15), 0.6, 0.3,
                boxstyle="round,pad=0.05", facecolor='white', 
                edgecolor=color, linewidth=1
            )
            ax.add_patch(prop_rect)
            ax.text(prop_x, prop_y, prop, ha='center', va='center', fontsize=8)
    
    # Add title and legend
    ax.text(4, 4.5, 'Python Data Structures', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Example usage
    examples = [
        'List: [1, 2, 3, 2]',
        'Tuple: (1, 2, 3)',
        'Set: {1, 2, 3}',
        'Dict: {"a": 1, "b": 2}'
    ]
    
    for i, example in enumerate(examples):
        ax.text(0.5, 3.5 - i * 0.3, example, fontsize=10, fontfamily='monospace')
    
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_structures.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_workflow_diagram():
    """Create a diagram showing typical AI/ML workflow."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define workflow steps
    steps = [
        {'name': 'Data Collection', 'pos': (1, 4), 'color': '#FF6B6B'},
        {'name': 'Data Cleaning', 'pos': (3, 4), 'color': '#4ECDC4'},
        {'name': 'Exploratory\nAnalysis', 'pos': (5, 4), 'color': '#45B7D1'},
        {'name': 'Feature\nEngineering', 'pos': (7, 4), 'color': '#96CEB4'},
        {'name': 'Model Selection', 'pos': (9, 4), 'color': '#FFEAA7'},
        {'name': 'Training', 'pos': (11, 4), 'color': '#DDA0DD'},
        {'name': 'Evaluation', 'pos': (13, 4), 'color': '#98D8C8'},
        {'name': 'Deployment', 'pos': (15, 4), 'color': '#F7DC6F'}
    ]
    
    # Draw steps
    for step in steps:
        x, y = step['pos']
        color = step['color']
        
        # Step box
        rect = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.1", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows connecting steps
    for i in range(len(steps) - 1):
        x1, y1 = steps[i]['pos']
        x2, y2 = steps[i + 1]['pos']
        
        arrow = FancyArrowPatch(
            (x1 + 0.8, y1), (x2 - 0.8, y2),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add feedback loops
    # Evaluation to Feature Engineering
    arrow1 = FancyArrowPatch(
        (13, 3.6), (7, 3.6),
        arrowstyle='->', mutation_scale=20, linewidth=2, color='red', linestyle='--'
    )
    ax.add_patch(arrow1)
    ax.text(10, 3.3, 'Iterate', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # Evaluation to Model Selection
    arrow2 = FancyArrowPatch(
        (13, 3.4), (9, 3.4),
        arrowstyle='->', mutation_scale=20, linewidth=2, color='red', linestyle='--'
    )
    ax.add_patch(arrow2)
    
    # Add tools/libraries
    tools = [
        'Pandas, NumPy', 'Pandas, NumPy', 'Matplotlib, Seaborn',
        'Scikit-learn', 'Scikit-learn', 'TensorFlow, PyTorch',
        'Scikit-learn', 'Flask, FastAPI'
    ]
    
    for i, (step, tool) in enumerate(zip(steps, tools)):
        x, y = step['pos']
        ax.text(x, y - 0.8, tool, ha='center', va='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 16)
    ax.set_ylim(2, 5)
    ax.axis('off')
    ax.set_title('AI/ML Workflow with Python Tools', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('workflow_diagram.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_learning_path_diagram():
    """Create a diagram showing Python learning path for AI/ML."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define learning stages
    stages = [
        {'name': 'Python Basics', 'level': 1, 'topics': ['Syntax', 'Variables', 'Control Flow']},
        {'name': 'Data Structures', 'level': 2, 'topics': ['Lists', 'Dictionaries', 'Sets']},
        {'name': 'Functions & OOP', 'level': 3, 'topics': ['Functions', 'Classes', 'Modules']},
        {'name': 'Data Science\nLibraries', 'level': 4, 'topics': ['NumPy', 'Pandas', 'Matplotlib']},
        {'name': 'Machine Learning', 'level': 5, 'topics': ['Scikit-learn', 'Modeling', 'Evaluation']},
        {'name': 'Deep Learning', 'level': 6, 'topics': ['TensorFlow', 'PyTorch', 'Neural Networks']},
        {'name': 'Advanced Topics', 'level': 7, 'topics': ['NLP', 'Computer Vision', 'Deployment']}
    ]
    
    # Colors for levels
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    # Draw stages
    for i, stage in enumerate(stages):
        level = stage['level']
        x = 2 + (level - 1) * 1.5
        y = 8 - (level - 1) * 1.2
        color = colors[i]
        
        # Stage box
        rect = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.1", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, stage['name'], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Topics
        for j, topic in enumerate(stage['topics']):
            topic_x = x - 0.6 + j * 0.4
            topic_y = y - 1.2
            topic_rect = FancyBboxPatch(
                (topic_x - 0.15, topic_y - 0.1), 0.3, 0.2,
                boxstyle="round,pad=0.02", facecolor='white', 
                edgecolor=color, linewidth=1
            )
            ax.add_patch(topic_rect)
            ax.text(topic_x, topic_y, topic, ha='center', va='center', fontsize=7)
    
    # Draw connections
    for i in range(len(stages) - 1):
        level1 = stages[i]['level']
        level2 = stages[i + 1]['level']
        x1 = 2 + (level1 - 1) * 1.5
        y1 = 8 - (level1 - 1) * 1.2
        x2 = 2 + (level2 - 1) * 1.5
        y2 = 8 - (level2 - 1) * 1.2
        
        arrow = FancyArrowPatch(
            (x1, y1 - 0.4), (x2, y2 + 0.4),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add time estimates
    time_estimates = ['2-4 weeks', '2-3 weeks', '3-4 weeks', '4-6 weeks', '6-8 weeks', '8-12 weeks', 'Ongoing']
    for i, stage in enumerate(stages):
        level = stage['level']
        x = 2 + (level - 1) * 1.5
        y = 8 - (level - 1) * 1.2
        ax.text(x, y + 0.6, time_estimates[i], ha='center', va='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_title('Python Learning Path for AI/ML/Data Science', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('learning_path.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_all_visualizations():
    """Generate all Python visualizations."""
    print("Generating Python visualizations...")
    
    print("1. Creating Python ecosystem diagram...")
    create_python_ecosystem_diagram()
    
    print("2. Creating data structures diagram...")
    create_data_structures_diagram()
    
    print("3. Creating workflow diagram...")
    create_workflow_diagram()
    
    print("4. Creating learning path diagram...")
    create_learning_path_diagram()
    
    print("\nAll Python visualizations generated successfully!")
    print("Images saved as:")
    print("- python_ecosystem.png")
    print("- data_structures.png")
    print("- workflow_diagram.png")
    print("- learning_path.png")


if __name__ == "__main__":
    create_all_visualizations() 