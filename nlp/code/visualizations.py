"""
NLP Visualizations
=================

This module creates various diagrams and visualizations for Natural Language Processing concepts.
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


def create_nlp_pipeline_diagram():
    """Create a diagram showing the NLP processing pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Define pipeline steps
    pipeline = [
        {
            'name': 'Text Input',
            'pos': (1, 4),
            'color': '#FF6B6B',
            'example': '"Hello world!"'
        },
        {
            'name': 'Tokenization',
            'pos': (3, 4),
            'color': '#4ECDC4',
            'example': '["Hello", "world", "!"]'
        },
        {
            'name': 'Text Cleaning',
            'pos': (5, 4),
            'color': '#45B7D1',
            'example': 'Lowercase, remove punctuation'
        },
        {
            'name': 'Stop Word\nRemoval',
            'pos': (7, 4),
            'color': '#96CEB4',
            'example': 'Remove common words'
        },
        {
            'name': 'Stemming/\nLemmatization',
            'pos': (9, 4),
            'color': '#FFEAA7',
            'example': 'running → run'
        },
        {
            'name': 'Feature\nExtraction',
            'pos': (11, 4),
            'color': '#DDA0DD',
            'example': 'TF-IDF, Word2Vec'
        },
        {
            'name': 'Model\nProcessing',
            'pos': (13, 4),
            'color': '#98D8C8',
            'example': 'Classification, NER'
        },
        {
            'name': 'Output',
            'pos': (15, 4),
            'color': '#F7DC6F',
            'example': 'Results, predictions'
        }
    ]
    
    # Draw pipeline steps
    for step in pipeline:
        x, y = step['pos']
        color = step['color']
        
        # Main step box
        rect = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.05", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, step['name'], ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Example
        ax.text(x, y - 0.8, step['example'], ha='center', va='center', fontsize=8, style='italic')
    
    # Connect steps
    for i in range(len(pipeline) - 1):
        x1, y1 = pipeline[i]['pos']
        x2, y2 = pipeline[i + 1]['pos']
        
        arrow = FancyArrowPatch(
            (x1 + 0.8, y1), (x2 - 0.8, y2),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add tools/libraries
    tools = [
        'Raw Text', 'NLTK, spaCy', 'Regex, NLTK',
        'NLTK, spaCy', 'NLTK, spaCy', 'Scikit-learn',
        'ML Models', 'Results'
    ]
    
    for i, (step, tool) in enumerate(zip(pipeline, tools)):
        x, y = step['pos']
        ax.text(x, y - 1.2, tool, ha='center', va='center', fontsize=8, style='italic')
    
    ax.set_xlim(0, 16)
    ax.set_ylim(2, 6)
    ax.axis('off')
    ax.set_title('NLP Processing Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('nlp_pipeline.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_word_embeddings_diagram():
    """Create a diagram showing word embeddings concept."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Sample words and their embeddings
    words = ['king', 'queen', 'man', 'woman', 'cat', 'dog']
    embeddings = np.array([
        [0.8, 0.9, 0.1],  # king
        [0.7, 0.8, 0.2],  # queen
        [0.9, 0.1, 0.8],  # man
        [0.8, 0.2, 0.7],  # woman
        [0.1, 0.1, 0.9],  # cat
        [0.2, 0.1, 0.8]   # dog
    ])
    
    # Create 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot words
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i, (word, emb, color) in enumerate(zip(words, embeddings, colors)):
        ax.scatter(emb[0], emb[1], emb[2], c=color, s=100, alpha=0.8)
        ax.text(emb[0], emb[1], emb[2], word, fontsize=12, fontweight='bold')
    
    # Add arrows showing relationships
    # king - man + woman = queen
    ax.quiver(embeddings[0, 0], embeddings[0, 1], embeddings[0, 2],
              embeddings[2, 0] - embeddings[0, 0], embeddings[2, 1] - embeddings[0, 1], embeddings[2, 2] - embeddings[0, 2],
              color='red', alpha=0.7, arrow_length_ratio=0.1)
    
    ax.quiver(embeddings[2, 0], embeddings[2, 1], embeddings[2, 2],
              embeddings[3, 0] - embeddings[2, 0], embeddings[3, 1] - embeddings[2, 1], embeddings[3, 2] - embeddings[2, 2],
              color='blue', alpha=0.7, arrow_length_ratio=0.1)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('Word Embeddings in 3D Space', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('word_embeddings.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_attention_mechanism_diagram():
    """Create a diagram showing attention mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Define input words
    input_words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    query_word = 'cat'
    
    # Position words
    word_positions = [(1, 7), (3, 7), (5, 7), (7, 7), (9, 7), (11, 7)]
    
    # Draw input words
    for i, (word, pos) in enumerate(zip(input_words, word_positions)):
        x, y = pos
        color = '#FF6B6B' if word == query_word else '#4ECDC4'
        
        # Word box
        rect = FancyBboxPatch(
            (x - 0.4, y - 0.2), 0.8, 0.4,
            boxstyle="round,pad=0.05", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, word, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw attention weights
    attention_weights = [0.1, 0.8, 0.05, 0.02, 0.01, 0.02]  # Focus on 'cat'
    
    # Query word (cat) at bottom
    query_x, query_y = 3, 3
    query_rect = FancyBboxPatch(
        (query_x - 0.4, query_y - 0.2), 0.8, 0.4,
        boxstyle="round,pad=0.05", facecolor='#FF6B6B', 
        edgecolor='black', linewidth=2
    )
    ax.add_patch(query_rect)
    ax.text(query_x, query_y, query_word, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(query_x, query_y - 0.5, 'Query', ha='center', va='center', fontsize=8, style='italic')
    
    # Draw attention connections
    for i, (pos, weight) in enumerate(zip(word_positions, attention_weights)):
        x, y = pos
        alpha = weight * 0.8 + 0.2  # Normalize alpha
        width = weight * 3 + 0.5    # Normalize width
        
        arrow = FancyArrowPatch(
            (query_x, query_y + 0.2), (x, y - 0.2),
            arrowstyle='->', mutation_scale=20, linewidth=width, 
            color='red', alpha=alpha
        )
        ax.add_patch(arrow)
        
        # Add weight label
        mid_x = (query_x + x) / 2
        mid_y = (query_y + y) / 2
        ax.text(mid_x, mid_y, f'{weight:.2f}', ha='center', va='center', 
                fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add attention formula
    ax.text(7, 1, 'Attention(Q,K,V) = softmax(QK^T/√d_k)V', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('Attention Mechanism', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('attention_mechanism.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_transformer_architecture():
    """Create a diagram showing transformer architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Define transformer components
    components = [
        {'name': 'Input\nEmbedding', 'pos': (2, 10), 'color': '#FF6B6B'},
        {'name': 'Positional\nEncoding', 'pos': (2, 8.5), 'color': '#4ECDC4'},
        {'name': 'Multi-Head\nAttention', 'pos': (5, 9), 'color': '#45B7D1'},
        {'name': 'Add & Norm', 'pos': (5, 7.5), 'color': '#96CEB4'},
        {'name': 'Feed\nForward', 'pos': (8, 9), 'color': '#FFEAA7'},
        {'name': 'Add & Norm', 'pos': (8, 7.5), 'color': '#96CEB4'},
        {'name': 'Output\nEmbedding', 'pos': (11, 10), 'color': '#DDA0DD'},
        {'name': 'Positional\nEncoding', 'pos': (11, 8.5), 'color': '#4ECDC4'},
        {'name': 'Masked\nAttention', 'pos': (14, 9), 'color': '#98D8C8'},
        {'name': 'Add & Norm', 'pos': (14, 7.5), 'color': '#96CEB4'},
        {'name': 'Cross\nAttention', 'pos': (17, 9), 'color': '#F7DC6F'},
        {'name': 'Add & Norm', 'pos': (17, 7.5), 'color': '#96CEB4'},
        {'name': 'Feed\nForward', 'pos': (20, 9), 'color': '#FFEAA7'},
        {'name': 'Add & Norm', 'pos': (20, 7.5), 'color': '#96CEB4'},
        {'name': 'Linear\nLayer', 'pos': (23, 9), 'color': '#BB8FCE'},
        {'name': 'Softmax', 'pos': (23, 7.5), 'color': '#85C1E9'}
    ]
    
    # Draw components
    for component in components:
        x, y = component['pos']
        color = component['color']
        
        # Component box
        rect = FancyBboxPatch(
            (x - 0.8, y - 0.4), 1.6, 0.8,
            boxstyle="round,pad=0.05", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y, component['name'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connections
    connections = [
        ((2, 9.6), (5, 9.4)),  # Input to attention
        ((5, 9.6), (8, 9.4)),  # Attention to FF
        ((8, 9.6), (11, 9.4)), # FF to output
        ((11, 9.6), (14, 9.4)), # Output to masked attention
        ((14, 9.6), (17, 9.4)), # Masked to cross attention
        ((17, 9.6), (20, 9.4)), # Cross to FF
        ((20, 9.6), (23, 9.4)), # FF to linear
        ((23, 9.6), (25, 9.4)), # Linear to output
    ]
    
    for (x1, y1), (x2, y2) in connections:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add residual connections
    residual_connections = [
        ((2, 9.6), (5, 7.1)),  # Input to Add&Norm
        ((5, 7.1), (8, 7.1)),  # Add&Norm to Add&Norm
        ((8, 7.1), (11, 7.1)), # Add&Norm to Add&Norm
        ((11, 9.6), (14, 7.1)), # Output to Add&Norm
        ((14, 7.1), (17, 7.1)), # Add&Norm to Add&Norm
        ((17, 7.1), (20, 7.1)), # Add&Norm to Add&Norm
        ((20, 7.1), (23, 7.1)), # Add&Norm to Add&Norm
    ]
    
    for (x1, y1), (x2, y2) in residual_connections:
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=15, linewidth=2, color='red', linestyle='--'
        )
        ax.add_patch(arrow)
    
    # Add labels
    ax.text(2, 11, 'Encoder', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax.text(11, 11, 'Decoder', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Add input/output labels
    ax.text(1, 9.6, 'Input', ha='right', va='center', fontsize=10, fontweight='bold')
    ax.text(25, 9.4, 'Output', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 26)
    ax.set_ylim(6, 12)
    ax.axis('off')
    ax.set_title('Transformer Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('transformer_architecture.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_text_classification_diagram():
    """Create a diagram showing text classification process."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Sample text and processing steps
    sample_text = "I love this movie! It's amazing."
    
    # Processing steps
    steps = [
        {'name': 'Raw Text', 'text': sample_text, 'pos': (2, 6), 'color': '#FF6B6B'},
        {'name': 'Tokenization', 'text': '["I", "love", "this", "movie", "!", "It", "\'s", "amazing", "."]', 'pos': (5, 6), 'color': '#4ECDC4'},
        {'name': 'Cleaning', 'text': '["love", "movie", "amazing"]', 'pos': (8, 6), 'color': '#45B7D1'},
        {'name': 'Vectorization', 'text': '[0.2, 0.8, 0.1, 0.9, 0.0, 0.0, 0.0, 0.7, 0.0]', 'pos': (11, 6), 'color': '#96CEB4'},
        {'name': 'Classification', 'text': 'Positive (0.85)', 'pos': (14, 6), 'color': '#FFEAA7'}
    ]
    
    # Draw processing steps
    for i, step in enumerate(steps):
        x, y = step['pos']
        color = step['color']
        
        # Step box
        rect = FancyBboxPatch(
            (x - 1.2, y - 0.8), 2.4, 1.6,
            boxstyle="round,pad=0.1", facecolor=color, 
            edgecolor='black', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.3, step['name'], ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x, y - 0.1, step['text'], ha='center', va='center', fontsize=8, fontfamily='monospace')
    
    # Connect steps
    for i in range(len(steps) - 1):
        x1, y1 = steps[i]['pos']
        x2, y2 = steps[i + 1]['pos']
        
        arrow = FancyArrowPatch(
            (x1 + 1.2, y1), (x2 - 1.2, y2),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='black'
        )
        ax.add_patch(arrow)
    
    # Add model details
    model_info = [
        'Model: BERT',
        'Task: Sentiment Analysis',
        'Classes: Positive, Negative, Neutral',
        'Accuracy: 92%'
    ]
    
    for i, info in enumerate(model_info):
        ax.text(2, 4 - i * 0.3, info, fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 16)
    ax.set_ylim(2, 8)
    ax.axis('off')
    ax.set_title('Text Classification Pipeline', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('text_classification.png', bbox_inches='tight', dpi=300)
    plt.show()


def create_all_visualizations():
    """Generate all NLP visualizations."""
    print("Generating NLP visualizations...")
    
    print("1. Creating NLP pipeline diagram...")
    create_nlp_pipeline_diagram()
    
    print("2. Creating word embeddings diagram...")
    create_word_embeddings_diagram()
    
    print("3. Creating attention mechanism diagram...")
    create_attention_mechanism_diagram()
    
    print("4. Creating transformer architecture...")
    create_transformer_architecture()
    
    print("5. Creating text classification diagram...")
    create_text_classification_diagram()
    
    print("\nAll NLP visualizations generated successfully!")
    print("Images saved as:")
    print("- nlp_pipeline.png")
    print("- word_embeddings.png")
    print("- attention_mechanism.png")
    print("- transformer_architecture.png")
    print("- text_classification.png")


if __name__ == "__main__":
    create_all_visualizations() 