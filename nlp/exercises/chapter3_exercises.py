"""
Chapter 3 Exercises: Attention Mechanisms and Transformers
========================================================

Practice exercises for Transformer concepts including:
- Attention mechanism implementation
- Multi-head attention
- Positional encoding
- Transformer architecture
- Attention patterns and applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math

# Import the Transformer code from the examples
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from chapter3_transformers import SimpleAttention, MultiHeadAttention, PositionalEncoding


class Chapter3Exercises:
    """Exercise problems for Chapter 3."""
    
    def __init__(self):
        pass
    
    def exercise_3_1_attention_implementation(self, query: np.ndarray, key: np.ndarray, 
                                           value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exercise 3.1: Implement attention mechanism from scratch.
        
        Requirements:
        - Compute attention scores using dot product
        - Apply scaling factor (sqrt of embedding dimension)
        - Apply softmax to get attention weights
        - Compute weighted sum of values
        - Handle edge cases (zero division, numerical stability)
        
        Args:
            query: Query matrix (seq_len_q, d_model)
            key: Key matrix (seq_len_k, d_model)
            value: Value matrix (seq_len_k, d_model)
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        # TODO: Implement your attention mechanism here
        # Replace this with your implementation
        pass
    
    def exercise_3_2_multi_head_attention(self, query: np.ndarray, key: np.ndarray, 
                                        value: np.ndarray, num_heads: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exercise 3.2: Implement multi-head attention.
        
        Requirements:
        - Split input into multiple heads
        - Apply attention to each head separately
        - Concatenate head outputs
        - Apply final linear transformation
        - Handle different sequence lengths for query and key
        
        Args:
            query: Query matrix (seq_len_q, d_model)
            key: Key matrix (seq_len_k, d_model)
            value: Value matrix (seq_len_k, d_model)
            num_heads: Number of attention heads
            
        Returns:
            output: Multi-head attention output
            attention_weights: Average attention weights across heads
        """
        # TODO: Implement your multi-head attention here
        # Replace this with your implementation
        pass
    
    def exercise_3_3_positional_encoding(self, seq_len: int, d_model: int) -> np.ndarray:
        """
        Exercise 3.3: Implement positional encoding.
        
        Requirements:
        - Use sine and cosine functions
        - Different frequencies for different dimensions
        - Handle even and odd dimensions differently
        - Ensure unique encoding for each position
        
        Args:
            seq_len: Length of the sequence
            d_model: Embedding dimension
            
        Returns:
            positional_encoding: Matrix of shape (seq_len, d_model)
        """
        # TODO: Implement your positional encoding here
        # Replace this with your implementation
        pass
    
    def exercise_3_4_causal_attention_mask(self, seq_len: int) -> np.ndarray:
        """
        Exercise 3.4: Create causal attention mask for autoregressive models.
        
        Requirements:
        - Create upper triangular mask
        - Allow attention to current and previous positions only
        - Use large negative values for masked positions
        - Ensure proper shape for attention computation
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            mask: Causal attention mask of shape (seq_len, seq_len)
        """
        # TODO: Implement your causal attention mask here
        # Replace this with your implementation
        pass
    
    def exercise_3_5_attention_visualization(self, attention_weights: np.ndarray, 
                                          words: List[str]) -> None:
        """
        Exercise 3.5: Create attention visualization.
        
        Requirements:
        - Create heatmap of attention weights
        - Add proper labels for words
        - Use appropriate colormap
        - Add title and axis labels
        - Handle different attention patterns
        
        Args:
            attention_weights: Attention weight matrix
            words: List of words corresponding to positions
        """
        # TODO: Implement your attention visualization here
        # Replace this with your implementation
        pass
    
    def exercise_3_6_attention_pattern_analysis(self, attention_weights: np.ndarray) -> dict:
        """
        Exercise 3.6: Analyze attention patterns.
        
        Requirements:
        - Calculate average attention distance
        - Identify local vs global attention patterns
        - Find positions with highest attention entropy
        - Detect attention heads that focus on specific patterns
        
        Args:
            attention_weights: Attention weight matrix
            
        Returns:
            analysis: Dictionary with analysis results
        """
        # TODO: Implement your attention pattern analysis here
        # Replace this with your implementation
        pass
    
    def exercise_3_7_transformer_layer(self, input_embeddings: np.ndarray, 
                                     d_model: int, num_heads: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exercise 3.7: Implement a single Transformer layer.
        
        Requirements:
        - Multi-head self-attention
        - Residual connections
        - Layer normalization (simplified)
        - Feed-forward network
        - Handle the complete forward pass
        
        Args:
            input_embeddings: Input embeddings (seq_len, d_model)
            d_model: Model dimension
            num_heads: Number of attention heads
            
        Returns:
            output: Layer output
            attention_weights: Attention weights
        """
        # TODO: Implement your Transformer layer here
        # Replace this with your implementation
        pass
    
    def exercise_3_8_attention_interpretation(self, attention_weights: np.ndarray, 
                                            words: List[str]) -> List[str]:
        """
        Exercise 3.8: Interpret attention patterns.
        
        Requirements:
        - Identify which words attend to which other words
        - Find attention patterns (e.g., attending to previous word, next word)
        - Detect syntactic vs semantic attention
        - Generate natural language descriptions of patterns
        
        Args:
            attention_weights: Attention weight matrix
            words: List of words
            
        Returns:
            interpretations: List of pattern descriptions
        """
        # TODO: Implement your attention interpretation here
        # Replace this with your implementation
        pass


def run_exercises():
    """Run all exercises with sample data."""
    
    exercises = Chapter3Exercises()
    
    print("=== Chapter 3 Exercises ===\n")
    
    # Exercise 3.1: Attention Implementation
    print("Exercise 3.1 - Attention Implementation:")
    seq_len = 5
    d_model = 8
    
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = np.random.randn(seq_len, d_model)
    
    try:
        output, weights = exercises.exercise_3_1_attention_implementation(query, key, value)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Attention weights shape: {weights.shape}")
        print(f"✓ Attention weights sum to 1: {np.allclose(weights.sum(axis=1), 1.0)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.2: Multi-Head Attention
    print("Exercise 3.2 - Multi-Head Attention:")
    num_heads = 4
    
    try:
        output, weights = exercises.exercise_3_2_multi_head_attention(query, key, value, num_heads)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Number of heads: {num_heads}")
        print(f"✓ Dimension per head: {d_model // num_heads}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.3: Positional Encoding
    print("Exercise 3.3 - Positional Encoding:")
    seq_len = 10
    d_model = 16
    
    try:
        pos_encoding = exercises.exercise_3_3_positional_encoding(seq_len, d_model)
        print(f"✓ Positional encoding shape: {pos_encoding.shape}")
        print(f"✓ Unique positions: {len(np.unique(pos_encoding, axis=0))}")
        print(f"✓ Encoding range: [{pos_encoding.min():.3f}, {pos_encoding.max():.3f}]")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.4: Causal Attention Mask
    print("Exercise 3.4 - Causal Attention Mask:")
    seq_len = 6
    
    try:
        mask = exercises.exercise_3_4_causal_attention_mask(seq_len)
        print(f"✓ Mask shape: {mask.shape}")
        print(f"✓ Upper triangular: {np.allclose(mask, np.triu(mask))}")
        print(f"✓ Mask values: {np.unique(mask)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.5: Attention Visualization
    print("Exercise 3.5 - Attention Visualization:")
    attention_weights = np.array([
        [0.8, 0.1, 0.1, 0.0],
        [0.1, 0.7, 0.1, 0.1],
        [0.0, 0.1, 0.8, 0.1],
        [0.0, 0.0, 0.1, 0.9]
    ])
    words = ["the", "cat", "sat", "down"]
    
    try:
        exercises.exercise_3_5_attention_visualization(attention_weights, words)
        print("✓ Visualization created successfully")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.6: Attention Pattern Analysis
    print("Exercise 3.6 - Attention Pattern Analysis:")
    
    try:
        analysis = exercises.exercise_3_6_attention_pattern_analysis(attention_weights)
        print(f"✓ Analysis completed: {list(analysis.keys())}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.7: Transformer Layer
    print("Exercise 3.7 - Transformer Layer:")
    input_embeddings = np.random.randn(seq_len, d_model)
    
    try:
        output, weights = exercises.exercise_3_7_transformer_layer(input_embeddings, d_model, num_heads)
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Input and output dimensions match: {input_embeddings.shape == output.shape}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 3.8: Attention Interpretation
    print("Exercise 3.8 - Attention Interpretation:")
    
    try:
        interpretations = exercises.exercise_3_8_attention_interpretation(attention_weights, words)
        print(f"✓ Found {len(interpretations)} patterns")
        for i, interpretation in enumerate(interpretations[:3]):
            print(f"  Pattern {i+1}: {interpretation}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")


def provide_solutions():
    """Provide solution hints and examples."""
    
    print("=== Exercise Solutions and Hints ===\n")
    
    print("Exercise 3.1 - Attention Implementation Hints:")
    print("- Use np.dot() for matrix multiplication")
    print("- Scale by sqrt(d_model) to prevent large values")
    print("- Use np.exp() and np.sum() for softmax")
    print("- Handle numerical stability with max subtraction")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.2 - Multi-Head Attention Hints:")
    print("- Reshape input to (num_heads, seq_len, d_k)")
    print("- Apply attention to each head separately")
    print("- Concatenate outputs along the last dimension")
    print("- Apply final linear transformation")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.3 - Positional Encoding Hints:")
    print("- Use sin() for even dimensions, cos() for odd")
    print("- Frequency = 1 / (10000^(2i/d_model))")
    print("- Position ranges from 0 to seq_len-1")
    print("- Each position gets unique encoding")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.4 - Causal Attention Mask Hints:")
    print("- Use np.triu() to create upper triangular matrix")
    print("- Set masked positions to large negative values (-1e9)")
    print("- Allow attention to current and previous positions")
    print("- Shape should be (seq_len, seq_len)")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.5 - Attention Visualization Hints:")
    print("- Use plt.imshow() or sns.heatmap()")
    print("- Set xticklabels and yticklabels to words")
    print("- Use appropriate colormap (e.g., 'Blues')")
    print("- Add title and axis labels")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.6 - Attention Pattern Analysis Hints:")
    print("- Calculate attention distance: |i - j| for position i attending to j")
    print("- Use entropy to measure attention concentration")
    print("- Look for diagonal patterns (local attention)")
    print("- Identify heads with specific focus patterns")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.7 - Transformer Layer Hints:")
    print("- Apply multi-head self-attention")
    print("- Add residual connection: output = input + attention_output")
    print("- Apply feed-forward network: FFN(x) = W2 * ReLU(W1 * x)")
    print("- Add another residual connection")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 3.8 - Attention Interpretation Hints:")
    print("- Find positions with highest attention weights")
    print("- Look for syntactic patterns (adjacent words)")
    print("- Identify semantic relationships")
    print("- Generate descriptive text about patterns")
    print("\n" + "-"*40 + "\n")


def demonstrate_attention_concepts():
    """Demonstrate key attention concepts for reference."""
    
    print("=== Attention Concepts Demonstration ===\n")
    
    # Create sample data
    seq_len = 4
    d_model = 6
    
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = np.random.randn(seq_len, d_model)
    
    print("Sample Data:")
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    
    # Demonstrate attention computation
    print("\nAttention Computation Steps:")
    
    # 1. Compute attention scores
    scores = np.dot(query, key.T)
    print(f"1. Raw scores shape: {scores.shape}")
    print(f"   Sample scores:\n{scores[:2, :2]}")
    
    # 2. Scale scores
    scale = math.sqrt(d_model)
    scaled_scores = scores / scale
    print(f"\n2. Scaled scores (scale={scale:.3f}):")
    print(f"   Sample scaled scores:\n{scaled_scores[:2, :2]}")
    
    # 3. Apply softmax
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    print(f"\n3. Attention weights:")
    print(f"   Weights sum to 1: {np.allclose(attention_weights.sum(axis=1), 1.0)}")
    print(f"   Sample weights:\n{attention_weights[:2, :2]}")
    
    # 4. Compute output
    output = np.dot(attention_weights, value)
    print(f"\n4. Output shape: {output.shape}")
    print(f"   Sample output:\n{output[:2, :2]}")
    
    # Demonstrate different attention patterns
    print("\n" + "="*50)
    print("Different Attention Patterns:")
    
    # Local attention
    local_weights = np.eye(seq_len) * 0.7 + np.eye(seq_len, k=1) * 0.3
    print(f"\nLocal attention (diagonal + next position):")
    print(local_weights)
    
    # Global attention
    global_weights = np.ones((seq_len, seq_len)) / seq_len
    print(f"\nGlobal attention (uniform):")
    print(global_weights)
    
    # Causal attention
    causal_weights = np.tril(np.ones((seq_len, seq_len))) / np.arange(1, seq_len + 1).reshape(-1, 1)
    print(f"\nCausal attention (lower triangular):")
    print(causal_weights)


if __name__ == "__main__":
    # Run exercises
    run_exercises()
    
    # Provide solution hints
    provide_solutions()
    
    # Demonstrate concepts
    demonstrate_attention_concepts() 