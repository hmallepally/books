"""
Chapter 3: Transformers and Attention Mechanisms
===============================================

This module demonstrates Transformer concepts including:
- Attention mechanism implementation
- Simplified Transformer architecture
- Multi-head attention
- Positional encoding
- Practical applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class SimpleAttention:
    """Simplified attention mechanism for demonstration."""
    
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.scale = math.sqrt(d_model)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, 
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention.
        
        Args:
            query: Query matrix (seq_len_q, d_model)
            key: Key matrix (seq_len_k, d_model)
            value: Value matrix (seq_len_k, d_model)
            mask: Optional mask matrix
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        # Compute attention scores
        scores = np.dot(query, key.T) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(scores, axis=-1)
        
        # Compute weighted sum of values
        output = np.dot(attention_weights, value)
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class MultiHeadAttention:
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear transformations for Q, K, V
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute multi-head attention.
        
        Args:
            query: Query matrix (seq_len_q, d_model)
            key: Key matrix (seq_len_k, d_model)
            value: Value matrix (seq_len_k, d_model)
            mask: Optional mask matrix
            
        Returns:
            output: Multi-head attention output
            attention_weights: Attention weights
        """
        batch_size = 1  # Simplified for demonstration
        
        # Linear transformations
        Q = np.dot(query, self.W_q)  # (seq_len_q, d_model)
        K = np.dot(key, self.W_k)    # (seq_len_k, d_model)
        V = np.dot(value, self.W_v)  # (seq_len_k, d_model)
        
        # Reshape for multi-head attention
        Q = Q.reshape(-1, self.num_heads, self.d_k)  # (seq_len_q, num_heads, d_k)
        K = K.reshape(-1, self.num_heads, self.d_k)  # (seq_len_k, num_heads, d_k)
        V = V.reshape(-1, self.num_heads, self.d_k)  # (seq_len_k, num_heads, d_k)
        
        # Transpose for batch matrix multiplication
        Q = Q.transpose(1, 0, 2)  # (num_heads, seq_len_q, d_k)
        K = K.transpose(1, 0, 2)  # (num_heads, seq_len_k, d_k)
        V = V.transpose(1, 0, 2)  # (num_heads, seq_len_k, d_k)
        
        # Compute attention for each head
        attention_outputs = []
        attention_weights_list = []
        
        for h in range(self.num_heads):
            # Compute attention scores
            scores = np.dot(Q[h], K[h].T) / self.scale  # (seq_len_q, seq_len_k)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores + mask
            
            # Apply softmax
            attention_weights = self._softmax(scores, axis=-1)
            
            # Compute weighted sum
            head_output = np.dot(attention_weights, V[h])  # (seq_len_q, d_k)
            
            attention_outputs.append(head_output)
            attention_weights_list.append(attention_weights)
        
        # Concatenate heads
        attention_output = np.concatenate(attention_outputs, axis=-1)  # (seq_len_q, d_model)
        
        # Final linear transformation
        output = np.dot(attention_output, self.W_o)
        
        # Average attention weights across heads for visualization
        avg_attention_weights = np.mean(attention_weights_list, axis=0)
        
        return output, avg_attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PositionalEncoding:
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (seq_len, d_model)
            
        Returns:
            x + positional_encoding
        """
        seq_len = x.shape[0]
        return x + self.pe[:seq_len, :]


class SimplifiedTransformer:
    """Simplified Transformer for demonstration."""
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, max_len: int = 100):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Embeddings
        self.embedding = np.random.randn(vocab_size, d_model) * 0.1
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.attention_layers = []
        self.feed_forward_layers = []
        
        for _ in range(num_layers):
            self.attention_layers.append(MultiHeadAttention(d_model, num_heads))
            # Simplified feed-forward network
            self.feed_forward_layers.append({
                'W1': np.random.randn(d_model, d_model * 4) * 0.1,
                'W2': np.random.randn(d_model * 4, d_model) * 0.1
            })
        
        # Output layer
        self.output_layer = np.random.randn(d_model, vocab_size) * 0.1
    
    def forward(self, input_ids: List[int], mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward pass through the Transformer.
        
        Args:
            input_ids: Input token IDs
            mask: Optional attention mask
            
        Returns:
            output: Final output logits
            attention_weights: Attention weights from all layers
        """
        seq_len = len(input_ids)
        
        # Get embeddings
        embeddings = self.embedding[input_ids]  # (seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding.forward(embeddings)
        
        attention_weights = []
        
        # Pass through transformer layers
        for layer_idx in range(self.num_layers):
            # Self-attention
            attn_output, attn_weights = self.attention_layers[layer_idx].forward(x, x, x, mask)
            attention_weights.append(attn_weights)
            
            # Residual connection
            x = x + attn_output
            
            # Feed-forward network
            ff_output = self._feed_forward(x, self.feed_forward_layers[layer_idx])
            
            # Residual connection
            x = x + ff_output
        
        # Output layer
        output = np.dot(x, self.output_layer)
        
        return output, attention_weights
    
    def _feed_forward(self, x: np.ndarray, layer: dict) -> np.ndarray:
        """Simple feed-forward network."""
        hidden = np.dot(x, layer['W1'])
        hidden = np.maximum(hidden, 0)  # ReLU
        output = np.dot(hidden, layer['W2'])
        return output


def demonstrate_attention_mechanism():
    """Demonstrate basic attention mechanism."""
    
    print("=== Attention Mechanism Demonstration ===\n")
    
    # Create sample data
    seq_len = 5
    d_model = 8
    
    # Sample embeddings (in practice, these would come from word embeddings)
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = np.random.randn(seq_len, d_model)
    
    print("Input shapes:")
    print(f"Query: {query.shape}")
    print(f"Key: {key.shape}")
    print(f"Value: {value.shape}")
    
    # Create attention mechanism
    attention = SimpleAttention(d_model)
    
    # Compute attention
    output, attention_weights = attention.forward(query, key, value)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nAttention weights (showing which positions attend to which):")
    print(attention_weights.round(3))
    
    # Visualize attention weights
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights, annot=True, cmap='Blues', 
                xticklabels=[f'Pos_{i}' for i in range(seq_len)],
                yticklabels=[f'Pos_{i}' for i in range(seq_len)])
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.tight_layout()
    plt.show()
    
    return attention_weights


def demonstrate_multi_head_attention():
    """Demonstrate multi-head attention."""
    
    print("\n=== Multi-Head Attention Demonstration ===\n")
    
    # Create sample data
    seq_len = 6
    d_model = 12
    num_heads = 3
    
    # Sample embeddings
    query = np.random.randn(seq_len, d_model)
    key = np.random.randn(seq_len, d_model)
    value = np.random.randn(seq_len, d_model)
    
    print(f"Input shapes: {query.shape}")
    print(f"Number of heads: {num_heads}")
    print(f"Dimension per head: {d_model // num_heads}")
    
    # Create multi-head attention
    mha = MultiHeadAttention(d_model, num_heads)
    
    # Compute multi-head attention
    output, attention_weights = mha.forward(query, key, value)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\nMulti-head attention weights:")
    print(attention_weights.round(3))
    
    # Visualize multi-head attention
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall attention
    sns.heatmap(attention_weights, annot=True, cmap='Blues', ax=axes[0],
                xticklabels=[f'Pos_{i}' for i in range(seq_len)],
                yticklabels=[f'Pos_{i}' for i in range(seq_len)])
    axes[0].set_title('Multi-Head Attention Weights (Averaged)')
    axes[0].set_xlabel('Key Positions')
    axes[0].set_ylabel('Query Positions')
    
    # Show how different heads might focus on different patterns
    # (In a real implementation, we'd have access to individual head weights)
    sample_head_weights = np.random.randn(seq_len, seq_len)
    sample_head_weights = np.abs(sample_head_weights)  # Make positive
    sample_head_weights = sample_head_weights / sample_head_weights.sum(axis=1, keepdims=True)
    
    sns.heatmap(sample_head_weights, annot=True, cmap='Reds', ax=axes[1],
                xticklabels=[f'Pos_{i}' for i in range(seq_len)],
                yticklabels=[f'Pos_{i}' for i in range(seq_len)])
    axes[1].set_title('Sample Individual Head Attention')
    axes[1].set_xlabel('Key Positions')
    axes[1].set_ylabel('Query Positions')
    
    plt.tight_layout()
    plt.show()


def demonstrate_positional_encoding():
    """Demonstrate positional encoding."""
    
    print("\n=== Positional Encoding Demonstration ===\n")
    
    d_model = 16
    max_len = 20
    
    # Create positional encoding
    pos_encoding = PositionalEncoding(d_model, max_len)
    
    # Create sample embeddings
    seq_len = 10
    embeddings = np.random.randn(seq_len, d_model) * 0.1
    
    # Add positional encoding
    encoded = pos_encoding.forward(embeddings)
    
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Encoded embeddings shape: {encoded.shape}")
    
    # Visualize positional encoding
    plt.figure(figsize=(12, 8))
    
    # Show positional encoding patterns
    plt.subplot(2, 2, 1)
    plt.imshow(pos_encoding.pe[:max_len, :], cmap='viridis', aspect='auto')
    plt.title('Positional Encoding Matrix')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    # Show how encoding varies with position
    plt.subplot(2, 2, 2)
    positions = range(max_len)
    for dim in range(0, d_model, 4):  # Show every 4th dimension
        plt.plot(positions, pos_encoding.pe[:, dim], label=f'Dim {dim}')
    plt.title('Positional Encoding by Position')
    plt.xlabel('Position')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    # Show how encoding varies with dimension
    plt.subplot(2, 2, 3)
    dims = range(d_model)
    for pos in range(0, max_len, 4):  # Show every 4th position
        plt.plot(dims, pos_encoding.pe[pos, :], label=f'Pos {pos}')
    plt.title('Positional Encoding by Dimension')
    plt.xlabel('Dimension')
    plt.ylabel('Encoding Value')
    plt.legend()
    
    # Show the difference between original and encoded embeddings
    plt.subplot(2, 2, 4)
    diff = encoded - embeddings
    plt.imshow(diff, cmap='RdBu', aspect='auto')
    plt.title('Difference (Encoded - Original)')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Position')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def demonstrate_transformer():
    """Demonstrate simplified Transformer."""
    
    print("\n=== Simplified Transformer Demonstration ===\n")
    
    # Create a small vocabulary
    vocab = ['<PAD>', 'the', 'cat', 'sat', 'on', 'mat', 'dog', 'ran', 'fast']
    vocab_size = len(vocab)
    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    # Create Transformer
    d_model = 16
    num_heads = 4
    num_layers = 2
    
    transformer = SimplifiedTransformer(vocab_size, d_model, num_heads, num_layers)
    
    # Create sample input
    sentence = "the cat sat on mat"
    input_ids = [word_to_id.get(word, 0) for word in sentence.split()]
    
    print(f"Input sentence: '{sentence}'")
    print(f"Input IDs: {input_ids}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model dimensions: d_model={d_model}, heads={num_heads}, layers={num_layers}")
    
    # Forward pass
    output, attention_weights = transformer.forward(input_ids)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of attention layers: {len(attention_weights)}")
    
    # Show attention weights for each layer
    fig, axes = plt.subplots(1, num_layers, figsize=(5*num_layers, 5))
    
    for layer_idx in range(num_layers):
        weights = attention_weights[layer_idx]
        
        sns.heatmap(weights, annot=True, cmap='Blues', ax=axes[layer_idx],
                    xticklabels=[id_to_word.get(idx, 'UNK') for idx in input_ids],
                    yticklabels=[id_to_word.get(idx, 'UNK') for idx in input_ids])
        axes[layer_idx].set_title(f'Layer {layer_idx + 1} Attention')
        axes[layer_idx].set_xlabel('Key Words')
        axes[layer_idx].set_ylabel('Query Words')
    
    plt.tight_layout()
    plt.show()
    
    # Show output probabilities
    print("\nOutput logits (showing top 3 predictions for each position):")
    for pos, word in enumerate(sentence.split()):
        logits = output[pos]
        top_indices = np.argsort(logits)[-3:][::-1]
        print(f"Position {pos} ('{word}'):")
        for idx in top_indices:
            prob = np.exp(logits[idx]) / np.sum(np.exp(logits))
            print(f"  {id_to_word[idx]}: {prob:.3f}")


def demonstrate_attention_patterns():
    """Demonstrate different attention patterns."""
    
    print("\n=== Attention Patterns Demonstration ===\n")
    
    # Create different attention patterns
    seq_len = 8
    
    # 1. Causal attention (for autoregressive models)
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    causal_mask = causal_mask * -1e9  # Large negative value for masked positions
    
    # 2. Local attention (focus on nearby positions)
    local_attention = np.zeros((seq_len, seq_len))
    window_size = 3
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        local_attention[i, start:end] = 1
    
    # 3. Sparse attention (attend to specific positions)
    sparse_attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        # Attend to position i, i+1, and i+2 (modulo seq_len)
        sparse_attention[i, i] = 1
        sparse_attention[i, (i + 1) % seq_len] = 1
        sparse_attention[i, (i + 2) % seq_len] = 1
    
    # Visualize different patterns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    patterns = [
        (causal_mask, 'Causal Attention (Autoregressive)'),
        (local_attention, 'Local Attention (Window-based)'),
        (sparse_attention, 'Sparse Attention (Fixed Pattern)')
    ]
    
    for idx, (pattern, title) in enumerate(patterns):
        sns.heatmap(pattern, annot=True, cmap='Blues', ax=axes[idx],
                    xticklabels=[f'Pos_{i}' for i in range(seq_len)],
                    yticklabels=[f'Pos_{i}' for i in range(seq_len)])
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Key Positions')
        axes[idx].set_ylabel('Query Positions')
    
    plt.tight_layout()
    plt.show()
    
    print("Different attention patterns:")
    print("1. Causal: Each position can only attend to previous positions")
    print("2. Local: Each position attends to a fixed window around itself")
    print("3. Sparse: Each position attends to a specific set of positions")


def demonstrate_practical_applications():
    """Demonstrate practical applications of attention."""
    
    print("\n=== Practical Applications of Attention ===\n")
    
    # 1. Machine Translation Example
    print("1. Machine Translation:")
    source_sentence = "I love you"
    target_sentence = "Je t'aime"
    
    # Simulate attention weights for translation
    attention_matrix = np.array([
        [0.9, 0.1, 0.0],  # "Je" attends to "I"
        [0.1, 0.6, 0.3],  # "t'aime" attends to "love" and "you"
    ])
    
    print(f"Source: {source_sentence}")
    print(f"Target: {target_sentence}")
    print("Attention weights:")
    print(attention_matrix)
    
    # 2. Question Answering Example
    print("\n2. Question Answering:")
    context = "The cat sat on the mat. The dog ran fast."
    question = "What did the cat do?"
    
    # Simulate attention weights for QA
    qa_attention = np.array([0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    context_words = context.split()
    
    print(f"Context: {context}")
    print(f"Question: {question}")
    print("Attention to context words:")
    for word, weight in zip(context_words, qa_attention):
        print(f"  {word}: {weight:.2f}")
    
    # 3. Text Summarization Example
    print("\n3. Text Summarization:")
    document = "The weather is sunny today. It is very warm. People are happy."
    summary = "The weather is nice."
    
    # Simulate attention weights for summarization
    summary_attention = np.array([0.4, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    doc_words = document.split()
    
    print(f"Document: {document}")
    print(f"Summary: {summary}")
    print("Attention to document words:")
    for word, weight in zip(doc_words, summary_attention):
        print(f"  {word}: {weight:.2f}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_attention_mechanism()
    demonstrate_multi_head_attention()
    demonstrate_positional_encoding()
    demonstrate_transformer()
    demonstrate_attention_patterns()
    demonstrate_practical_applications() 