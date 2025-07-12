"""
Chapter 2: Word Embeddings
==========================

This module demonstrates word embedding concepts including:
- One-hot encoding limitations
- Word2Vec implementation
- GloVe embeddings
- Semantic similarity and analogies
- Word embedding applications
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# For demonstration purposes, we'll create simple embeddings
# In practice, you would use pre-trained models like gensim's Word2Vec or GloVe


class OneHotEncoder:
    """Demonstrate the limitations of one-hot encoding."""
    
    def __init__(self, vocabulary: List[str]):
        self.vocabulary = vocabulary
        self.word_to_index = {word: idx for idx, word in enumerate(vocabulary)}
        self.index_to_word = {idx: word for idx, word in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)
    
    def encode(self, word: str) -> np.ndarray:
        """Convert a word to one-hot encoding."""
        if word not in self.word_to_index:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        vector = np.zeros(self.vocab_size)
        vector[self.word_to_index[word]] = 1
        return vector
    
    def decode(self, vector: np.ndarray) -> str:
        """Convert one-hot vector back to word."""
        idx = np.argmax(vector)
        return self.index_to_word[idx]
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two one-hot vectors."""
        vec1 = self.encode(word1)
        vec2 = self.encode(word2)
        return cosine_similarity([vec1], [vec2])[0][0]


class SimpleWordEmbeddings:
    """Simple word embeddings for demonstration."""
    
    def __init__(self, embedding_dim: int = 3):
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        
    def add_word(self, word: str, vector: np.ndarray):
        """Add a word and its embedding vector."""
        if len(vector) != self.embedding_dim:
            raise ValueError(f"Vector dimension {len(vector)} != {self.embedding_dim}")
        self.word_vectors[word] = vector
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word."""
        return self.word_vectors.get(word)
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two word vectors."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def find_most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words to the given word."""
        if word not in self.word_vectors:
            return []
        
        similarities = []
        target_vec = self.word_vectors[word]
        
        for other_word, other_vec in self.word_vectors.items():
            if other_word != word:
                sim = cosine_similarity([target_vec], [other_vec])[0][0]
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def solve_analogy(self, word_a: str, word_b: str, word_c: str) -> Optional[str]:
        """Solve analogy: word_a is to word_b as word_c is to ?"""
        if not all(word in self.word_vectors for word in [word_a, word_b, word_c]):
            return None
        
        # Calculate analogy vector: word_b - word_a + word_c
        analogy_vec = (self.word_vectors[word_b] - 
                      self.word_vectors[word_a] + 
                      self.word_vectors[word_c])
        
        # Find the word with the most similar vector
        best_word = None
        best_similarity = -1
        
        for word, vec in self.word_vectors.items():
            if word not in [word_a, word_b, word_c]:
                sim = cosine_similarity([analogy_vec], [vec])[0][0]
                if sim > best_similarity:
                    best_similarity = sim
                    best_word = word
        
        return best_word


class SimpleWord2Vec:
    """Simplified Word2Vec implementation for demonstration."""
    
    def __init__(self, embedding_dim: int = 100, window_size: int = 2):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.word_vectors = {}
        self.vocabulary = set()
        
    def preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing."""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def build_vocabulary(self, texts: List[str], min_freq: int = 2):
        """Build vocabulary from texts."""
        word_counts = defaultdict(int)
        
        for text in texts:
            words = self.preprocess_text(text)
            for word in words:
                word_counts[word] += 1
        
        # Keep words that appear at least min_freq times
        self.vocabulary = {word for word, count in word_counts.items() 
                          if count >= min_freq}
        
        # Initialize random vectors for each word
        for word in self.vocabulary:
            self.word_vectors[word] = np.random.randn(self.embedding_dim)
    
    def get_context_pairs(self, text: str) -> List[Tuple[str, str]]:
        """Generate context pairs for training."""
        words = self.preprocess_text(text)
        pairs = []
        
        for i, target_word in enumerate(words):
            if target_word not in self.vocabulary:
                continue
            
            # Get context words within window
            start = max(0, i - self.window_size)
            end = min(len(words), i + self.window_size + 1)
            
            for j in range(start, end):
                if i != j and words[j] in self.vocabulary:
                    pairs.append((target_word, words[j]))
        
        return pairs
    
    def train_step(self, target_word: str, context_word: str, learning_rate: float = 0.01):
        """Single training step for a word pair."""
        if target_word not in self.word_vectors or context_word not in self.word_vectors:
            return
        
        target_vec = self.word_vectors[target_word]
        context_vec = self.word_vectors[context_word]
        
        # Simple update rule (simplified from actual Word2Vec)
        # In practice, this would involve negative sampling and more complex updates
        similarity = np.dot(target_vec, context_vec)
        
        # Simple gradient update
        gradient = 0.1 * (1 - similarity) * context_vec
        self.word_vectors[target_word] += learning_rate * gradient
    
    def train(self, texts: List[str], epochs: int = 10, learning_rate: float = 0.01):
        """Train the word embeddings."""
        print(f"Training Word2Vec with {len(texts)} texts, {epochs} epochs...")
        
        for epoch in range(epochs):
            total_pairs = 0
            for text in texts:
                pairs = self.get_context_pairs(text)
                total_pairs += len(pairs)
                
                for target_word, context_word in pairs:
                    self.train_step(target_word, context_word, learning_rate)
            
            print(f"Epoch {epoch + 1}/{epochs}: Processed {total_pairs} word pairs")


def demonstrate_one_hot_limitations():
    """Demonstrate the limitations of one-hot encoding."""
    
    print("=== One-Hot Encoding Limitations ===\n")
    
    # Sample vocabulary
    vocabulary = ["cat", "dog", "bird", "computer", "laptop", "phone"]
    encoder = OneHotEncoder(vocabulary)
    
    print("One-Hot Encodings:")
    for word in vocabulary:
        vector = encoder.encode(word)
        print(f"{word}: {vector}")
    
    print("\nSimilarity Analysis:")
    print("(Note: All similarities are 0 because one-hot vectors are orthogonal)")
    
    pairs = [("cat", "dog"), ("cat", "computer"), ("laptop", "phone")]
    for word1, word2 in pairs:
        similarity = encoder.get_similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity}")
    
    print("\nProblems with One-Hot Encoding:")
    print("1. High dimensionality: Each word is a {}-dimensional vector".format(len(vocabulary)))
    print("2. No semantic information: All similarities are 0")
    print("3. No analogies possible: Can't do 'cat - dog + bird'")
    print("4. Poor generalization: Can't handle unseen words")


def demonstrate_word_embeddings():
    """Demonstrate word embeddings with semantic relationships."""
    
    print("\n=== Word Embeddings Demonstration ===\n")
    
    # Create simple embeddings with semantic relationships
    embeddings = SimpleWordEmbeddings(embedding_dim=3)
    
    # Add words with meaningful vectors
    # Animals cluster together
    embeddings.add_word("cat", np.array([0.8, 0.1, 0.1]))
    embeddings.add_word("dog", np.array([0.7, 0.2, 0.1]))
    embeddings.add_word("bird", np.array([0.6, 0.3, 0.1]))
    
    # Technology cluster together
    embeddings.add_word("computer", np.array([0.1, 0.8, 0.1]))
    embeddings.add_word("laptop", np.array([0.2, 0.7, 0.1]))
    embeddings.add_word("phone", np.array([0.1, 0.9, 0.0]))
    
    # Gender relationships
    embeddings.add_word("king", np.array([0.5, 0.1, 0.9]))
    embeddings.add_word("queen", np.array([0.5, 0.1, 0.8]))
    embeddings.add_word("man", np.array([0.4, 0.1, 0.9]))
    embeddings.add_word("woman", np.array([0.4, 0.1, 0.8]))
    
    print("Word Vectors:")
    for word, vector in embeddings.word_vectors.items():
        print(f"{word}: {vector}")
    
    print("\nSemantic Similarities:")
    pairs = [("cat", "dog"), ("cat", "computer"), ("laptop", "phone"), ("king", "queen")]
    for word1, word2 in pairs:
        similarity = embeddings.get_similarity(word1, word2)
        print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")
    
    print("\nMost Similar Words:")
    for word in ["cat", "computer", "king"]:
        similar = embeddings.find_most_similar(word, top_k=3)
        print(f"Words similar to '{word}': {similar}")
    
    print("\nWord Analogies:")
    analogies = [
        ("king", "man", "woman"),
        ("cat", "dog", "bird")
    ]
    
    for word_a, word_b, word_c in analogies:
        result = embeddings.solve_analogy(word_a, word_b, word_c)
        print(f"'{word_a}' is to '{word_b}' as '{word_c}' is to '{result}'")


def demonstrate_word2vec():
    """Demonstrate a simple Word2Vec implementation."""
    
    print("\n=== Simple Word2Vec Implementation ===\n")
    
    # Sample texts for training
    texts = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew in the sky",
        "the computer is on the desk",
        "the laptop is portable",
        "the phone is mobile",
        "cats and dogs are pets",
        "computers and laptops are devices",
        "the king ruled the kingdom",
        "the queen wore a crown",
        "men and women are people",
        "the cat chased the dog",
        "the computer processes data",
        "the phone makes calls"
    ]
    
    # Initialize and train Word2Vec
    word2vec = SimpleWord2Vec(embedding_dim=10, window_size=2)
    word2vec.build_vocabulary(texts, min_freq=1)
    word2vec.train(texts, epochs=5, learning_rate=0.01)
    
    print("Trained Word Vectors:")
    for word, vector in word2vec.word_vectors.items():
        print(f"{word}: {vector[:5]}...")  # Show first 5 dimensions
    
    print("\nSimilarities in Trained Embeddings:")
    pairs = [("cat", "dog"), ("computer", "laptop"), ("king", "queen")]
    for word1, word2 in pairs:
        if word1 in word2vec.word_vectors and word2 in word2vec.word_vectors:
            similarity = word2vec.get_similarity(word1, word2)
            print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")


def visualize_embeddings():
    """Visualize word embeddings using t-SNE."""
    
    print("\n=== Word Embeddings Visualization ===\n")
    
    # Create embeddings for visualization
    embeddings = SimpleWordEmbeddings(embedding_dim=10)
    
    # Add words with some structure
    words = ["cat", "dog", "bird", "computer", "laptop", "phone", 
             "king", "queen", "man", "woman", "happy", "sad", "good", "bad"]
    
    # Generate vectors with some clustering
    np.random.seed(42)  # For reproducible results
    
    for word in words:
        if word in ["cat", "dog", "bird"]:
            # Animal cluster
            vector = np.random.normal([0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.1)
        elif word in ["computer", "laptop", "phone"]:
            # Technology cluster
            vector = np.random.normal([0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.1)
        elif word in ["king", "queen", "man", "woman"]:
            # People cluster
            vector = np.random.normal([0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.1)
        else:
            # Emotion cluster
            vector = np.random.normal([0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.1)
        
        embeddings.add_word(word, vector)
    
    # Prepare data for t-SNE
    word_list = list(embeddings.word_vectors.keys())
    vectors = np.array([embeddings.word_vectors[word] for word in word_list])
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(word_list)-1))
    vectors_2d = tsne.fit_transform(vectors)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Color code by category
    colors = []
    for word in word_list:
        if word in ["cat", "dog", "bird"]:
            colors.append('red')  # Animals
        elif word in ["computer", "laptop", "phone"]:
            colors.append('blue')  # Technology
        elif word in ["king", "queen", "man", "woman"]:
            colors.append('green')  # People
        else:
            colors.append('orange')  # Emotions
    
    # Plot points
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=colors, s=100, alpha=0.7)
    
    # Add labels
    for i, word in enumerate(word_list):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.title('Word Embeddings Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(['Animals', 'Technology', 'People', 'Emotions'], 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('word_embeddings_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'word_embeddings_visualization.png'")
    plt.show()


def demonstrate_applications():
    """Demonstrate practical applications of word embeddings."""
    
    print("\n=== Word Embeddings Applications ===\n")
    
    # Create embeddings for demonstration
    embeddings = SimpleWordEmbeddings(embedding_dim=5)
    
    # Add words with semantic structure
    embeddings.add_word("happy", np.array([0.9, 0.1, 0.1, 0.1, 0.1]))
    embeddings.add_word("joyful", np.array([0.8, 0.2, 0.1, 0.1, 0.1]))
    embeddings.add_word("sad", np.array([0.1, 0.9, 0.1, 0.1, 0.1]))
    embeddings.add_word("depressed", np.array([0.2, 0.8, 0.1, 0.1, 0.1]))
    embeddings.add_word("good", np.array([0.1, 0.1, 0.9, 0.1, 0.1]))
    embeddings.add_word("excellent", np.array([0.1, 0.1, 0.8, 0.2, 0.1]))
    embeddings.add_word("bad", np.array([0.1, 0.1, 0.1, 0.9, 0.1]))
    embeddings.add_word("terrible", np.array([0.1, 0.1, 0.2, 0.8, 0.1]))
    
    print("1. Semantic Search:")
    query = "happy"
    similar_words = embeddings.find_most_similar(query, top_k=3)
    print(f"Words similar to '{query}': {similar_words}")
    
    print("\n2. Sentiment Analysis:")
    positive_words = ["happy", "joyful", "good", "excellent"]
    negative_words = ["sad", "depressed", "bad", "terrible"]
    
    # Calculate sentiment score for a sentence
    sentence_words = ["I", "feel", "happy", "today"]
    sentiment_score = 0
    for word in sentence_words:
        if word in embeddings.word_vectors:
            # Find similarity to positive and negative words
            pos_sim = max(embeddings.get_similarity(word, pos_word) for pos_word in positive_words)
            neg_sim = max(embeddings.get_similarity(word, neg_word) for neg_word in negative_words)
            sentiment_score += pos_sim - neg_sim
    
    print(f"Sentiment score for 'I feel happy today': {sentiment_score:.3f}")
    
    print("\n3. Document Similarity:")
    doc1_words = ["happy", "good", "excellent"]
    doc2_words = ["sad", "bad", "terrible"]
    
    # Calculate document vectors (average of word vectors)
    def get_doc_vector(words):
        vectors = [embeddings.word_vectors[word] for word in words if word in embeddings.word_vectors]
        return np.mean(vectors, axis=0) if vectors else np.zeros(embeddings.embedding_dim)
    
    doc1_vec = get_doc_vector(doc1_words)
    doc2_vec = get_doc_vector(doc2_words)
    
    doc_similarity = cosine_similarity([doc1_vec], [doc2_vec])[0][0]
    print(f"Similarity between positive and negative documents: {doc_similarity:.3f}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_one_hot_limitations()
    demonstrate_word_embeddings()
    demonstrate_word2vec()
    demonstrate_applications()
    
    # Uncomment to see visualization (requires matplotlib)
    # visualize_embeddings() 