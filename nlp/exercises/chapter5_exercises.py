"""
Chapter 5 Exercises: Retrieval-Augmented Generation (RAG) Systems
==============================================================

Practice exercises for RAG system concepts including:
- Document processing and chunking
- Vector database operations
- Retrieval mechanisms
- RAG pipeline implementation
- Evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import Counter
import math
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# Import the RAG code from the examples
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from chapter5_rag_systems import Document, Chunk, SimpleEmbeddingModel, DocumentProcessor, VectorDatabase


class Chapter5Exercises:
    """Exercise problems for Chapter 5."""
    
    def __init__(self):
        pass
    
    def exercise_5_1_document_chunking(self, document: Document, chunk_size: int = 150, 
                                     overlap: int = 30) -> List[Chunk]:
        """
        Exercise 5.1: Implement document chunking with overlap.
        
        Requirements:
        - Split document into chunks of specified size
        - Implement overlap between chunks
        - Handle edge cases (documents smaller than chunk size)
        - Preserve document metadata in chunks
        
        Args:
            document: Document to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunks
        """
        # TODO: Implement your document chunking here
        # Replace this with your implementation
        pass
    
    def exercise_5_2_semantic_chunking(self, document: Document) -> List[Chunk]:
        """
        Exercise 5.2: Implement semantic chunking based on sentence boundaries.
        
        Requirements:
        - Split document at sentence boundaries
        - Group related sentences together
        - Avoid breaking semantic units
        - Handle different sentence types (declarative, interrogative, etc.)
        
        Args:
            document: Document to chunk semantically
            
        Returns:
            List of semantic chunks
        """
        # TODO: Implement your semantic chunking here
        # Replace this with your implementation
        pass
    
    def exercise_5_3_embedding_similarity(self, query_embedding: np.ndarray, 
                                        document_embeddings: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Exercise 5.3: Implement similarity search with different metrics.
        
        Requirements:
        - Implement cosine similarity
        - Implement Euclidean distance
        - Implement dot product similarity
        - Return ranked results with scores
        - Handle edge cases (zero vectors, normalization)
        
        Args:
            query_embedding: Query vector
            document_embeddings: List of document vectors
            
        Returns:
            List of (document_index, similarity_score) tuples, ranked by similarity
        """
        # TODO: Implement your similarity search here
        # Replace this with your implementation
        pass
    
    def exercise_5_4_hybrid_search(self, query: str, documents: List[str], 
                                 top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Exercise 5.4: Implement hybrid search combining dense and sparse retrieval.
        
        Requirements:
        - Implement TF-IDF sparse retrieval
        - Implement dense embedding retrieval
        - Combine scores from both methods
        - Weight the combination appropriately
        - Return ranked results with method indicators
        
        Args:
            query: Search query
            documents: List of documents
            top_k: Number of results to return
            
        Returns:
            List of (doc_index, combined_score, method) tuples
        """
        # TODO: Implement your hybrid search here
        # Replace this with your implementation
        pass
    
    def exercise_5_5_rag_prompt_engineering(self, context: str, question: str) -> str:
        """
        Exercise 5.5: Design effective prompts for RAG systems.
        
        Requirements:
        - Create a prompt that encourages factuality
        - Include instructions for handling missing information
        - Add attribution requirements
        - Handle uncertainty appropriately
        - Ensure clear output format
        
        Args:
            context: Retrieved context
            question: User question
            
        Returns:
            Well-designed prompt for RAG system
        """
        # TODO: Implement your RAG prompt here
        # Replace this with your implementation
        pass
    
    def exercise_5_6_retrieval_evaluation(self, query: str, retrieved_docs: List[str], 
                                        relevant_docs: List[str]) -> Dict[str, float]:
        """
        Exercise 5.6: Implement retrieval evaluation metrics.
        
        Requirements:
        - Calculate precision at k (P@k)
        - Calculate recall at k (R@k)
        - Calculate F1 score
        - Calculate Mean Reciprocal Rank (MRR)
        - Calculate Normalized Discounted Cumulative Gain (NDCG)
        
        Args:
            query: Search query
            retrieved_docs: List of retrieved document IDs
            relevant_docs: List of relevant document IDs
            
        Returns:
            Dictionary with evaluation metrics
        """
        # TODO: Implement your retrieval evaluation here
        # Replace this with your implementation
        pass
    
    def exercise_5_7_context_window_optimization(self, chunks: List[Chunk], 
                                               max_tokens: int = 1000) -> List[Chunk]:
        """
        Exercise 5.7: Optimize context window usage for RAG.
        
        Requirements:
        - Select chunks to fit within token limit
        - Prioritize most relevant chunks
        - Ensure diversity in selected chunks
        - Handle overlapping information
        - Maintain semantic coherence
        
        Args:
            chunks: List of candidate chunks
            max_tokens: Maximum tokens allowed in context
            
        Returns:
            Optimized list of chunks for context window
        """
        # TODO: Implement your context window optimization here
        # Replace this with your implementation
        pass
    
    def exercise_5_8_rag_system_analysis(self, rag_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Exercise 5.8: Analyze RAG system performance and behavior.
        
        Requirements:
        - Analyze retrieval quality patterns
        - Identify common failure modes
        - Measure response consistency
        - Analyze source attribution quality
        - Generate improvement recommendations
        
        Args:
            rag_results: List of RAG query results
            
        Returns:
            Analysis report with insights and recommendations
        """
        # TODO: Implement your RAG system analysis here
        # Replace this with your implementation
        pass


def run_exercises():
    """Run all exercises with sample data."""
    
    exercises = Chapter5Exercises()
    
    print("=== Chapter 5 Exercises ===\n")
    
    # Exercise 5.1: Document Chunking
    print("Exercise 5.1 - Document Chunking:")
    document = Document(
        id="test_doc",
        content="Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand text. Computer vision processes images and videos.",
        metadata={'title': 'AI Overview', 'author': 'Test Author'}
    )
    
    try:
        chunks = exercises.exercise_5_1_document_chunking(document, chunk_size=100, overlap=20)
        print(f"✓ Created {len(chunks)} chunks")
        print(f"✓ Average chunk size: {np.mean([len(c.content) for c in chunks]):.1f} characters")
        print(f"✓ Chunk overlap implemented: {chunks[0].metadata.get('overlap', False)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.2: Semantic Chunking
    print("Exercise 5.2 - Semantic Chunking:")
    
    try:
        semantic_chunks = exercises.exercise_5_2_semantic_chunking(document)
        print(f"✓ Created {len(semantic_chunks)} semantic chunks")
        print(f"✓ Average sentences per chunk: {np.mean([c.metadata.get('sentence_count', 1) for c in semantic_chunks]):.1f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.3: Embedding Similarity
    print("Exercise 5.3 - Embedding Similarity:")
    query_embedding = np.random.randn(128)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    document_embeddings = [np.random.randn(128) for _ in range(5)]
    for i, emb in enumerate(document_embeddings):
        document_embeddings[i] = emb / np.linalg.norm(emb)
    
    try:
        similarities = exercises.exercise_5_3_embedding_similarity(query_embedding, document_embeddings)
        print(f"✓ Found {len(similarities)} similar documents")
        print(f"✓ Top similarity score: {similarities[0][1]:.3f}")
        print(f"✓ Results properly ranked: {similarities[0][1] >= similarities[-1][1]}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.4: Hybrid Search
    print("Exercise 5.4 - Hybrid Search:")
    query = "machine learning applications"
    documents = [
        "Machine learning is used in healthcare for diagnosis",
        "Deep learning applications in computer vision",
        "Natural language processing for chatbots",
        "Reinforcement learning in robotics",
        "Data science and machine learning techniques"
    ]
    
    try:
        hybrid_results = exercises.exercise_5_4_hybrid_search(query, documents, top_k=3)
        print(f"✓ Retrieved {len(hybrid_results)} documents")
        print(f"✓ Methods used: {set(result[2] for result in hybrid_results)}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.5: RAG Prompt Engineering
    print("Exercise 5.5 - RAG Prompt Engineering:")
    context = "Machine learning enables computers to learn from data. Deep learning uses neural networks."
    question = "What is the difference between machine learning and deep learning?"
    
    try:
        prompt = exercises.exercise_5_5_rag_prompt_engineering(context, question)
        print(f"✓ Prompt length: {len(prompt)} characters")
        print(f"✓ Contains attribution: {'attribution' in prompt.lower()}")
        print(f"✓ Contains uncertainty handling: {'uncertain' in prompt.lower() or 'missing' in prompt.lower()}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.6: Retrieval Evaluation
    print("Exercise 5.6 - Retrieval Evaluation:")
    query = "machine learning"
    retrieved_docs = ["doc_1", "doc_3", "doc_5", "doc_7", "doc_9"]
    relevant_docs = ["doc_1", "doc_3", "doc_8", "doc_10"]
    
    try:
        metrics = exercises.exercise_5_6_retrieval_evaluation(query, retrieved_docs, relevant_docs)
        print(f"✓ Calculated {len(metrics)} metrics")
        print(f"✓ Precision: {metrics.get('precision', 0):.3f}")
        print(f"✓ Recall: {metrics.get('recall', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.7: Context Window Optimization
    print("Exercise 5.7 - Context Window Optimization:")
    chunks = [
        Chunk("chunk_1", "Machine learning basics", "doc_1", 0, 10, {'relevance': 0.9, 'tokens': 200}),
        Chunk("chunk_2", "Deep learning overview", "doc_2", 0, 10, {'relevance': 0.8, 'tokens': 300}),
        Chunk("chunk_3", "NLP applications", "doc_3", 0, 10, {'relevance': 0.7, 'tokens': 250}),
        Chunk("chunk_4", "Computer vision", "doc_4", 0, 10, {'relevance': 0.6, 'tokens': 180}),
        Chunk("chunk_5", "Reinforcement learning", "doc_5", 0, 10, {'relevance': 0.5, 'tokens': 220})
    ]
    
    try:
        optimized_chunks = exercises.exercise_5_7_context_window_optimization(chunks, max_tokens=800)
        total_tokens = sum(c.metadata.get('tokens', 0) for c in optimized_chunks)
        print(f"✓ Selected {len(optimized_chunks)} chunks")
        print(f"✓ Total tokens: {total_tokens} (limit: 800)")
        print(f"✓ Within limit: {total_tokens <= 800}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 5.8: RAG System Analysis
    print("Exercise 5.8 - RAG System Analysis:")
    rag_results = [
        {
            'query': 'What is machine learning?',
            'response': 'Machine learning is a subset of AI.',
            'retrieved_chunks': [{'content': 'ML is AI subset', 'score': 0.9}],
            'sources': ['doc_1']
        },
        {
            'query': 'How does deep learning work?',
            'response': 'Deep learning uses neural networks.',
            'retrieved_chunks': [{'content': 'DL uses neural nets', 'score': 0.8}],
            'sources': ['doc_2']
        }
    ]
    
    try:
        analysis = exercises.exercise_5_8_rag_system_analysis(rag_results)
        print(f"✓ Analysis completed: {list(analysis.keys())}")
        print(f"✓ Insights generated: {len(analysis.get('insights', []))}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")


def provide_solutions():
    """Provide solution hints and examples."""
    
    print("=== Exercise Solutions and Hints ===\n")
    
    print("Exercise 5.1 - Document Chunking Hints:")
    print("- Use sliding window approach with overlap")
    print("- Handle documents smaller than chunk size")
    print("- Preserve document metadata in each chunk")
    print("- Consider sentence boundaries when possible")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.2 - Semantic Chunking Hints:")
    print("- Split at sentence boundaries using regex")
    print("- Group related sentences based on topics")
    print("- Avoid breaking semantic units")
    print("- Consider paragraph boundaries")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.3 - Embedding Similarity Hints:")
    print("- Cosine similarity: dot product of normalized vectors")
    print("- Euclidean distance: L2 norm of vector difference")
    print("- Dot product: direct multiplication of vectors")
    print("- Normalize vectors to handle scale differences")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.4 - Hybrid Search Hints:")
    print("- TF-IDF for sparse retrieval")
    print("- Embeddings for dense retrieval")
    print("- Combine scores with weighted average")
    print("- Normalize scores before combining")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.5 - RAG Prompt Engineering Hints:")
    print("- Include clear instructions for factuality")
    print("- Add attribution requirements")
    print("- Handle missing information gracefully")
    print("- Specify output format and tone")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.6 - Retrieval Evaluation Hints:")
    print("- Precision = relevant retrieved / total retrieved")
    print("- Recall = relevant retrieved / total relevant")
    print("- F1 = 2 * (precision * recall) / (precision + recall)")
    print("- MRR = 1 / rank of first relevant document")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.7 - Context Window Optimization Hints:")
    print("- Sort chunks by relevance score")
    print("- Add chunks until token limit is reached")
    print("- Ensure diversity by avoiding similar chunks")
    print("- Consider semantic coherence")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 5.8 - RAG System Analysis Hints:")
    print("- Analyze retrieval quality patterns")
    print("- Identify common failure modes")
    print("- Measure response consistency")
    print("- Generate actionable recommendations")
    print("\n" + "-"*40 + "\n")


def demonstrate_rag_concepts():
    """Demonstrate key RAG concepts for reference."""
    
    print("=== RAG Concepts Demonstration ===\n")
    
    # Create sample documents
    documents = [
        Document("doc_1", "Machine learning enables computers to learn from data.", {'title': 'ML Intro'}),
        Document("doc_2", "Deep learning uses neural networks with multiple layers.", {'title': 'DL Basics'}),
        Document("doc_3", "Natural language processing helps computers understand text.", {'title': 'NLP Guide'})
    ]
    
    print("Sample Documents:")
    for doc in documents:
        print(f"- {doc.metadata['title']}: {doc.content}")
    
    # Demonstrate chunking
    processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    print(f"\nChunking Results:")
    for doc in documents:
        chunks = processor.create_chunks(doc)
        print(f"- {doc.metadata['title']}: {len(chunks)} chunks")
        for chunk in chunks:
            print(f"  Chunk: {chunk.content[:30]}...")
    
    # Demonstrate embedding
    embedder = SimpleEmbeddingModel(dimension=64)
    texts = [doc.content for doc in documents]
    embedder.fit(texts)
    embeddings = embedder.encode(texts)
    
    print(f"\nEmbedding Results:")
    print(f"- Embedding dimension: {embeddings.shape[1]}")
    print(f"- Number of documents: {embeddings.shape[0]}")
    
    # Demonstrate similarity
    query = "How do computers learn?"
    query_embedding = embedder.encode([query])[0]
    
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    print(f"\nSimilarity Search Results:")
    print(f"Query: {query}")
    for i, (doc, sim) in enumerate(zip(documents, similarities)):
        print(f"- {doc.metadata['title']}: {sim:.3f}")


if __name__ == "__main__":
    # Run exercises
    run_exercises()
    
    # Provide solution hints
    provide_solutions()
    
    # Demonstrate concepts
    demonstrate_rag_concepts() 