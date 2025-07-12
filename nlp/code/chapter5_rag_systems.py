"""
Chapter 5: Retrieval-Augmented Generation (RAG) Systems
=====================================================

This module demonstrates RAG system concepts including:
- Document processing and chunking
- Vector database operations
- Retrieval mechanisms
- RAG pipeline implementation
- Evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import json
import re
from collections import Counter
import math
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class Chunk:
    """Represents a text chunk."""
    id: str
    content: str
    document_id: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


class SimpleEmbeddingModel:
    """Simplified embedding model for demonstration."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
        # Simple random embeddings for demonstration
        # In practice, you'd use a real embedding model like SentenceTransformer
        self.vocab = {}
        self.embedding_matrix = None
    
    def fit(self, texts: List[str]):
        """Fit the embedding model to the texts."""
        # Create vocabulary
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        
        self.vocab = {word: idx for idx, word in enumerate(all_words)}
        
        # Create random embedding matrix
        vocab_size = len(self.vocab)
        self.embedding_matrix = np.random.randn(vocab_size, self.dimension) * 0.1
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        self.embedding_matrix = self.embedding_matrix / norms
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.embedding_matrix is None:
            raise ValueError("Model must be fitted before encoding")
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            word_embeddings = []
            
            for word in words:
                if word in self.vocab:
                    word_embeddings.append(self.embedding_matrix[self.vocab[word]])
            
            if word_embeddings:
                # Average word embeddings
                text_embedding = np.mean(word_embeddings, axis=0)
            else:
                # Zero vector if no words found
                text_embedding = np.zeros(self.dimension)
            
            # Normalize
            norm = np.linalg.norm(text_embedding)
            if norm > 0:
                text_embedding = text_embedding / norm
            
            embeddings.append(text_embedding)
        
        return np.array(embeddings)


class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, document: Document) -> List[Chunk]:
        """Create chunks from a document."""
        cleaned_text = self.clean_text(document.content)
        sentences = self.split_into_sentences(cleaned_text)
        
        chunks = []
        chunk_id = 0
        
        for i in range(0, len(sentences), max(1, len(sentences) // (len(cleaned_text) // self.chunk_size))):
            # Create chunk from sentences
            chunk_sentences = sentences[i:i + 3]  # Take 3 sentences per chunk
            chunk_content = ' '.join(chunk_sentences)
            
            if len(chunk_content) > 50:  # Minimum chunk size
                chunk = Chunk(
                    id=f"{document.id}_chunk_{chunk_id}",
                    content=chunk_content,
                    document_id=document.id,
                    start_idx=i,
                    end_idx=min(i + 3, len(sentences)),
                    metadata={
                        'document_title': document.metadata.get('title', ''),
                        'chunk_size': len(chunk_content),
                        'sentence_count': len(chunk_sentences)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks


class VectorDatabase:
    """Simple in-memory vector database."""
    
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []
        self.index = None
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the database."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        
        # Update index
        self._build_index()
    
    def _build_index(self):
        """Build similarity index."""
        if self.embeddings:
            self.index = np.array(self.embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks."""
        if self.index is None or len(self.index) == 0:
            return []
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], self.index)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only return positive similarities
                results.append((self.chunks[idx], similarities[idx]))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None


class RAGSystem:
    """Complete RAG system implementation."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_model = SimpleEmbeddingModel(embedding_dim)
        self.document_processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.documents: List[Document] = []
        
        # Simple LLM for generation (in practice, you'd use a real LLM)
        self.llm = self._create_simple_llm()
    
    def _create_simple_llm(self):
        """Create a simple LLM for demonstration."""
        # This is a very simplified LLM that just returns template responses
        # In practice, you'd use OpenAI GPT, Claude, or other LLMs
        
        class SimpleLLM:
            def generate(self, prompt: str) -> str:
                # Extract context and question from prompt
                if "Context:" in prompt and "Question:" in prompt:
                    context_start = prompt.find("Context:") + 9
                    context_end = prompt.find("Question:")
                    context = prompt[context_start:context_end].strip()
                    
                    question_start = prompt.find("Question:") + 10
                    question = prompt[question_start:].strip()
                    
                    # Generate a simple response based on context
                    if "machine learning" in context.lower():
                        return "Based on the context, machine learning is a subset of artificial intelligence that enables computers to learn from data."
                    elif "deep learning" in context.lower():
                        return "According to the context, deep learning uses neural networks with multiple layers to process complex data."
                    elif "natural language" in context.lower():
                        return "The context indicates that natural language processing helps computers understand and generate human language."
                    else:
                        return f"Based on the provided context, I can answer your question about '{question}'. The relevant information is: {context[:100]}..."
                else:
                    return "I need more context to provide an accurate answer."
        
        return SimpleLLM()
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        
        # Process documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.document_processor.create_chunks(doc)
            all_chunks.extend(chunks)
        
        # Create embeddings for chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        self.embedding_model.fit(chunk_texts)
        chunk_embeddings = self.embedding_model.encode(chunk_texts)
        
        # Add to vector database
        self.vector_db.add_chunks(all_chunks, chunk_embeddings)
        
        print(f"Added {len(documents)} documents, created {len(all_chunks)} chunks")
    
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Query the RAG system."""
        # 1. Encode the question
        question_embedding = self.embedding_model.encode([question])[0]
        
        # 2. Retrieve relevant chunks
        retrieved_chunks = self.vector_db.search(question_embedding, top_k)
        
        # 3. Create context from retrieved chunks
        context = "\n\n".join([chunk.content for chunk, score in retrieved_chunks])
        
        # 4. Generate response
        prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        response = self.llm.generate(prompt)
        
        # 5. Prepare result
        result = {
            'question': question,
            'response': response,
            'retrieved_chunks': [
                {
                    'id': chunk.id,
                    'content': chunk.content,
                    'similarity_score': score,
                    'document_id': chunk.document_id
                }
                for chunk, score in retrieved_chunks
            ],
            'context': context
        }
        
        return result
    
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        precision_scores = []
        recall_scores = []
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_chunk_ids = set(test_case['relevant_chunk_ids'])
            
            # Get retrieval results
            question_embedding = self.embedding_model.encode([query])[0]
            retrieved_chunks = self.vector_db.search(question_embedding, top_k=5)
            
            retrieved_chunk_ids = {chunk.id for chunk, _ in retrieved_chunks}
            
            # Calculate precision and recall
            if retrieved_chunk_ids:
                precision = len(relevant_chunk_ids & retrieved_chunk_ids) / len(retrieved_chunk_ids)
                precision_scores.append(precision)
            
            if relevant_chunk_ids:
                recall = len(relevant_chunk_ids & retrieved_chunk_ids) / len(relevant_chunk_ids)
                recall_scores.append(recall)
        
        # Calculate averages
        avg_precision = np.mean(precision_scores) if precision_scores else 0.0
        avg_recall = np.mean(recall_scores) if recall_scores else 0.0
        
        # Calculate F1 score
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score
        }


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    documents = [
        Document(
            id="doc_1",
            content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions. Machine learning has applications in various fields including healthcare, finance, and autonomous vehicles.",
            metadata={'title': 'Introduction to Machine Learning', 'author': 'AI Expert', 'date': '2023'}
        ),
        Document(
            id="doc_2",
            content="Deep learning is a type of machine learning that uses neural networks with multiple layers to process complex data. These neural networks are inspired by the human brain and can automatically learn hierarchical representations of data. Deep learning has achieved remarkable success in image recognition, natural language processing, and speech recognition.",
            metadata={'title': 'Deep Learning Fundamentals', 'author': 'ML Researcher', 'date': '2023'}
        ),
        Document(
            id="doc_3",
            content="Natural language processing (NLP) is a field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language. NLP combines computational linguistics with machine learning to process and analyze large amounts of natural language data. Applications include chatbots, translation services, and sentiment analysis.",
            metadata={'title': 'Natural Language Processing Guide', 'author': 'NLP Specialist', 'date': '2023'}
        ),
        Document(
            id="doc_4",
            content="Computer vision is a field of artificial intelligence that enables machines to interpret and understand visual information from the world. It involves developing algorithms and systems that can process, analyze, and make decisions based on images or video. Computer vision is used in facial recognition, autonomous driving, medical imaging, and quality control systems.",
            metadata={'title': 'Computer Vision Overview', 'author': 'CV Engineer', 'date': '2023'}
        ),
        Document(
            id="doc_5",
            content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward. The agent learns through trial and error, receiving feedback in the form of rewards or penalties. Reinforcement learning has been successfully applied to game playing, robotics, and recommendation systems.",
            metadata={'title': 'Reinforcement Learning Basics', 'author': 'RL Researcher', 'date': '2023'}
        )
    ]
    return documents


def demonstrate_rag_system():
    """Demonstrate the complete RAG system."""
    
    print("=== RAG System Demonstration ===\n")
    
    # Create RAG system
    rag = RAGSystem()
    
    # Add documents
    documents = create_sample_documents()
    rag.add_documents(documents)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are the applications of NLP?",
        "How is computer vision used?",
        "What is reinforcement learning?"
    ]
    
    print("Testing RAG system with various queries:\n")
    
    for query in test_queries:
        print(f"Query: {query}")
        result = rag.query(query)
        
        print(f"Response: {result['response']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks:")
        for chunk in result['retrieved_chunks']:
            print(f"  - {chunk['content'][:100]}... (score: {chunk['similarity_score']:.3f})")
        print("-" * 80)
    
    return rag


def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies."""
    
    print("\n=== Chunking Strategies Demonstration ===\n")
    
    # Sample document
    document = Document(
        id="sample_doc",
        content="Machine learning is a subset of artificial intelligence. It enables computers to learn from data. Deep learning uses neural networks. Natural language processing helps computers understand text. Computer vision processes images. Reinforcement learning uses trial and error.",
        metadata={'title': 'AI Overview'}
    )
    
    # Test different chunking strategies
    strategies = [
        ("Small chunks (100 chars)", DocumentProcessor(chunk_size=100, chunk_overlap=20)),
        ("Medium chunks (200 chars)", DocumentProcessor(chunk_size=200, chunk_overlap=50)),
        ("Large chunks (300 chars)", DocumentProcessor(chunk_size=300, chunk_overlap=100))
    ]
    
    for strategy_name, processor in strategies:
        print(f"{strategy_name}:")
        chunks = processor.create_chunks(document)
        
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {chunk.content[:80]}...")
        print()


def demonstrate_embedding_similarity():
    """Demonstrate embedding similarity search."""
    
    print("\n=== Embedding Similarity Demonstration ===\n")
    
    # Create embedding model
    embedder = SimpleEmbeddingModel(dimension=64)
    
    # Sample texts
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning uses trial and error",
        "The weather is sunny today",
        "Cooking requires following recipes carefully",
        "Music can evoke strong emotions in listeners"
    ]
    
    # Fit and encode
    embedder.fit(texts)
    embeddings = embedder.encode(texts)
    
    # Test similarity search
    query = "How do computers understand language?"
    query_embedding = embedder.encode([query])[0]
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Show results
    print(f"Query: {query}")
    print("\nSimilarity scores:")
    for i, (text, sim) in enumerate(zip(texts, similarities)):
        print(f"  {i+1}. {text[:50]}... (similarity: {sim:.3f})")
    
    # Visualize similarities
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(texts)), similarities)
    plt.title('Similarity Scores for Query')
    plt.xlabel('Document Index')
    plt.ylabel('Similarity Score')
    plt.xticks(range(len(texts)), [f'Doc {i+1}' for i in range(len(texts))], rotation=45)
    plt.tight_layout()
    plt.show()


def demonstrate_rag_evaluation():
    """Demonstrate RAG system evaluation."""
    
    print("\n=== RAG System Evaluation ===\n")
    
    # Create RAG system
    rag = RAGSystem()
    documents = create_sample_documents()
    rag.add_documents(documents)
    
    # Create test queries with known relevant chunks
    test_queries = [
        {
            'query': 'What is machine learning?',
            'relevant_chunk_ids': ['doc_1_chunk_0', 'doc_1_chunk_1']
        },
        {
            'query': 'How does deep learning work?',
            'relevant_chunk_ids': ['doc_2_chunk_0', 'doc_2_chunk_1']
        },
        {
            'query': 'What is NLP used for?',
            'relevant_chunk_ids': ['doc_3_chunk_0', 'doc_3_chunk_1']
        }
    ]
    
    # Evaluate
    metrics = rag.evaluate_retrieval(test_queries)
    
    print("Evaluation Results:")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    # Visualize metrics
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    metrics_names = ['Precision', 'Recall', 'F1 Score']
    metrics_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    bars = ax.bar(metrics_names, metrics_values, color=['blue', 'green', 'red'])
    ax.set_ylabel('Score')
    ax.set_title('RAG System Evaluation Metrics')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def demonstrate_prompt_engineering_for_rag():
    """Demonstrate prompt engineering for RAG systems."""
    
    print("\n=== Prompt Engineering for RAG ===\n")
    
    # Sample context and question
    context = """Machine learning is a subset of artificial intelligence that enables computers to learn from data. Deep learning uses neural networks with multiple layers. Natural language processing helps computers understand text."""
    
    question = "What is the relationship between machine learning and deep learning?"
    
    # Different prompt templates
    prompt_templates = [
        # Basic template
        f"Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        
        # Enhanced template
        f"""You are a helpful AI assistant. Use the following context to answer the question.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer is not in the context, say so
- Provide specific details from the context

Answer:""",
        
        # Structured template
        f"""Task: Answer the following question using the provided context.

Context: {context}

Question: {question}

Requirements:
1. Base your answer entirely on the context
2. Be specific and detailed
3. If information is missing, acknowledge it
4. Use a clear, professional tone

Answer:"""
    ]
    
    template_names = ["Basic", "Enhanced", "Structured"]
    
    print("Comparing different prompt templates:\n")
    
    for name, template in zip(template_names, prompt_templates):
        print(f"{name} Template:")
        print(f"Prompt: {template[:100]}...")
        
        # In a real system, you'd send this to an actual LLM
        # For demonstration, we'll show what the prompt looks like
        print(f"Length: {len(template)} characters")
        print(f"Contains instructions: {'Instructions:' in template}")
        print(f"Contains requirements: {'Requirements:' in template}")
        print("-" * 50)


if __name__ == "__main__":
    # Run all demonstrations
    rag_system = demonstrate_rag_system()
    demonstrate_chunking_strategies()
    demonstrate_embedding_similarity()
    demonstrate_rag_evaluation()
    demonstrate_prompt_engineering_for_rag() 