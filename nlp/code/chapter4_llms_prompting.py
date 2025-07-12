"""
Chapter 4: Large Language Models and Prompt Engineering
=====================================================

This module demonstrates LLM concepts including:
- Sampling strategies (temperature, top-k, top-p)
- Prompt engineering techniques
- Few-shot learning examples
- Practical LLM applications
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
import random
import json
import re
from collections import Counter
import math


class SimpleLLM:
    """Simplified LLM for demonstration purposes."""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Simple vocabulary (in practice, this would be much larger)
        self.vocab = self._create_vocabulary()
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        # Simple transition matrix (simulating learned probabilities)
        self.transition_matrix = self._create_transition_matrix()
        
        # Context window
        self.context_window = 10
    
    def _create_vocabulary(self) -> List[str]:
        """Create a simple vocabulary for demonstration."""
        words = [
            # Common words
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            
            # Nouns
            'cat', 'dog', 'house', 'car', 'book', 'computer', 'phone', 'tree', 'flower', 'bird',
            'man', 'woman', 'child', 'teacher', 'student', 'doctor', 'engineer', 'artist',
            'weather', 'sun', 'rain', 'snow', 'wind', 'cloud', 'sky', 'earth', 'water', 'fire',
            
            # Verbs
            'run', 'walk', 'jump', 'eat', 'drink', 'sleep', 'work', 'play', 'study', 'read', 'write',
            'speak', 'listen', 'watch', 'see', 'hear', 'feel', 'think', 'know', 'understand',
            'love', 'like', 'hate', 'want', 'need', 'help', 'give', 'take', 'make', 'create',
            
            # Adjectives
            'big', 'small', 'good', 'bad', 'happy', 'sad', 'hot', 'cold', 'fast', 'slow',
            'beautiful', 'ugly', 'smart', 'stupid', 'rich', 'poor', 'young', 'old', 'new', 'old',
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink', 'purple', 'orange',
            
            # Special tokens
            '<PAD>', '<UNK>', '<START>', '<END>', '<SEP>', '<CLS>'
        ]
        
        # Add some domain-specific words
        words.extend(['python', 'programming', 'code', 'algorithm', 'data', 'machine', 'learning'])
        
        return words[:self.vocab_size]
    
    def _create_transition_matrix(self) -> np.ndarray:
        """Create a simple transition matrix simulating learned probabilities."""
        # Initialize with random probabilities
        matrix = np.random.rand(self.vocab_size, self.vocab_size)
        
        # Add some structure to make it more realistic
        # Common word pairs
        common_pairs = [
            ('the', 'cat'), ('the', 'dog'), ('the', 'house'), ('the', 'book'),
            ('a', 'big'), ('a', 'small'), ('a', 'good'), ('a', 'bad'),
            ('is', 'good'), ('is', 'bad'), ('is', 'big'), ('is', 'small'),
            ('cat', 'runs'), ('dog', 'runs'), ('man', 'walks'), ('woman', 'walks'),
            ('weather', 'is'), ('sun', 'is'), ('rain', 'is'), ('snow', 'is'),
            ('python', 'is'), ('programming', 'is'), ('code', 'is'), ('data', 'is')
        ]
        
        for word1, word2 in common_pairs:
            if word1 in self.word_to_id and word2 in self.word_to_id:
                idx1, idx2 = self.word_to_id[word1], self.word_to_id[word2]
                matrix[idx1, idx2] += 10  # Increase probability
        
        # Normalize rows
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / row_sums
        
        return matrix
    
    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization."""
        words = text.lower().split()
        return [self.word_to_id.get(word, self.word_to_id['<UNK>']) for word in words]
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text."""
        return ' '.join([self.id_to_word.get(token, '<UNK>') for token in tokens])
    
    def get_next_token_probabilities(self, context: List[int]) -> np.ndarray:
        """Get probability distribution for next token given context."""
        if not context:
            # If no context, return uniform distribution
            return np.ones(self.vocab_size) / self.vocab_size
        
        # Use the last token in context to predict next token
        last_token = context[-1]
        return self.transition_matrix[last_token]
    
    def sample_token(self, probabilities: np.ndarray, temperature: float = 1.0, 
                    top_k: Optional[int] = None, top_p: Optional[float] = None) -> int:
        """
        Sample a token using various strategies.
        
        Args:
            probabilities: Probability distribution over vocabulary
            temperature: Controls randomness (lower = more deterministic)
            top_k: Sample from top k most likely tokens
            top_p: Sample from tokens with cumulative probability >= p
        """
        # Apply temperature
        if temperature != 1.0:
            logits = np.log(probabilities + 1e-10)
            scaled_logits = logits / temperature
            probabilities = np.exp(scaled_logits)
            probabilities = probabilities / np.sum(probabilities)
        
        # Apply top-k filtering
        if top_k is not None:
            top_k = min(top_k, len(probabilities))
            top_indices = np.argsort(probabilities)[-top_k:]
            mask = np.zeros_like(probabilities)
            mask[top_indices] = 1
            probabilities = probabilities * mask
            probabilities = probabilities / np.sum(probabilities)
        
        # Apply top-p (nucleus) sampling
        if top_p is not None:
            sorted_indices = np.argsort(probabilities)[::-1]
            cumulative_probs = np.cumsum(probabilities[sorted_indices])
            cutoff_index = np.where(cumulative_probs >= top_p)[0][0] + 1
            top_indices = sorted_indices[:cutoff_index]
            mask = np.zeros_like(probabilities)
            mask[top_indices] = 1
            probabilities = probabilities * mask
            probabilities = probabilities / np.sum(probabilities)
        
        # Sample from the distribution
        return np.random.choice(len(probabilities), p=probabilities)
    
    def generate(self, prompt: str, max_length: int = 20, temperature: float = 1.0,
                top_k: Optional[int] = None, top_p: Optional[float] = None,
                stop_tokens: Optional[List[str]] = None) -> str:
        """
        Generate text continuation given a prompt.
        
        Args:
            prompt: Input text to continue
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stop_tokens: Stop generation when these tokens appear
            
        Returns:
            Generated text continuation
        """
        # Tokenize prompt
        context = self.tokenize(prompt)
        generated_tokens = context.copy()
        
        # Convert stop tokens to IDs
        stop_token_ids = []
        if stop_tokens:
            stop_token_ids = [self.word_to_id.get(token, -1) for token in stop_tokens]
        
        for _ in range(max_length):
            # Get next token probabilities
            probs = self.get_next_token_probabilities(generated_tokens)
            
            # Sample next token
            next_token = self.sample_token(probs, temperature, top_k, top_p)
            
            # Check for stop tokens
            if next_token in stop_token_ids:
                break
            
            generated_tokens.append(next_token)
        
        # Convert back to text
        return self.detokenize(generated_tokens)


class PromptEngineer:
    """Class for demonstrating prompt engineering techniques."""
    
    def __init__(self):
        self.llm = SimpleLLM()
    
    def zero_shot_prompting(self, task: str, input_text: str) -> str:
        """
        Zero-shot prompting: Give the model a task without examples.
        
        Args:
            task: Description of what to do
            input_text: Input text to process
            
        Returns:
            Generated response
        """
        prompt = f"{task}\n\nInput: {input_text}\n\nOutput:"
        return self.llm.generate(prompt, max_length=15, temperature=0.7)
    
    def few_shot_prompting(self, task: str, examples: List[Tuple[str, str]], 
                          input_text: str) -> str:
        """
        Few-shot prompting: Provide examples to guide the model.
        
        Args:
            task: Description of what to do
            examples: List of (input, output) pairs
            input_text: Input text to process
            
        Returns:
            Generated response
        """
        prompt = f"{task}\n\n"
        
        # Add examples
        for i, (example_input, example_output) in enumerate(examples):
            prompt += f"Example {i+1}:\nInput: {example_input}\nOutput: {example_output}\n\n"
        
        prompt += f"Input: {input_text}\n\nOutput:"
        return self.llm.generate(prompt, max_length=15, temperature=0.7)
    
    def chain_of_thought_prompting(self, problem: str) -> str:
        """
        Chain-of-thought prompting: Ask the model to show its reasoning.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Generated response with reasoning
        """
        prompt = f"""Solve this problem step by step:

Problem: {problem}

Let's think about this step by step:

1) First, I need to understand what's being asked...
2) Then, I should consider the relevant information...
3) Finally, I can arrive at the answer...

Answer:"""
        
        return self.llm.generate(prompt, max_length=30, temperature=0.8)
    
    def role_prompting(self, role: str, task: str, input_text: str) -> str:
        """
        Role prompting: Ask the model to act as a specific character or expert.
        
        Args:
            role: Role to play (e.g., "expert programmer", "helpful teacher")
            task: Task to perform
            input_text: Input text
            
        Returns:
            Generated response in the specified role
        """
        prompt = f"""You are a {role}. 

{task}

Input: {input_text}

Response:"""
        
        return self.llm.generate(prompt, max_length=20, temperature=0.7)
    
    def structured_prompting(self, template: str, **kwargs) -> str:
        """
        Structured prompting: Use a template with placeholders.
        
        Args:
            template: Prompt template with {placeholder} syntax
            **kwargs: Values to fill in placeholders
            
        Returns:
            Generated response
        """
        prompt = template.format(**kwargs)
        return self.llm.generate(prompt, max_length=15, temperature=0.7)


def demonstrate_sampling_strategies():
    """Demonstrate different sampling strategies."""
    
    print("=== Sampling Strategies Demonstration ===\n")
    
    llm = SimpleLLM()
    prompt = "The weather is"
    
    print(f"Prompt: '{prompt}'")
    print("\nDifferent sampling strategies:")
    
    # Get base probabilities
    context = llm.tokenize(prompt)
    probs = llm.get_next_token_probabilities(context)
    
    # Show top 5 most likely tokens
    top_indices = np.argsort(probs)[-5:][::-1]
    print(f"\nTop 5 most likely tokens:")
    for i, idx in enumerate(top_indices):
        word = llm.id_to_word[idx]
        prob = probs[idx]
        print(f"  {i+1}. '{word}': {prob:.4f}")
    
    # Test different strategies
    strategies = [
        ("Greedy (temperature=0.1)", {"temperature": 0.1}),
        ("Creative (temperature=1.5)", {"temperature": 1.5}),
        ("Top-k (k=3)", {"top_k": 3}),
        ("Top-p (p=0.9)", {"top_p": 0.9}),
        ("Balanced (temperature=0.8, top_p=0.9)", {"temperature": 0.8, "top_p": 0.9})
    ]
    
    print(f"\nGenerated continuations:")
    for strategy_name, params in strategies:
        continuation = llm.generate(prompt, max_length=5, **params)
        print(f"  {strategy_name}: {continuation}")
    
    # Visualize probability distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original distribution
    axes[0, 0].bar(range(20), probs[:20])
    axes[0, 0].set_title('Original Distribution')
    axes[0, 0].set_xlabel('Token Index')
    axes[0, 0].set_ylabel('Probability')
    
    # Low temperature
    low_temp_probs = np.exp(np.log(probs + 1e-10) / 0.1)
    low_temp_probs = low_temp_probs / np.sum(low_temp_probs)
    axes[0, 1].bar(range(20), low_temp_probs[:20])
    axes[0, 1].set_title('Low Temperature (0.1)')
    
    # High temperature
    high_temp_probs = np.exp(np.log(probs + 1e-10) / 1.5)
    high_temp_probs = high_temp_probs / np.sum(high_temp_probs)
    axes[0, 2].bar(range(20), high_temp_probs[:20])
    axes[0, 2].set_title('High Temperature (1.5)')
    
    # Top-k
    top_k_probs = probs.copy()
    top_k = 3
    top_indices = np.argsort(top_k_probs)[-top_k:]
    mask = np.zeros_like(top_k_probs)
    mask[top_indices] = 1
    top_k_probs = top_k_probs * mask
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    axes[1, 0].bar(range(20), top_k_probs[:20])
    axes[1, 0].set_title(f'Top-k (k={top_k})')
    
    # Top-p
    top_p_probs = probs.copy()
    top_p = 0.9
    sorted_indices = np.argsort(top_p_probs)[::-1]
    cumulative_probs = np.cumsum(top_p_probs[sorted_indices])
    cutoff_index = np.where(cumulative_probs >= top_p)[0][0] + 1
    top_indices = sorted_indices[:cutoff_index]
    mask = np.zeros_like(top_p_probs)
    mask[top_indices] = 1
    top_p_probs = top_p_probs * mask
    top_p_probs = top_p_probs / np.sum(top_p_probs)
    axes[1, 1].bar(range(20), top_p_probs[:20])
    axes[1, 1].set_title(f'Top-p (p={top_p})')
    
    # Combined
    combined_probs = np.exp(np.log(probs + 1e-10) / 0.8)
    combined_probs = combined_probs / np.sum(combined_probs)
    sorted_indices = np.argsort(combined_probs)[::-1]
    cumulative_probs = np.cumsum(combined_probs[sorted_indices])
    cutoff_index = np.where(cumulative_probs >= 0.9)[0][0] + 1
    top_indices = sorted_indices[:cutoff_index]
    mask = np.zeros_like(combined_probs)
    mask[top_indices] = 1
    combined_probs = combined_probs * mask
    combined_probs = combined_probs / np.sum(combined_probs)
    axes[1, 2].bar(range(20), combined_probs[:20])
    axes[1, 2].set_title('Combined (temp=0.8, top-p=0.9)')
    
    plt.tight_layout()
    plt.show()


def demonstrate_prompt_engineering():
    """Demonstrate various prompt engineering techniques."""
    
    print("\n=== Prompt Engineering Demonstration ===\n")
    
    engineer = PromptEngineer()
    
    # Zero-shot prompting
    print("1. Zero-shot Prompting:")
    task = "Translate the following text to a more formal tone:"
    input_text = "Hey, what's up? The weather is really nice today."
    response = engineer.zero_shot_prompting(task, input_text)
    print(f"Task: {task}")
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print()
    
    # Few-shot prompting
    print("2. Few-shot Prompting:")
    task = "Classify the sentiment of the following text as positive, negative, or neutral:"
    examples = [
        ("I love this movie!", "positive"),
        ("This is terrible.", "negative"),
        ("The weather is cloudy.", "neutral")
    ]
    input_text = "This restaurant has amazing food!"
    response = engineer.few_shot_prompting(task, examples, input_text)
    print(f"Task: {task}")
    print(f"Examples: {examples}")
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print()
    
    # Chain-of-thought prompting
    print("3. Chain-of-Thought Prompting:")
    problem = "If a cat has 4 legs and a dog has 4 legs, how many legs do 3 cats and 2 dogs have together?"
    response = engineer.chain_of_thought_prompting(problem)
    print(f"Problem: {problem}")
    print(f"Response: {response}")
    print()
    
    # Role prompting
    print("4. Role Prompting:")
    role = "expert programmer"
    task = "Explain what a function is in simple terms"
    input_text = "function"
    response = engineer.role_prompting(role, task, input_text)
    print(f"Role: {role}")
    print(f"Task: {task}")
    print(f"Input: {input_text}")
    print(f"Response: {response}")
    print()
    
    # Structured prompting
    print("5. Structured Prompting:")
    template = """You are a helpful assistant. 

User: {user_question}

Please provide a {tone} response that is {length} in length.

Assistant:"""
    
    response = engineer.structured_prompting(
        template,
        user_question="What is machine learning?",
        tone="educational",
        length="concise"
    )
    print(f"Template: {template}")
    print(f"Response: {response}")


def demonstrate_prompt_analysis():
    """Analyze and compare different prompts."""
    
    print("\n=== Prompt Analysis ===\n")
    
    llm = SimpleLLM()
    
    # Test different prompt formulations
    base_prompt = "Write a story about a cat"
    
    prompt_variations = [
        ("Basic", "Write a story about a cat"),
        ("Detailed", "Write a short, creative story about a cat who goes on an adventure"),
        ("Structured", "Write a story about a cat with the following structure:\n1. Introduction\n2. Conflict\n3. Resolution"),
        ("Role-based", "You are a famous children's book author. Write a story about a cat"),
        ("Few-shot", "Here are some story examples:\nExample 1: The dog ran fast.\nExample 2: The bird flew high.\n\nNow write a story about a cat"),
        ("Chain-of-thought", "Let me think about writing a story about a cat. First, I need to consider the setting, then the characters, then the plot...")
    ]
    
    print("Comparing different prompt formulations:")
    print(f"Base prompt: '{base_prompt}'")
    print()
    
    for name, prompt in prompt_variations:
        response = llm.generate(prompt, max_length=10, temperature=0.8)
        print(f"{name}:")
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Response: {response}")
        print()
    
    # Analyze prompt effectiveness
    print("Prompt Analysis Metrics:")
    print("- Length: Shorter prompts may be more focused")
    print("- Specificity: More specific prompts often produce better results")
    print("- Structure: Clear instructions help guide the model")
    print("- Examples: Few-shot prompts can improve performance")
    print("- Role: Setting a role can influence output style")


def demonstrate_practical_applications():
    """Demonstrate practical LLM applications."""
    
    print("\n=== Practical LLM Applications ===\n")
    
    llm = SimpleLLM()
    
    # 1. Text summarization
    print("1. Text Summarization:")
    long_text = "The weather today is sunny and warm. The temperature is 75 degrees Fahrenheit. People are enjoying the outdoors. Children are playing in the park. Birds are singing in the trees. It's a beautiful day."
    prompt = f"Summarize the following text in one sentence:\n\n{long_text}\n\nSummary:"
    summary = llm.generate(prompt, max_length=5, temperature=0.5)
    print(f"Original: {long_text}")
    print(f"Summary: {summary}")
    print()
    
    # 2. Code generation
    print("2. Code Generation:")
    prompt = "Write a Python function to calculate the factorial of a number:"
    code = llm.generate(prompt, max_length=15, temperature=0.3)
    print(f"Prompt: {prompt}")
    print(f"Generated code: {code}")
    print()
    
    # 3. Question answering
    print("3. Question Answering:")
    context = "Python is a programming language created by Guido van Rossum in 1991. It is known for its simplicity and readability."
    question = "Who created Python?"
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    answer = llm.generate(prompt, max_length=5, temperature=0.5)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
    
    # 4. Text classification
    print("4. Text Classification:")
    text = "I absolutely love this product! It's amazing and works perfectly."
    prompt = f"Classify the sentiment of this text as positive, negative, or neutral:\n\n{text}\n\nSentiment:"
    sentiment = llm.generate(prompt, max_length=3, temperature=0.3)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print()
    
    # 5. Creative writing
    print("5. Creative Writing:")
    prompt = "Write a haiku about programming:"
    haiku = llm.generate(prompt, max_length=10, temperature=1.0)
    print(f"Prompt: {prompt}")
    print(f"Haiku: {haiku}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_sampling_strategies()
    demonstrate_prompt_engineering()
    demonstrate_prompt_analysis()
    demonstrate_practical_applications() 