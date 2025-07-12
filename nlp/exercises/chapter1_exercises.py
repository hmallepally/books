"""
Chapter 1 Exercises: Basic NLP Concepts
=======================================

Practice exercises for fundamental NLP concepts including:
- Text preprocessing
- Tokenization
- Part-of-speech tagging
- Named Entity Recognition
- Text analysis
"""

from typing import List, Dict, Tuple
import re
from collections import Counter

# Import the BasicNLPProcessor from the code examples
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
from chapter1_basics import BasicNLPProcessor


class Chapter1Exercises:
    """Exercise problems for Chapter 1."""
    
    def __init__(self):
        self.processor = BasicNLPProcessor()
    
    def exercise_1_1_text_cleaning(self, text: str) -> str:
        """
        Exercise 1.1: Implement your own text cleaning function.
        
        Requirements:
        - Convert to lowercase
        - Remove punctuation (except apostrophes in contractions)
        - Normalize whitespace
        - Remove extra spaces
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # TODO: Implement your text cleaning function here
        # Replace this with your implementation
        pass
    
    def exercise_1_2_custom_tokenizer(self, text: str) -> List[str]:
        """
        Exercise 1.2: Implement a custom word tokenizer.
        
        Requirements:
        - Split text into words
        - Handle contractions properly (e.g., "don't" -> ["don't"])
        - Handle hyphenated words (e.g., "state-of-the-art" -> ["state-of-the-art"])
        - Remove empty tokens
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        # TODO: Implement your custom tokenizer here
        # Replace this with your implementation
        pass
    
    def exercise_1_3_sentence_boundary_detection(self, text: str) -> List[str]:
        """
        Exercise 1.3: Implement sentence boundary detection.
        
        Requirements:
        - Split text into sentences
        - Handle common abbreviations (e.g., "Mr.", "Dr.", "U.S.A.")
        - Handle multiple punctuation marks (e.g., "Really?!")
        - Handle quotes and parentheses
        
        Args:
            text: Input text to split into sentences
            
        Returns:
            List of sentences
        """
        # TODO: Implement your sentence boundary detection here
        # Replace this with your implementation
        pass
    
    def exercise_1_4_custom_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Exercise 1.4: Create a custom stop words removal function.
        
        Requirements:
        - Remove common stop words
        - Keep important words that might be stop words in context
        - Handle case sensitivity
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stop words removed
        """
        # TODO: Implement your custom stop words removal here
        # Replace this with your implementation
        pass
    
    def exercise_1_5_word_frequency_analysis(self, text: str) -> Dict[str, int]:
        """
        Exercise 1.5: Perform word frequency analysis.
        
        Requirements:
        - Clean and tokenize the text
        - Remove stop words
        - Count word frequencies
        - Return the 10 most common words
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of word frequencies
        """
        # TODO: Implement your word frequency analysis here
        # Replace this with your implementation
        pass
    
    def exercise_1_6_named_entity_extraction(self, text: str) -> List[Tuple[str, str]]:
        """
        Exercise 1.6: Extract named entities using pattern matching.
        
        Requirements:
        - Identify person names (e.g., "John Smith", "Dr. Jane Doe")
        - Identify organization names (e.g., "Google Inc.", "Microsoft Corporation")
        - Identify locations (e.g., "New York", "San Francisco, CA")
        - Use regex patterns for extraction
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of (entity, entity_type) tuples
        """
        # TODO: Implement your named entity extraction here
        # Replace this with your implementation
        pass
    
    def exercise_1_7_text_complexity_analysis(self, text: str) -> Dict[str, float]:
        """
        Exercise 1.7: Analyze text complexity.
        
        Requirements:
        - Calculate average sentence length
        - Calculate average word length
        - Calculate type-token ratio (vocabulary diversity)
        - Calculate readability score (Flesch Reading Ease)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with complexity metrics
        """
        # TODO: Implement your text complexity analysis here
        # Replace this with your implementation
        pass


def run_exercises():
    """Run all exercises with sample data."""
    
    exercises = Chapter1Exercises()
    
    # Sample texts for exercises
    sample_texts = [
        "Natural Language Processing (NLP) is a fascinating field! Dr. Smith works at Google Inc. in Mountain View, CA. The company was founded by Larry Page and Sergey Brin in 1998.",
        "I can't believe it's already Monday! The weather is beautiful today. Mr. Johnson said, 'This is amazing!' Really?!",
        "The state-of-the-art AI system achieved remarkable results. Prof. Williams from Stanford University presented the findings."
    ]
    
    print("=== Chapter 1 Exercises ===\n")
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Sample Text {i}:")
        print(f'"{text}"')
        print("\n" + "="*60 + "\n")
        
        # Exercise 1.1: Text Cleaning
        print("Exercise 1.1 - Text Cleaning:")
        try:
            cleaned = exercises.exercise_1_1_text_cleaning(text)
            print(f"Result: {cleaned}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.2: Custom Tokenizer
        print("Exercise 1.2 - Custom Tokenizer:")
        try:
            tokens = exercises.exercise_1_2_custom_tokenizer(text)
            print(f"Result: {tokens}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.3: Sentence Boundary Detection
        print("Exercise 1.3 - Sentence Boundary Detection:")
        try:
            sentences = exercises.exercise_1_3_sentence_boundary_detection(text)
            for j, sentence in enumerate(sentences, 1):
                print(f"{j}. {sentence}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.4: Custom Stop Words
        print("Exercise 1.4 - Custom Stop Words Removal:")
        try:
            # First tokenize the text
            tokens = exercises.processor.tokenize_words(text.lower())
            filtered_tokens = exercises.exercise_1_4_custom_stopwords(tokens)
            print(f"Original tokens: {tokens}")
            print(f"Filtered tokens: {filtered_tokens}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.5: Word Frequency Analysis
        print("Exercise 1.5 - Word Frequency Analysis:")
        try:
            freq_analysis = exercises.exercise_1_5_word_frequency_analysis(text)
            print("Top 10 most common words:")
            for word, count in sorted(freq_analysis.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {word}: {count}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.6: Named Entity Extraction
        print("Exercise 1.6 - Named Entity Extraction:")
        try:
            entities = exercises.exercise_1_6_named_entity_extraction(text)
            for entity, entity_type in entities:
                print(f"  {entity}: {entity_type}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "-"*40 + "\n")
        
        # Exercise 1.7: Text Complexity Analysis
        print("Exercise 1.7 - Text Complexity Analysis:")
        try:
            complexity = exercises.exercise_1_7_text_complexity_analysis(text)
            for metric, value in complexity.items():
                print(f"  {metric}: {value:.2f}")
        except NotImplementedError:
            print("Not implemented yet")
        print("\n" + "="*60 + "\n")


def provide_solutions():
    """Provide solution hints and examples."""
    
    print("=== Exercise Solutions and Hints ===\n")
    
    print("Exercise 1.1 - Text Cleaning Hints:")
    print("- Use re.sub() for pattern replacement")
    print("- Convert to lowercase with .lower()")
    print("- Use regex pattern r'[^\w\s\']' to keep only word characters, spaces, and apostrophes")
    print("- Use re.sub(r'\\s+', ' ', text) to normalize whitespace")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.2 - Custom Tokenizer Hints:")
    print("- Use re.findall() with word pattern")
    print("- Pattern: r'\\b\\w+(?:'\\w+)?\\b' for words and contractions")
    print("- Filter out empty strings")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.3 - Sentence Boundary Detection Hints:")
    print("- Use regex with lookahead/lookbehind")
    print("- Handle abbreviations: r'(?<!\\w)(?<!\\w\\.)(?<!\\w\\.\\w)(?<!\\w\\.\\w\\.)\\w\\.\\s'")
    print("- Split on sentence-ending punctuation")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.4 - Custom Stop Words Hints:")
    print("- Define a set of common stop words")
    print("- Consider context (e.g., 'can' might be important in some contexts)")
    print("- Use case-insensitive comparison")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.5 - Word Frequency Analysis Hints:")
    print("- Use Counter from collections")
    print("- Clean text first, then tokenize")
    print("- Remove stop words before counting")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.6 - Named Entity Extraction Hints:")
    print("- Use regex patterns for different entity types")
    print("- Person names: r'\\b[A-Z][a-z]+\\s+[A-Z][a-z]+\\b'")
    print("- Organizations: look for Inc., Corp., LLC, etc.")
    print("- Locations: look for city, state patterns")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.7 - Text Complexity Analysis Hints:")
    print("- Count sentences, words, and unique words")
    print("- Type-token ratio = unique_words / total_words")
    print("- Flesch Reading Ease = 206.835 - 1.015 * (total_words/total_sentences) - 84.6 * (total_syllables/total_words)")
    print("\n" + "-"*40 + "\n")


if __name__ == "__main__":
    # Run exercises
    run_exercises()
    
    # Provide solution hints
    provide_solutions() 