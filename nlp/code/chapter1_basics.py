"""
Chapter 1: Basic NLP Concepts and Text Preprocessing
===================================================

This module demonstrates fundamental NLP concepts including:
- Text preprocessing
- Tokenization
- Part-of-speech tagging
- Named Entity Recognition
- Basic text analysis
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
import spacy

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class BasicNLPProcessor:
    """A basic NLP processor demonstrating fundamental concepts."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model for advanced NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: remove special characters, normalize whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into individual words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        return word_tokenize(text)
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentence tokens
        """
        return sent_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stop words from token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of tokens with stop words removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def stem_words(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to reduce words to their root form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_words(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to reduce words to their dictionary form.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_pos_tags(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of (word, pos_tag) tuples
        """
        return pos_tag(tokens)
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities from text using NLTK.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity, entity_type) tuples
        """
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        named_entities = ne_chunk(pos_tags)
        
        entities = []
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity = ' '.join(c[0] for c in chunk)
                entity_type = chunk.label()
                entities.append((entity, entity_type))
        
        return entities
    
    def extract_named_entities_spacy(self, text: str) -> List[Dict]:
        """
        Extract named entities using spaCy (more accurate).
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries with text, label, and position
        """
        if self.nlp is None:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def get_word_frequencies(self, tokens: List[str]) -> Dict[str, int]:
        """
        Calculate word frequencies in a token list.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Dictionary mapping words to their frequencies
        """
        return dict(Counter(tokens))
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing various text analysis results
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        sentences = self.tokenize_sentences(text)
        words = self.tokenize_words(cleaned_text)
        
        # Remove stop words
        words_no_stop = self.remove_stopwords(words)
        
        # Apply stemming and lemmatization
        stemmed_words = self.stem_words(words)
        lemmatized_words = self.lemmatize_words(words)
        
        # Get POS tags
        pos_tags = self.get_pos_tags(words)
        
        # Extract named entities
        entities_nltk = self.extract_named_entities(text)
        entities_spacy = self.extract_named_entities_spacy(text)
        
        # Calculate statistics
        word_freq = self.get_word_frequencies(words)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_unique_words': len(set(words)),
            'sentences': sentences,
            'words': words,
            'words_no_stopwords': words_no_stop,
            'stemmed_words': stemmed_words,
            'lemmatized_words': lemmatized_words,
            'pos_tags': pos_tags,
            'named_entities_nltk': entities_nltk,
            'named_entities_spacy': entities_spacy,
            'word_frequencies': word_freq,
            'most_common_words': Counter(words).most_common(10)
        }


def demonstrate_basic_nlp():
    """Demonstrate basic NLP concepts with examples."""
    
    # Sample text for analysis
    sample_text = """
    Natural Language Processing (NLP) is a subfield of artificial intelligence 
    that focuses on the interaction between computers and human language. 
    John Smith works at Google Inc. in Mountain View, California. 
    The company was founded by Larry Page and Sergey Brin in 1998.
    """
    
    print("=== Basic NLP Concepts Demonstration ===\n")
    
    # Initialize processor
    processor = BasicNLPProcessor()
    
    # Perform comprehensive analysis
    analysis = processor.analyze_text(sample_text)
    
    print("Original Text:")
    print(analysis['original_text'])
    print("\n" + "="*50 + "\n")
    
    print("Text Statistics:")
    print(f"Number of sentences: {analysis['num_sentences']}")
    print(f"Number of words: {analysis['num_words']}")
    print(f"Number of unique words: {analysis['num_unique_words']}")
    print("\n" + "="*50 + "\n")
    
    print("Sentence Tokenization:")
    for i, sentence in enumerate(analysis['sentences'], 1):
        print(f"{i}. {sentence.strip()}")
    print("\n" + "="*50 + "\n")
    
    print("Word Tokenization:")
    print(analysis['words'])
    print("\n" + "="*50 + "\n")
    
    print("Words without Stop Words:")
    print(analysis['words_no_stopwords'])
    print("\n" + "="*50 + "\n")
    
    print("Stemmed Words:")
    print(analysis['stemmed_words'])
    print("\n" + "="*50 + "\n")
    
    print("Lemmatized Words:")
    print(analysis['lemmatized_words'])
    print("\n" + "="*50 + "\n")
    
    print("Part-of-Speech Tags:")
    for word, tag in analysis['pos_tags']:
        print(f"{word}: {tag}")
    print("\n" + "="*50 + "\n")
    
    print("Named Entities (NLTK):")
    for entity, entity_type in analysis['named_entities_nltk']:
        print(f"{entity}: {entity_type}")
    print("\n" + "="*50 + "\n")
    
    if analysis['named_entities_spacy']:
        print("Named Entities (spaCy):")
        for entity in analysis['named_entities_spacy']:
            print(f"{entity['text']}: {entity['label']}")
        print("\n" + "="*50 + "\n")
    
    print("Most Common Words:")
    for word, count in analysis['most_common_words']:
        print(f"{word}: {count}")


def demonstrate_ambiguity_examples():
    """Demonstrate various types of linguistic ambiguity."""
    
    print("\n=== Linguistic Ambiguity Examples ===\n")
    
    # Lexical ambiguity examples
    lexical_examples = [
        "I went to the bank to deposit money.",
        "I sat by the bank of the river.",
        "The chicken is ready to eat.",
        "Time flies like an arrow."
    ]
    
    print("Lexical Ambiguity Examples:")
    for i, example in enumerate(lexical_examples, 1):
        print(f"{i}. {example}")
    print("\n" + "="*50 + "\n")
    
    # Syntactic ambiguity examples
    syntactic_examples = [
        "I saw the man with the telescope.",
        "The old men and women were evacuated.",
        "They are cooking apples."
    ]
    
    print("Syntactic Ambiguity Examples:")
    for i, example in enumerate(syntactic_examples, 1):
        print(f"{i}. {example}")
    print("\n" + "="*50 + "\n")
    
    # Context dependence examples
    context_examples = [
        "I'm feeling blue today.",
        "The blue car is faster than the red one.",
        "Oh, great, another Monday!",
        "This is a great movie!"
    ]
    
    print("Context Dependence Examples:")
    for i, example in enumerate(context_examples, 1):
        print(f"{i}. {example}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_basic_nlp()
    demonstrate_ambiguity_examples() 