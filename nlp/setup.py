#!/usr/bin/env python3
"""
Setup script for the NLP Comprehensive Primer
=============================================

This script helps set up the environment for the NLP book examples.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_requirements():
    """Install required packages."""
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    )


def download_nltk_data():
    """Download required NLTK data."""
    nltk_script = """
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK data
data_packages = [
    'punkt',
    'stopwords', 
    'wordnet',
    'averaged_perceptron_tagger',
    'maxent_ne_chunker',
    'words'
]

for package in data_packages:
    try:
        nltk.download(package, quiet=True)
        print(f"Downloaded {package}")
    except Exception as e:
        print(f"Failed to download {package}: {e}")
"""
    
    return run_command(
        f"{sys.executable} -c \"{nltk_script}\"",
        "Downloading NLTK data"
    )


def download_spacy_model():
    """Download spaCy English model."""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading spaCy English model"
    )


def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models', 'outputs', 'notebooks']
    
    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    test_script = """
try:
    import numpy
    import pandas
    import sklearn
    import nltk
    import spacy
    import matplotlib
    import torch
    import transformers
    print("✓ All core packages imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
"""
    
    return run_command(
        f"{sys.executable} -c \"{test_script}\"",
        "Testing package imports"
    )


def run_basic_example():
    """Run a basic example to verify everything works."""
    print("\nRunning basic example...")
    
    if os.path.exists('code/chapter1_basics.py'):
        return run_command(
            f"{sys.executable} code/chapter1_basics.py",
            "Running Chapter 1 basic example"
        )
    else:
        print("✗ Chapter 1 example not found")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("NLP Comprehensive Primer - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n✗ Setup failed: Incompatible Python version")
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Setup failed: Could not install requirements")
        return False
    
    # Download NLTK data
    if not download_nltk_data():
        print("\n⚠ Warning: Could not download all NLTK data")
    
    # Download spaCy model
    if not download_spacy_model():
        print("\n⚠ Warning: Could not download spaCy model")
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("\n✗ Setup failed: Import test failed")
        return False
    
    # Run basic example
    if not run_basic_example():
        print("\n⚠ Warning: Basic example failed")
    
    print("\n" + "=" * 60)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Read the book: book/nlp-comprehensive-primer.md")
    print("2. Run examples: python code/chapter1_basics.py")
    print("3. Try exercises: python exercises/chapter1_exercises.py")
    print("4. Explore Jupyter notebooks in the notebooks/ directory")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 