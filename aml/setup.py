#!/usr/bin/env python3
"""
Setup script for Machine Learning & Advanced Machine Learning Book
================================================================

This script automates the setup of the Python environment for the ML book,
including package installation, data download, and basic testing.
"""

import sys
import subprocess
import os
import platform
from pathlib import Path
import importlib.util


def check_python_version():
    """Check if Python version is suitable for ML development."""
    print("=== Checking Python Version ===")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 8):
        print("❌ Python 3.8 or higher is required for modern ML libraries")
        print("Please upgrade your Python installation")
        return False
    elif version < (3, 9):
        print("⚠️  Python 3.9+ is recommended for best compatibility")
        print("Current version will work but consider upgrading")
        return True
    else:
        print("✅ Python version is suitable for ML development")
        return True


def check_pip():
    """Check if pip is available and working."""
    print("\n=== Checking pip ===")
    
    try:
        import pip
        print(f"✅ pip version: {pip.__version__}")
        return True
    except ImportError:
        print("❌ pip is not installed")
        return False


def install_package(package, upgrade=False):
    """Install a package using pip."""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def install_requirements():
    """Install required packages from requirements.txt."""
    print("\n=== Installing Required Packages ===")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        print(f"Error output: {e.stderr}")
        return False


def verify_installations():
    """Verify that all essential packages are installed."""
    print("\n=== Verifying Package Installations ===")
    
    essential_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'jupyter',
        'requests'
    ]
    
    optional_packages = [
        'tensorflow',
        'torch',
        'plotly',
        'bokeh'
    ]
    
    print("Essential packages:")
    for package in essential_packages:
        if check_package(package):
            print(f"  ✅ {package}")
        else:
            print(f"  ❌ {package}")
    
    print("\nOptional packages:")
    for package in optional_packages:
        if check_package(package):
            print(f"  ✅ {package}")
        else:
            print(f"  ⚠️  {package} (not installed)")


def create_directories():
    """Create necessary directories for the book."""
    print("\n=== Creating Directory Structure ===")
    
    directories = [
        "data",
        "models", 
        "results",
        "logs",
        "notebooks",
        "src",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def download_sample_data():
    """Download sample datasets for exercises."""
    print("\n=== Downloading Sample Data ===")
    
    try:
        import sklearn.datasets
        
        # Download some sample datasets
        datasets = {
            'iris': sklearn.datasets.load_iris,
            'breast_cancer': sklearn.datasets.load_breast_cancer,
            'diabetes': sklearn.datasets.load_diabetes,
            'boston': sklearn.datasets.load_boston
        }
        
        for name, loader in datasets.items():
            try:
                data = loader()
                print(f"✅ Downloaded {name} dataset")
            except Exception as e:
                print(f"⚠️  Could not download {name} dataset: {e}")
                
    except ImportError:
        print("⚠️  scikit-learn not available for data download")


def run_basic_tests():
    """Run basic tests to verify the setup."""
    print("\n=== Running Basic Tests ===")
    
    test_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate sample data
X, y = make_classification(n_samples=100, n_features=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy:.3f}")
print("✅ Basic ML workflow test passed!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Basic test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def create_jupyter_kernel():
    """Create a Jupyter kernel for the project."""
    print("\n=== Setting up Jupyter Kernel ===")
    
    try:
        # Install ipykernel if not already installed
        if not check_package('ipykernel'):
            install_package('ipykernel')
        
        # Create kernel
        kernel_name = "ml-book"
        cmd = [sys.executable, "-m", "ipykernel", "install", 
               "--user", "--name", kernel_name, "--display-name", "ML Book"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Jupyter kernel '{kernel_name}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create Jupyter kernel: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Open notebooks/01_ml_basics.ipynb")
    print("4. Begin with Chapter 1 exercises")
    print("\nDirectory structure created:")
    print("  📁 data/          - Store your datasets")
    print("  📁 models/        - Save trained models")
    print("  📁 results/       - Save analysis results")
    print("  📁 notebooks/     - Jupyter notebooks")
    print("  📁 src/           - Source code")
    print("  📁 tests/         - Unit tests")
    print("\nHappy learning! 🚀")


def main():
    """Main setup function."""
    print("Machine Learning Book Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        print("Please install pip first")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements")
        sys.exit(1)
    
    # Verify installations
    verify_installations()
    
    # Create directories
    create_directories()
    
    # Download sample data
    download_sample_data()
    
    # Run basic tests
    if not run_basic_tests():
        print("Warning: Basic tests failed")
    
    # Create Jupyter kernel
    create_jupyter_kernel()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 