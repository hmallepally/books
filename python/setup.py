#!/usr/bin/env python3
"""
Setup script for Python for AI/ML/Data Science Book
==================================================

This script automates the setup of the Python environment for the Python book,
including package installation, environment setup, and basic testing.
"""

import sys
import subprocess
import os
import platform
from pathlib import Path
import importlib.util


def check_python_version():
    """Check if Python version is suitable for data science."""
    print("=== Checking Python Version ===")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 7):
        print("❌ Python 3.7 or higher is required for data science")
        print("Please upgrade your Python installation")
        return False
    elif version < (3, 8):
        print("⚠️  Python 3.8+ is recommended for best compatibility")
        print("Current version will work but consider upgrading")
        return True
    else:
        print("✅ Python version is suitable for data science")
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
        "src",
        "notebooks",
        "exercises",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_sample_files():
    """Create sample files for learning."""
    print("\n=== Creating Sample Files ===")
    
    # Create sample Python script
    sample_script = '''#!/usr/bin/env python3
"""
Sample Python script for data science.
This file demonstrates basic Python concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Main function demonstrating basic operations."""
    print("Hello, Data Science!")
    
    # Create sample data
    data = np.random.randn(100)
    print(f"Generated {len(data)} random numbers")
    print(f"Mean: {np.mean(data):.3f}")
    print(f"Standard deviation: {np.std(data):.3f}")
    
    # Create a simple plot
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, alpha=0.7, color='skyblue')
    plt.title('Sample Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
'''
    
    with open("src/sample_script.py", "w") as f:
        f.write(sample_script)
    print("✅ Created src/sample_script.py")
    
    # Create sample data file
    sample_data = '''Name,Age,City,Salary
John,30,New York,50000
Alice,25,Los Angeles,60000
Bob,35,Chicago,70000
Charlie,28,Boston,55000
Diana,32,Seattle,65000
'''
    
    with open("data/sample_data.csv", "w") as f:
        f.write(sample_data)
    print("✅ Created data/sample_data.csv")
    
    # Create README for the project
    readme_content = '''# Python for AI/ML/Data Science

This project contains examples and exercises for learning Python for AI, Machine Learning, and Data Science.

## Directory Structure

- `data/` - Sample datasets and data files
- `src/` - Source code and examples
- `notebooks/` - Jupyter notebooks
- `exercises/` - Practice exercises
- `tests/` - Unit tests
- `docs/` - Documentation

## Getting Started

1. Make sure you have Python 3.8+ installed
2. Install required packages: `pip install -r requirements.txt`
3. Start Jupyter: `jupyter notebook`
4. Open notebooks/01_python_basics.ipynb

## Learning Path

1. **Chapter 1**: Python Basics and Environment Setup
2. **Chapter 2**: Data Structures and Control Flow
3. **Chapter 3**: Functions and Object-Oriented Programming
4. **Chapter 4**: Data Science Libraries (NumPy, Pandas)
5. **Chapter 5**: Visualization and Analysis
6. **Chapter 6**: Best Practices and Project Structure

Happy learning! 🐍📊
'''
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✅ Created README.md")


def run_basic_tests():
    """Run basic tests to verify the setup."""
    print("\n=== Running Basic Tests ===")
    
    test_code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Test NumPy
arr = np.array([1, 2, 3, 4, 5])
print(f"NumPy test: {arr.mean():.1f}")

# Test Pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(f"Pandas test: {df.shape}")

# Test Matplotlib
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
plt.close()  # Close to avoid displaying
print("Matplotlib test: OK")

print("✅ All basic tests passed!")
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
        kernel_name = "python-datascience"
        cmd = [sys.executable, "-m", "ipykernel", "install", 
               "--user", "--name", kernel_name, "--display-name", "Python Data Science"]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Jupyter kernel '{kernel_name}' created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create Jupyter kernel: {e}")
        return False


def demonstrate_python_basics():
    """Demonstrate basic Python concepts."""
    print("\n=== Python Basics Demonstration ===")
    
    demo_code = """
# Variables and data types
name = "Python"
version = 3.9
is_awesome = True

print(f"Language: {name}")
print(f"Version: {version}")
print(f"Awesome: {is_awesome}")

# Lists and list comprehensions
numbers = [1, 2, 3, 4, 5]
squares = [x**2 for x in numbers]
print(f"Numbers: {numbers}")
print(f"Squares: {squares}")

# Dictionaries
person = {'name': 'John', 'age': 30, 'city': 'NYC'}
print(f"Person: {person}")

# Functions
def greet(name):
    return f"Hello, {name}!"

print(greet("Data Scientist"))

print("✅ Python basics demonstration completed!")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", demo_code], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Python basics demo failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("PYTHON SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Open notebooks/01_python_basics.ipynb")
    print("4. Run the sample script: python src/sample_script.py")
    print("5. Begin with Chapter 1 exercises")
    print("\nDirectory structure created:")
    print("  📁 data/          - Sample datasets")
    print("  📁 src/           - Source code examples")
    print("  📁 notebooks/     - Jupyter notebooks")
    print("  📁 exercises/     - Practice exercises")
    print("  📁 tests/         - Unit tests")
    print("  📁 docs/          - Documentation")
    print("\nSample files created:")
    print("  📄 src/sample_script.py - Basic Python example")
    print("  📄 data/sample_data.csv - Sample dataset")
    print("  📄 README.md - Project documentation")
    print("\nHappy coding! 🐍🚀")


def main():
    """Main setup function."""
    print("Python for AI/ML/Data Science Setup")
    print("=" * 45)
    
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
    
    # Create sample files
    create_sample_files()
    
    # Run basic tests
    if not run_basic_tests():
        print("Warning: Basic tests failed")
    
    # Demonstrate Python basics
    demonstrate_python_basics()
    
    # Create Jupyter kernel
    create_jupyter_kernel()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 