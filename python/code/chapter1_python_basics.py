"""
Chapter 1: Python Basics for AI/ML/Data Science
===============================================

This module demonstrates Python fundamentals and setup for AI/ML/Data Science:
- Python installation and environment setup
- Basic Python concepts
- Essential libraries and tools
- Best practices for data science
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
import json
import requests
from typing import List, Dict, Any, Optional


class PythonEnvironment:
    """Manages Python environment setup and verification."""
    
    def __init__(self):
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.packages = {}
    
    def check_python_version(self) -> bool:
        """Check if Python version is suitable for data science."""
        
        print("=== Python Version Check ===")
        print(f"Python Version: {sys.version}")
        print(f"Platform: {self.platform}")
        
        # Check minimum version (3.8+ recommended for modern data science)
        if self.python_version >= (3, 8):
            print("✓ Python version is suitable for data science")
            return True
        else:
            print("⚠ Python version is older than recommended (3.8+)")
            return False
    
    def check_essential_packages(self) -> Dict[str, bool]:
        """Check if essential data science packages are installed."""
        
        print("\n=== Essential Packages Check ===")
        
        essential_packages = {
            'numpy': 'Numerical computing',
            'pandas': 'Data manipulation',
            'matplotlib': 'Basic plotting',
            'seaborn': 'Statistical plotting',
            'scikit-learn': 'Machine learning',
            'jupyter': 'Interactive notebooks',
            'requests': 'HTTP library',
            'pathlib': 'Path handling'
        }
        
        results = {}
        
        for package, description in essential_packages.items():
            try:
                __import__(package)
                print(f"✓ {package}: {description}")
                results[package] = True
            except ImportError:
                print(f"✗ {package}: {description} - NOT INSTALLED")
                results[package] = False
        
        return results
    
    def create_virtual_environment(self, env_name: str = "datascience_env") -> bool:
        """Create a virtual environment for data science projects."""
        
        print(f"\n=== Creating Virtual Environment: {env_name} ===")
        
        try:
            # Check if venv module is available
            import venv
            
            # Create virtual environment
            venv_path = Path(env_name)
            if venv_path.exists():
                print(f"Environment {env_name} already exists")
                return True
            
            venv.create(env_name, with_pip=True)
            print(f"✓ Virtual environment created: {env_name}")
            
            # Create requirements.txt
            requirements = [
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "matplotlib>=3.4.0",
                "seaborn>=0.11.0",
                "scikit-learn>=1.0.0",
                "jupyter>=1.0.0",
                "requests>=2.25.0",
                "plotly>=5.0.0",
                "ipywidgets>=7.6.0"
            ]
            
            with open(f"{env_name}/requirements.txt", "w") as f:
                f.write("\n".join(requirements))
            
            print(f"✓ Requirements file created")
            return True
            
        except Exception as e:
            print(f"✗ Error creating virtual environment: {e}")
            return False
    
    def install_packages(self, packages: List[str]) -> bool:
        """Install packages using pip."""
        
        print(f"\n=== Installing Packages ===")
        
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Error installing {package}: {e}")
                return False
        
        return True


class PythonBasics:
    """Demonstrates basic Python concepts for data science."""
    
    def __init__(self):
        self.examples = {}
    
    def demonstrate_variables_and_types(self):
        """Demonstrate Python variables and data types."""
        
        print("=== Python Variables and Data Types ===")
        
        # Numeric types
        integer_var = 42
        float_var = 3.14159
        complex_var = 1 + 2j
        
        print(f"Integer: {integer_var} (type: {type(integer_var)})")
        print(f"Float: {float_var} (type: {type(float_var)})")
        print(f"Complex: {complex_var} (type: {type(complex_var)})")
        
        # String type
        string_var = "Hello, Data Science!"
        print(f"String: {string_var} (type: {type(string_var)})")
        
        # Boolean type
        bool_var = True
        print(f"Boolean: {bool_var} (type: {type(bool_var)})")
        
        # None type
        none_var = None
        print(f"None: {none_var} (type: {type(none_var)})")
        
        # Type conversion
        print(f"\nType Conversions:")
        print(f"String to int: {int('123')}")
        print(f"Float to int: {int(3.9)}")
        print(f"Int to float: {float(42)}")
        print(f"Number to string: {str(42)}")
        
        self.examples['variables'] = {
            'integer': integer_var,
            'float': float_var,
            'string': string_var,
            'boolean': bool_var
        }
    
    def demonstrate_control_flow(self):
        """Demonstrate Python control flow structures."""
        
        print("\n=== Control Flow Structures ===")
        
        # If-elif-else
        print("1. If-elif-else:")
        score = 85
        
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        else:
            grade = 'F'
        
        print(f"   Score: {score} -> Grade: {grade}")
        
        # For loops
        print("\n2. For loops:")
        numbers = [1, 2, 3, 4, 5]
        squared = []
        
        for num in numbers:
            squared.append(num ** 2)
        
        print(f"   Numbers: {numbers}")
        print(f"   Squared: {squared}")
        
        # List comprehension (Pythonic way)
        squared_comprehension = [num ** 2 for num in numbers]
        print(f"   Squared (comprehension): {squared_comprehension}")
        
        # While loops
        print("\n3. While loops:")
        count = 0
        while count < 3:
            print(f"   Count: {count}")
            count += 1
        
        # Break and continue
        print("\n4. Break and continue:")
        for i in range(10):
            if i == 3:
                continue  # Skip 3
            if i == 7:
                break     # Stop at 7
            print(f"   {i}", end=" ")
        print()
        
        self.examples['control_flow'] = {
            'grade': grade,
            'squared': squared,
            'count': count
        }
    
    def demonstrate_functions(self):
        """Demonstrate Python functions."""
        
        print("\n=== Functions ===")
        
        # Basic function
        def greet(name: str) -> str:
            """Return a greeting message."""
            return f"Hello, {name}!"
        
        # Function with default parameters
        def calculate_area(length: float, width: float = 1.0) -> float:
            """Calculate area of a rectangle."""
            return length * width
        
        # Function with multiple return values
        def analyze_numbers(numbers: List[float]) -> Dict[str, float]:
            """Analyze a list of numbers."""
            if not numbers:
                return {}
            
            return {
                'mean': sum(numbers) / len(numbers),
                'min': min(numbers),
                'max': max(numbers),
                'count': len(numbers)
            }
        
        # Lambda functions (anonymous functions)
        square = lambda x: x ** 2
        add = lambda x, y: x + y
        
        # Test functions
        print(f"1. Basic function: {greet('Data Scientist')}")
        print(f"2. Default parameters: {calculate_area(5)}")
        print(f"3. Multiple return values: {analyze_numbers([1, 2, 3, 4, 5])}")
        print(f"4. Lambda functions: square(4) = {square(4)}, add(2, 3) = {add(2, 3)}")
        
        self.examples['functions'] = {
            'greet': greet,
            'calculate_area': calculate_area,
            'analyze_numbers': analyze_numbers
        }
    
    def demonstrate_data_structures(self):
        """Demonstrate Python data structures."""
        
        print("\n=== Data Structures ===")
        
        # Lists
        print("1. Lists:")
        my_list = [1, 2, 3, 4, 5]
        my_list.append(6)
        my_list.insert(0, 0)
        print(f"   List: {my_list}")
        print(f"   Length: {len(my_list)}")
        print(f"   First element: {my_list[0]}")
        print(f"   Last element: {my_list[-1]}")
        print(f"   Slice [1:4]: {my_list[1:4]}")
        
        # Tuples (immutable)
        print("\n2. Tuples:")
        my_tuple = (1, 2, 3, 'hello')
        print(f"   Tuple: {my_tuple}")
        print(f"   Type: {type(my_tuple)}")
        
        # Dictionaries
        print("\n3. Dictionaries:")
        my_dict = {
            'name': 'John',
            'age': 30,
            'city': 'New York',
            'skills': ['Python', 'ML', 'Data Science']
        }
        print(f"   Dictionary: {my_dict}")
        print(f"   Keys: {list(my_dict.keys())}")
        print(f"   Values: {list(my_dict.values())}")
        print(f"   Age: {my_dict['age']}")
        
        # Sets
        print("\n4. Sets:")
        my_set = {1, 2, 3, 3, 4, 4, 5}  # Duplicates are automatically removed
        print(f"   Set: {my_set}")
        print(f"   Length: {len(my_set)}")
        
        # Set operations
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        print(f"   Set1: {set1}")
        print(f"   Set2: {set2}")
        print(f"   Union: {set1 | set2}")
        print(f"   Intersection: {set1 & set2}")
        print(f"   Difference: {set1 - set2}")
        
        self.examples['data_structures'] = {
            'list': my_list,
            'tuple': my_tuple,
            'dict': my_dict,
            'set': my_set
        }
    
    def demonstrate_error_handling(self):
        """Demonstrate Python error handling."""
        
        print("\n=== Error Handling ===")
        
        # Try-except blocks
        print("1. Basic try-except:")
        
        try:
            result = 10 / 0
        except ZeroDivisionError as e:
            print(f"   Error caught: {e}")
        
        # Multiple except blocks
        print("\n2. Multiple except blocks:")
        
        try:
            value = int("not_a_number")
        except ValueError as e:
            print(f"   ValueError: {e}")
        except TypeError as e:
            print(f"   TypeError: {e}")
        except Exception as e:
            print(f"   General error: {e}")
        
        # Try-except-else-finally
        print("\n3. Try-except-else-finally:")
        
        try:
            result = 10 / 2
        except ZeroDivisionError as e:
            print(f"   Error: {e}")
        else:
            print(f"   Success: {result}")
        finally:
            print("   This always executes")
        
        # Custom exceptions
        print("\n4. Custom exceptions:")
        
        class DataValidationError(Exception):
            """Custom exception for data validation errors."""
            pass
        
        def validate_age(age):
            if age < 0:
                raise DataValidationError("Age cannot be negative")
            if age > 150:
                raise DataValidationError("Age seems unrealistic")
            return True
        
        try:
            validate_age(-5)
        except DataValidationError as e:
            print(f"   Custom error: {e}")
    
    def demonstrate_file_operations(self):
        """Demonstrate Python file operations."""
        
        print("\n=== File Operations ===")
        
        # Writing to a file
        print("1. Writing to a file:")
        
        with open('sample_data.txt', 'w') as f:
            f.write("Name,Age,City\n")
            f.write("John,30,New York\n")
            f.write("Alice,25,Los Angeles\n")
            f.write("Bob,35,Chicago\n")
        
        print("   Created sample_data.txt")
        
        # Reading from a file
        print("\n2. Reading from a file:")
        
        with open('sample_data.txt', 'r') as f:
            content = f.read()
        
        print(f"   File content:\n{content}")
        
        # Reading line by line
        print("\n3. Reading line by line:")
        
        with open('sample_data.txt', 'r') as f:
            for i, line in enumerate(f, 1):
                print(f"   Line {i}: {line.strip()}")
        
        # Working with JSON
        print("\n4. JSON operations:")
        
        data = {
            'name': 'John Doe',
            'age': 30,
            'skills': ['Python', 'ML', 'Data Science'],
            'active': True
        }
        
        # Write JSON
        with open('data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print("   Created data.json")
        
        # Read JSON
        with open('data.json', 'r') as f:
            loaded_data = json.load(f)
        
        print(f"   Loaded data: {loaded_data}")
        
        # Clean up
        for filename in ['sample_data.txt', 'data.json']:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"   Cleaned up {filename}")


class DataScienceTools:
    """Demonstrates essential tools for data science."""
    
    def __init__(self):
        self.tools = {}
    
    def demonstrate_pathlib(self):
        """Demonstrate pathlib for file path handling."""
        
        print("\n=== Pathlib for File Paths ===")
        
        # Current directory
        current_dir = Path.cwd()
        print(f"Current directory: {current_dir}")
        
        # Create directories
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        print(f"Created directory: {data_dir}")
        
        # File operations
        file_path = data_dir / "sample.csv"
        print(f"File path: {file_path}")
        print(f"File exists: {file_path.exists()}")
        print(f"File extension: {file_path.suffix}")
        print(f"File name: {file_path.name}")
        print(f"Parent directory: {file_path.parent}")
        
        # List files
        print(f"\nFiles in current directory:")
        for item in Path.cwd().iterdir():
            if item.is_file():
                print(f"   File: {item.name}")
            elif item.is_dir():
                print(f"   Directory: {item.name}")
        
        # Clean up
        if data_dir.exists():
            data_dir.rmdir()
            print(f"Cleaned up {data_dir}")
    
    def demonstrate_requests(self):
        """Demonstrate requests library for HTTP operations."""
        
        print("\n=== Requests Library ===")
        
        try:
            # GET request
            response = requests.get('https://httpbin.org/get')
            print(f"GET request status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)[:3]}...")
            
            # POST request
            data = {'name': 'John', 'age': 30}
            response = requests.post('https://httpbin.org/post', json=data)
            print(f"POST request status: {response.status_code}")
            
            # Check if request was successful
            if response.status_code == 200:
                print("   Request successful")
            else:
                print(f"   Request failed with status {response.status_code}")
                
        except requests.RequestException as e:
            print(f"   Network error: {e}")
    
    def demonstrate_environment_info(self):
        """Display environment information."""
        
        print("\n=== Environment Information ===")
        
        info = {
            'Python Version': sys.version,
            'Platform': platform.platform(),
            'Architecture': platform.architecture(),
            'Machine': platform.machine(),
            'Processor': platform.processor(),
            'Python Executable': sys.executable,
            'Working Directory': os.getcwd()
        }
        
        for key, value in info.items():
            print(f"{key}: {value}")


def run_demonstrations():
    """Run all Python basics demonstrations."""
    
    print("Python Basics for AI/ML/Data Science")
    print("=" * 50)
    
    # Environment setup
    env = PythonEnvironment()
    env.check_python_version()
    env.check_essential_packages()
    
    # Python basics
    basics = PythonBasics()
    basics.demonstrate_variables_and_types()
    basics.demonstrate_control_flow()
    basics.demonstrate_functions()
    basics.demonstrate_data_structures()
    basics.demonstrate_error_handling()
    basics.demonstrate_file_operations()
    
    # Data science tools
    tools = DataScienceTools()
    tools.demonstrate_pathlib()
    tools.demonstrate_requests()
    tools.demonstrate_environment_info()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")
    print("=" * 50)
    
    return env, basics, tools


def create_sample_project():
    """Create a sample data science project structure."""
    
    print("\n=== Creating Sample Project Structure ===")
    
    # Project structure
    project_structure = {
        'my_datascience_project': {
            'data': {},
            'src': {
                '__init__.py': '',
                'data_processing.py': '# Data processing functions\n',
                'models.py': '# ML model definitions\n',
                'utils.py': '# Utility functions\n'
            },
            'notebooks': {
                '01_exploratory_analysis.ipynb': '# Exploratory data analysis\n',
                '02_model_training.ipynb': '# Model training and evaluation\n'
            },
            'tests': {
                '__init__.py': '',
                'test_data_processing.py': '# Tests for data processing\n'
            },
            'requirements.txt': '''numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
pytest>=6.0.0''',
            'README.md': '''# My Data Science Project

This is a sample data science project demonstrating best practices.

## Setup
1. Create virtual environment: `python -m venv venv`
2. Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
3. Install requirements: `pip install -r requirements.txt`

## Usage
- Run notebooks in the `notebooks/` directory
- Use functions from the `src/` directory
- Run tests with `pytest tests/`
'''
        }
    }
    
    def create_structure(structure, parent_path=Path('.')):
        for name, content in structure.items():
            path = parent_path / name
            if isinstance(content, dict):
                path.mkdir(exist_ok=True)
                create_structure(content, path)
            else:
                path.write_text(content)
    
    create_structure(project_structure)
    print("✓ Sample project structure created")
    print("  - my_datascience_project/")
    print("    - data/ (for datasets)")
    print("    - src/ (source code)")
    print("    - notebooks/ (Jupyter notebooks)")
    print("    - tests/ (unit tests)")
    print("    - requirements.txt (dependencies)")
    print("    - README.md (documentation)")


if __name__ == "__main__":
    # Run demonstrations
    env, basics, tools = run_demonstrations()
    
    # Create sample project
    create_sample_project()
    
    print("\nKey Takeaways:")
    print("1. Python provides excellent tools for data science")
    print("2. Virtual environments keep projects isolated")
    print("3. Essential packages: numpy, pandas, matplotlib, scikit-learn")
    print("4. Good project structure improves maintainability")
    print("5. Error handling and file operations are crucial") 