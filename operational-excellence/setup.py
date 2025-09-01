#!/usr/bin/env python3
"""
Setup script for Harnessing AI for Performance Optimization
A Comprehensive Guide to Operational Excellence in the AI Era
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 80)
    print("🚀 Harnessing AI for Performance Optimization")
    print("   A Comprehensive Guide to Operational Excellence in the AI Era")
    print("   Designed for Managers, Executives, and MBA Students")
    print("=" * 80)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    print()

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("   Please try installing manually: pip install -r requirements.txt")
        sys.exit(1)
    print()

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    directories = [
        "code",
        "exercises", 
        "notebooks",
        "data",
        "images",
        "models",
        "results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✅ Created {directory}/")
    print()

def create_sample_data():
    """Create sample data files"""
    print("📊 Creating sample data files...")
    
    # Sample operational data
    operational_data = """timestamp,process_id,temperature,pressure,quality_score,defect_rate
2024-01-01 08:00:00,P001,185.5,2.3,0.95,0.02
2024-01-01 08:15:00,P001,186.2,2.4,0.94,0.03
2024-01-01 08:30:00,P001,184.8,2.2,0.96,0.01
2024-01-01 08:45:00,P001,187.1,2.5,0.93,0.04
2024-01-01 09:00:00,P001,185.9,2.3,0.95,0.02"""
    
    with open("data/sample_operational_data.csv", "w") as f:
        f.write(operational_data)
    print("   ✅ Created data/sample_operational_data.csv")
    
    # Sample quality metrics
    quality_data = """date,product_line,first_pass_yield,defect_rate,customer_satisfaction
2024-01-01,Line_A,0.95,0.02,4.8
2024-01-02,Line_A,0.94,0.03,4.7
2024-01-03,Line_A,0.96,0.01,4.9
2024-01-04,Line_A,0.93,0.04,4.6
2024-01-05,Line_A,0.95,0.02,4.8"""
    
    with open("data/quality_metrics.csv", "w") as f:
        f.write(quality_data)
    print("   ✅ Created data/quality_metrics.csv")
    
    # Sample supply chain data
    supply_chain_data = """supplier_id,product_id,lead_time,cost,quality_rating,delivery_performance
SUP001,PROD_A,5,100.50,4.8,0.98
SUP002,PROD_B,7,85.20,4.6,0.95
SUP003,PROD_C,3,120.75,4.9,0.99
SUP004,PROD_D,6,95.30,4.7,0.96
SUP005,PROD_E,4,110.45,4.8,0.97"""
    
    with open("data/supply_chain_data.csv", "w") as f:
        f.write(supply_chain_data)
    print("   ✅ Created data/supply_chain_data.csv")
    
    # Sample customer feedback
    customer_feedback = """customer_id,product_id,rating,feedback_text,sentiment
CUST001,PROD_A,5,Excellent product quality and fast delivery,positive
CUST002,PROD_B,4,Good product but delivery was slightly delayed,neutral
CUST003,PROD_C,5,Outstanding quality and customer service,positive
CUST004,PROD_D,3,Product was okay but packaging could be better,negative
CUST005,PROD_E,4,Good value for money,positive"""
    
    with open("data/customer_feedback.csv", "w") as f:
        f.write(customer_feedback)
    print("   ✅ Created data/customer_feedback.csv")
    print()

def create_sample_code():
    """Create sample code files"""
    print("💻 Creating sample code files...")
    
    # Chapter 1: AI Operations
    chapter1_code = '''"""
Chapter 1: Introduction to AI in Operations
Sample code demonstrating AI-enhanced operational excellence
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class AIOperationsFramework:
    """AI-enhanced operations framework"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.is_trained = False
    
    def load_data(self, filepath):
        """Load operational data"""
        return pd.read_csv(filepath)
    
    def preprocess_data(self, data):
        """Preprocess operational data"""
        # Add your preprocessing logic here
        return data
    
    def train_model(self, X, y):
        """Train AI model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self.model.score(X_test, y_test)
    
    def predict_performance(self, X):
        """Predict operational performance"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)

def main():
    """Main function demonstrating AI operations"""
    print("🚀 AI Operations Framework Demo")
    
    # Initialize framework
    framework = AIOperationsFramework()
    
    # Load sample data
    try:
        data = framework.load_data("data/sample_operational_data.csv")
        print(f"✅ Loaded data with {len(data)} records")
        
        # Example preprocessing and training
        print("📊 Data loaded successfully!")
        print("   Ready for AI-enhanced operational excellence!")
        
    except FileNotFoundError:
        print("⚠️  Sample data not found. Please ensure data files are created.")
    
    print("\\n🎯 Ready to explore AI-enhanced operational excellence!")

if __name__ == "__main__":
    main()
'''
    
    with open("code/chapter1_ai_operations.py", "w") as f:
        f.write(chapter1_code)
    print("   ✅ Created code/chapter1_ai_operations.py")
    
    # Chapter 2: Quality Control
    chapter2_code = '''"""
Chapter 2: AI-Powered Quality Control and Manufacturing
Sample code demonstrating AI-enhanced quality control
"""

import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

class AIQualityControl:
    """AI-powered quality control system"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.is_trained = False
    
    def detect_defects(self, image):
        """Detect defects in manufacturing images"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return len(contours) > 0
    
    def analyze_quality_metrics(self, data):
        """Analyze quality metrics using AI"""
        if not self.is_trained:
            self.anomaly_detector.fit(data)
            self.is_trained = True
        
        # Detect anomalies
        predictions = self.anomaly_detector.predict(data)
        return predictions
    
    def predict_quality_issues(self, historical_data):
        """Predict potential quality issues"""
        # Add your prediction logic here
        return "Quality prediction model ready!"

def main():
    """Main function demonstrating AI quality control"""
    print("🔍 AI Quality Control System Demo")
    
    # Initialize quality control system
    qc_system = AIQualityControl()
    
    # Load quality metrics
    try:
        quality_data = pd.read_csv("data/quality_metrics.csv")
        print(f"✅ Loaded quality data with {len(quality_data)} records")
        
        # Example quality analysis
        print("📊 Quality control system ready!")
        print("   Ready for AI-enhanced quality management!")
        
    except FileNotFoundError:
        print("⚠️  Quality data not found. Please ensure data files are created.")
    
    print("\\n🎯 Ready to explore AI-enhanced quality control!")

if __name__ == "__main__":
    main()
'''
    
    with open("code/chapter2_quality_control.py", "w") as f:
        f.write(chapter2_code)
    print("   ✅ Created code/chapter2_quality_control.py")
    print()

def run_tests():
    """Run basic tests to verify setup"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import sklearn
        print("   ✅ Core libraries imported successfully")
        
        # Test data loading
        if os.path.exists("data/sample_operational_data.csv"):
            data = pd.read_csv("data/sample_operational_data.csv")
            print(f"   ✅ Sample data loaded: {len(data)} records")
        
        print("   ✅ All tests passed!")
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   Please ensure all packages are installed correctly")
    except Exception as e:
        print(f"   ❌ Test error: {e}")
    
    print()

def print_next_steps():
    """Print next steps for the user"""
    print("🎯 Next Steps:")
    print("   1. 📖 Read Executive Summary: EXECUTIVE_SUMMARY.md")
    print("   2. 📚 Read the main book: book/harnessing-ai-for-performance-optimization.md")
    print("   3. 💼 For Managers: Focus on strategic chapters and case studies")
    print("   4. 💻 For Technical: Run sample code: python code/chapter1_ai_operations.py")
    print("   5. 📊 Explore notebooks: jupyter notebook notebooks/")
    print("   6. 🏋️  Complete exercises: exercises/")
    print()
    print("📚 Learning Path:")
    print("   • For Executives: Start with Executive Summary and strategic chapters")
    print("   • For Managers: Focus on Chapters 1-4, 9-12 (implementation focus)")
    print("   • For Students: Follow complete learning path Chapters 1-12")
    print("   • For Technical: All chapters with expanded code sections")
    print()
    print("🚀 Happy learning and building AI-enhanced operational excellence!")

def main():
    """Main setup function"""
    print_banner()
    check_python_version()
    install_requirements()
    create_directories()
    create_sample_data()
    create_sample_code()
    run_tests()
    print_next_steps()

if __name__ == "__main__":
    main()
