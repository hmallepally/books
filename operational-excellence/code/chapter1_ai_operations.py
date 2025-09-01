"""
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
    print("AI Operations Framework Demo")
    
    # Initialize framework
    framework = AIOperationsFramework()
    
    # Load sample data
    try:
        data = framework.load_data("data/sample_operational_data.csv")
        print(f"Loaded data with {len(data)} records")
        
        # Example preprocessing and training
        print("Data loaded successfully!")
        print("Ready for AI-enhanced operational excellence!")
        
    except FileNotFoundError:
        print("Sample data not found. Please ensure data files are created.")
    
    print("\nReady to explore AI-enhanced operational excellence!")

if __name__ == "__main__":
    main()
