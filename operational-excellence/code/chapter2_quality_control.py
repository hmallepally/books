"""
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
    print("AI Quality Control System Demo")
    
    # Initialize quality control system
    qc_system = AIQualityControl()
    
    # Load quality metrics
    try:
        quality_data = pd.read_csv("data/quality_metrics.csv")
        print(f"Loaded quality data with {len(quality_data)} records")
        
        # Example quality analysis
        print("Quality control system ready!")
        print("Ready for AI-enhanced quality management!")
        
    except FileNotFoundError:
        print("Quality data not found. Please ensure data files are created.")
    
    print("\nReady to explore AI-enhanced quality control!")

if __name__ == "__main__":
    main()
