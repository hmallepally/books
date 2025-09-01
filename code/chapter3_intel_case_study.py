"""
Chapter 3: Strategic AI Implementation - Intel Case Study
Code examples demonstrating Intel's AI transformation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import cv2
import numpy as np
from tensorflow import keras
import pulp
import pandas as pd

class IntelPredictiveMaintenance:
    """Intel's Predictive Maintenance System"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_equipment_data(self, filepath):
        """Load equipment sensor data"""
        return pd.read_csv(filepath)
    
    def preprocess_data(self, data):
        """Preprocess equipment data"""
        # Extract features
        features = ['temperature', 'pressure', 'vibration', 'current', 'voltage']
        X = data[features]
        y = data['maintenance_needed']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, X, y):
        """Train predictive maintenance model"""
        self.model.fit(X, y)
        self.is_trained = True
        return self.model.score(X, y)
    
    def predict_maintenance(self, equipment_data):
        """Predict maintenance needs"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(equipment_data)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'maintenance_priority': self.calculate_priority(probabilities)
        }
    
    def calculate_priority(self, probabilities):
        """Calculate maintenance priority"""
        return np.argmax(probabilities, axis=1)

class IntelQualityControl:
    """Intel's Quality Control System"""
    
    def __init__(self):
        self.model = keras.Sequential([
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def preprocess_image(self, image):
        """Preprocess manufacturing images"""
        # Resize image
        resized = cv2.resize(image, (64, 64))
        
        # Normalize
        normalized = resized / 255.0
        
        return normalized
    
    def detect_defects(self, image):
        """Detect defects in manufacturing images"""
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(np.expand_dims(processed_image, axis=0))
        
        return {
            'defect_probability': prediction[0][0],
            'is_defective': prediction[0][0] > 0.5,
            'confidence': max(prediction[0][0], 1 - prediction[0][0])
        }
    
    def analyze_quality_trends(self, historical_data):
        """Analyze quality trends over time"""
        trends = {
            'defect_rate_trend': self.calculate_trend(historical_data['defect_rate']),
            'quality_score_trend': self.calculate_trend(historical_data['quality_score']),
            'improvement_areas': self.identify_improvement_areas(historical_data)
        }
        return trends
    
    def calculate_trend(self, data):
        """Calculate trend in quality metrics"""
        return np.polyfit(range(len(data)), data, 1)[0]

class IntelSupplyChainOptimization:
    """Intel's Supply Chain Optimization System"""
    
    def __init__(self):
        self.optimizer = pulp.LpProblem("Supply_Chain_Optimization", pulp.LpMinimize)
        
    def optimize_inventory(self, demand_data, cost_data):
        """Optimize inventory levels"""
        # Create optimization variables
        inventory_vars = {}
        for product in demand_data['products']:
            inventory_vars[product] = pulp.LpVariable(f"inventory_{product}", 0, None)
        
        # Objective function: minimize total cost
        total_cost = pulp.lpSum([
            cost_data['holding_cost'][product] * inventory_vars[product]
            for product in demand_data['products']
        ])
        
        self.optimizer += total_cost
        
        # Constraints: meet demand
        for product in demand_data['products']:
            self.optimizer += inventory_vars[product] >= demand_data['demand'][product]
        
        # Solve optimization
        self.optimizer.solve()
        
        # Extract results
        optimal_inventory = {}
        for product in demand_data['products']:
            optimal_inventory[product] = inventory_vars[product].value()
        
        return {
            'optimal_inventory': optimal_inventory,
            'total_cost': pulp.value(self.optimizer.objective),
            'status': pulp.LpStatus[self.optimizer.status]
        }
    
    def optimize_supplier_selection(self, supplier_data, requirements):
        """Optimize supplier selection"""
        # Create optimization variables
        supplier_vars = {}
        for supplier in supplier_data['suppliers']:
            supplier_vars[supplier] = pulp.LpVariable(f"supplier_{supplier}", 0, 1, pulp.LpBinary)
        
        # Objective function: minimize total cost
        total_cost = pulp.lpSum([
            supplier_data['cost'][supplier] * supplier_vars[supplier]
            for supplier in supplier_data['suppliers']
        ])
        
        self.optimizer += total_cost
        
        # Constraints: meet requirements
        for requirement in requirements:
            self.optimizer += pulp.lpSum([
                supplier_data[requirement][supplier] * supplier_vars[supplier]
                for supplier in supplier_data['suppliers']
            ]) >= requirements[requirement]
        
        # Solve optimization
        self.optimizer.solve()
        
        # Extract results
        selected_suppliers = []
        for supplier in supplier_data['suppliers']:
            if supplier_vars[supplier].value() == 1:
                selected_suppliers.append(supplier)
        
        return {
            'selected_suppliers': selected_suppliers,
            'total_cost': pulp.value(self.optimizer.objective),
            'status': pulp.LpStatus[self.optimizer.status]
        }

def main():
    """Main function demonstrating Intel's AI systems"""
    print("Intel AI Transformation Case Study")
    print("=" * 50)
    
    # Initialize systems
    maintenance_system = IntelPredictiveMaintenance()
    quality_system = IntelQualityControl()
    supply_chain_system = IntelSupplyChainOptimization()
    
    print("✅ AI systems initialized successfully")
    print("📊 Ready for Intel's AI-enhanced operational excellence!")
    
    return {
        'maintenance': maintenance_system,
        'quality': quality_system,
        'supply_chain': supply_chain_system
    }

if __name__ == "__main__":
    main()
