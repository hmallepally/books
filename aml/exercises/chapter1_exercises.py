"""
Chapter 1 Exercises: Machine Learning Basics
===========================================

Practice exercises for fundamental machine learning concepts including:
- Types of machine learning
- Basic ML workflow
- Data preprocessing
- Model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any


class Chapter1Exercises:
    """Exercise problems for Chapter 1."""
    
    def __init__(self):
        pass
    
    def exercise_1_1_ml_types_classification(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Exercise 1.1: Implement supervised learning classification.
        
        Requirements:
        - Split data into training and test sets
        - Train a logistic regression model
        - Evaluate using accuracy score
        - Return dictionary with training and test accuracy
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary with 'train_accuracy' and 'test_accuracy'
        """
        # TODO: Implement your classification solution here
        # Replace this with your implementation
        pass
    
    def exercise_1_2_ml_types_regression(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Exercise 1.2: Implement supervised learning regression.
        
        Requirements:
        - Split data into training and test sets
        - Train a linear regression model
        - Evaluate using MSE and R² score
        - Return dictionary with metrics
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with 'train_mse', 'test_mse', 'train_r2', 'test_r2'
        """
        # TODO: Implement your regression solution here
        # Replace this with your implementation
        pass
    
    def exercise_1_3_ml_types_clustering(self, X: np.ndarray, n_clusters: int = 3) -> Dict[str, Any]:
        """
        Exercise 1.3: Implement unsupervised learning clustering.
        
        Requirements:
        - Apply K-means clustering
        - Evaluate using silhouette score
        - Return cluster centers and labels
        - Return evaluation metrics
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results and metrics
        """
        # TODO: Implement your clustering solution here
        # Replace this with your implementation
        pass
    
    def exercise_1_4_ml_workflow(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Exercise 1.4: Implement complete ML workflow.
        
        Requirements:
        - Data preprocessing (scaling)
        - Train/test split
        - Model training
        - Model evaluation
        - Feature importance analysis
        - Return complete workflow results
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary with complete workflow results
        """
        # TODO: Implement your ML workflow here
        # Replace this with your implementation
        pass
    
    def exercise_1_5_data_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Exercise 1.5: Implement data preprocessing pipeline.
        
        Requirements:
        - Handle missing values
        - Scale numerical features
        - Encode categorical variables
        - Remove outliers (optional)
        - Return preprocessed data
        
        Args:
            data: Raw DataFrame with mixed data types
            
        Returns:
            Preprocessed DataFrame
        """
        # TODO: Implement your data preprocessing here
        # Replace this with your implementation
        pass
    
    def exercise_1_6_model_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    task_type: str = 'classification') -> Dict[str, float]:
        """
        Exercise 1.6: Implement comprehensive model evaluation.
        
        Requirements:
        - For classification: accuracy, precision, recall, F1-score
        - For regression: MSE, MAE, R² score
        - Handle both binary and multiclass classification
        - Return appropriate metrics based on task type
        
        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with evaluation metrics
        """
        # TODO: Implement your model evaluation here
        # Replace this with your implementation
        pass
    
    def exercise_1_7_learning_types_comparison(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Exercise 1.7: Compare different learning types on same dataset.
        
        Requirements:
        - Implement supervised learning (classification)
        - Implement semi-supervised learning (use subset of labels)
        - Implement unsupervised learning (clustering)
        - Compare performance across learning types
        - Return comparison metrics
        
        Args:
            X: Feature matrix
            y: Target labels (for supervised/semi-supervised)
            
        Returns:
            Dictionary with comparison results
        """
        # TODO: Implement your learning types comparison here
        # Replace this with your implementation
        pass
    
    def exercise_1_8_traditional_vs_ml(self, data: List[str]) -> Dict[str, Any]:
        """
        Exercise 1.8: Compare traditional programming vs ML approaches.
        
        Requirements:
        - Implement rule-based approach for text classification
        - Implement ML-based approach for same task
        - Compare accuracy and flexibility
        - Analyze pros and cons of each approach
        
        Args:
            data: List of text samples
            
        Returns:
            Dictionary with comparison results
        """
        # TODO: Implement your traditional vs ML comparison here
        # Replace this with your implementation
        pass


def run_exercises():
    """Run all exercises with sample data."""
    
    exercises = Chapter1Exercises()
    
    print("=== Chapter 1 Exercises ===\n")
    
    # Exercise 1.1: Classification
    print("Exercise 1.1 - ML Types Classification:")
    X_class, y_class = make_classification(
        n_samples=1000, n_features=2, n_informative=2, 
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    try:
        results = exercises.exercise_1_1_ml_types_classification(X_class, y_class)
        print(f"✓ Classification completed")
        print(f"✓ Training accuracy: {results.get('train_accuracy', 0):.3f}")
        print(f"✓ Test accuracy: {results.get('test_accuracy', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.2: Regression
    print("Exercise 1.2 - ML Types Regression:")
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=1, noise=10, random_state=42
    )
    
    try:
        results = exercises.exercise_1_2_ml_types_regression(X_reg, y_reg)
        print(f"✓ Regression completed")
        print(f"✓ Test MSE: {results.get('test_mse', 0):.3f}")
        print(f"✓ Test R²: {results.get('test_r2', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.3: Clustering
    print("Exercise 1.3 - ML Types Clustering:")
    X_cluster, _ = make_blobs(
        n_samples=300, centers=4, cluster_std=0.60, random_state=42
    )
    
    try:
        results = exercises.exercise_1_3_ml_types_clustering(X_cluster, n_clusters=4)
        print(f"✓ Clustering completed")
        print(f"✓ Silhouette score: {results.get('silhouette_score', 0):.3f}")
        print(f"✓ Number of clusters: {len(results.get('cluster_centers', []))}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.4: ML Workflow
    print("Exercise 1.4 - ML Workflow:")
    
    try:
        results = exercises.exercise_1_4_ml_workflow(X_class, y_class)
        print(f"✓ ML workflow completed")
        print(f"✓ Test accuracy: {results.get('test_accuracy', 0):.3f}")
        print(f"✓ Feature importance calculated: {len(results.get('feature_importance', []))} features")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.5: Data Preprocessing
    print("Exercise 1.5 - Data Preprocessing:")
    # Create sample data with missing values and mixed types
    data = pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],
        'categorical': ['A', 'B', 'A', np.nan, 'B'],
        'text': ['hello', 'world', 'hello', 'world', 'hello']
    })
    
    try:
        preprocessed = exercises.exercise_1_5_data_preprocessing(data)
        print(f"✓ Data preprocessing completed")
        print(f"✓ Original shape: {data.shape}")
        print(f"✓ Preprocessed shape: {preprocessed.shape}")
        print(f"✓ Missing values: {preprocessed.isnull().sum().sum()}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.6: Model Evaluation
    print("Exercise 1.6 - Model Evaluation:")
    # Create sample predictions
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    
    try:
        results = exercises.exercise_1_6_model_evaluation(y_true, y_pred, 'classification')
        print(f"✓ Model evaluation completed")
        print(f"✓ Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"✓ Precision: {results.get('precision', 0):.3f}")
        print(f"✓ Recall: {results.get('recall', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.7: Learning Types Comparison
    print("Exercise 1.7 - Learning Types Comparison:")
    
    try:
        results = exercises.exercise_1_7_learning_types_comparison(X_class, y_class)
        print(f"✓ Learning types comparison completed")
        print(f"✓ Supervised: {results.get('supervised', 0):.3f}")
        print(f"✓ Semi-supervised: {results.get('semi_supervised', 0):.3f}")
        print(f"✓ Unsupervised: {results.get('unsupervised', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")
    
    # Exercise 1.8: Traditional vs ML
    print("Exercise 1.8 - Traditional vs ML:")
    sample_texts = [
        "Hello, how are you?",
        "FREE MONEY! You're a WINNER!",
        "Meeting tomorrow at 3 PM",
        "URGENT: Claim your prize now!"
    ]
    
    try:
        results = exercises.exercise_1_8_traditional_vs_ml(sample_texts)
        print(f"✓ Traditional vs ML comparison completed")
        print(f"✓ Traditional accuracy: {results.get('traditional_accuracy', 0):.3f}")
        print(f"✓ ML accuracy: {results.get('ml_accuracy', 0):.3f}")
    except NotImplementedError:
        print("Not implemented yet")
    print("\n" + "-"*40 + "\n")


def provide_solutions():
    """Provide solution hints and examples."""
    
    print("=== Exercise Solutions and Hints ===\n")
    
    print("Exercise 1.1 - ML Types Classification Hints:")
    print("- Use train_test_split for data splitting")
    print("- Use LogisticRegression for classification")
    print("- Use accuracy_score for evaluation")
    print("- Return both training and test accuracy")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.2 - ML Types Regression Hints:")
    print("- Use LinearRegression for regression")
    print("- Use mean_squared_error and r2_score for evaluation")
    print("- Compare training and test performance")
    print("- Handle feature scaling if needed")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.3 - ML Types Clustering Hints:")
    print("- Use KMeans for clustering")
    print("- Use silhouette_score for evaluation")
    print("- Return cluster centers and labels")
    print("- Consider different numbers of clusters")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.4 - ML Workflow Hints:")
    print("- Follow the standard ML workflow steps")
    print("- Include data preprocessing, splitting, training, evaluation")
    print("- Use feature importance for interpretation")
    print("- Document each step clearly")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.5 - Data Preprocessing Hints:")
    print("- Handle missing values with fillna or dropna")
    print("- Use StandardScaler for numerical features")
    print("- Use LabelEncoder or OneHotEncoder for categorical features")
    print("- Consider outlier detection and removal")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.6 - Model Evaluation Hints:")
    print("- Use appropriate metrics for task type")
    print("- For classification: accuracy, precision, recall, F1")
    print("- For regression: MSE, MAE, R²")
    print("- Handle edge cases (empty predictions, etc.)")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.7 - Learning Types Comparison Hints:")
    print("- Implement supervised learning with full labels")
    print("- Implement semi-supervised with subset of labels")
    print("- Implement unsupervised clustering")
    print("- Use appropriate evaluation metrics for each")
    print("\n" + "-"*40 + "\n")
    
    print("Exercise 1.8 - Traditional vs ML Hints:")
    print("- Create rule-based text classifier")
    print("- Implement ML-based text classifier")
    print("- Compare accuracy and flexibility")
    print("- Analyze pros and cons of each approach")
    print("\n" + "-"*40 + "\n")


def demonstrate_ml_concepts():
    """Demonstrate key ML concepts for reference."""
    
    print("=== ML Concepts Demonstration ===\n")
    
    # Generate sample data
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, 
        n_redundant=0, n_clusters_per_class=1, random_state=42
    )
    
    print("Sample Data:")
    print(f"- Features: {X.shape[1]}")
    print(f"- Samples: {X.shape[0]}")
    print(f"- Classes: {len(np.unique(y))}")
    
    # Demonstrate basic workflow
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nBasic ML Workflow Results:")
    print(f"- Training accuracy: {train_score:.3f}")
    print(f"- Test accuracy: {test_score:.3f}")
    print(f"- Model parameters: {model.coef_.shape}")


if __name__ == "__main__":
    # Run exercises
    run_exercises()
    
    # Provide solution hints
    provide_solutions()
    
    # Demonstrate concepts
    demonstrate_ml_concepts() 