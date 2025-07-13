"""
Chapter 1: Machine Learning Basics
=================================

This module demonstrates fundamental machine learning concepts including:
- Types of machine learning
- Basic ML workflow
- Simple implementations of different learning types
- Data preprocessing basics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class MLBasics:
    """Demonstrates basic machine learning concepts and implementations."""
    
    def __init__(self):
        self.data = {}
        self.models = {}
        self.results = {}
    
    def demonstrate_supervised_learning(self):
        """Demonstrate supervised learning with classification and regression."""
        
        print("=== Supervised Learning Demonstration ===\n")
        
        # 1. Classification Example
        print("1. Classification Example:")
        print("-" * 30)
        
        # Generate classification data
        X_class, y_class = make_classification(
            n_samples=1000, n_features=2, n_informative=2, 
            n_redundant=0, n_clusters_per_class=1, random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        
        # Train logistic regression
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {accuracy:.3f}")
        
        # Store results
        self.data['classification'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        self.models['classification'] = clf
        self.results['classification_accuracy'] = accuracy
        
        # 2. Regression Example
        print("\n2. Regression Example:")
        print("-" * 30)
        
        # Generate regression data
        X_reg, y_reg = make_regression(
            n_samples=1000, n_features=1, noise=10, random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Train linear regression
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # Make predictions
        y_pred = reg.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"Regression MSE: {mse:.3f}")
        print(f"Regression R²: {reg.score(X_test, y_test):.3f}")
        
        # Store results
        self.data['regression'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }
        self.models['regression'] = reg
        self.results['regression_mse'] = mse
    
    def demonstrate_unsupervised_learning(self):
        """Demonstrate unsupervised learning with clustering."""
        
        print("\n=== Unsupervised Learning Demonstration ===\n")
        
        # Generate clustering data
        X_cluster, y_true = make_blobs(
            n_samples=300, centers=4, cluster_std=0.60, random_state=42
        )
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        y_pred = kmeans.fit_predict(X_cluster)
        
        # Evaluate clustering
        silhouette_avg = silhouette_score(X_cluster, y_pred)
        
        print(f"Number of clusters: {len(np.unique(y_pred))}")
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
        
        # Store results
        self.data['clustering'] = {
            'X': X_cluster, 'y_true': y_true, 'y_pred': y_pred
        }
        self.models['clustering'] = kmeans
        self.results['silhouette_score'] = silhouette_avg
    
    def demonstrate_ml_workflow(self):
        """Demonstrate a complete ML workflow."""
        
        print("\n=== Complete ML Workflow Demonstration ===\n")
        
        # 1. Data Generation
        print("1. Data Generation:")
        X, y = make_classification(
            n_samples=1000, n_features=5, n_informative=3, 
            n_redundant=1, n_clusters_per_class=1, random_state=42
        )
        print(f"   Generated {X.shape[0]} samples with {X.shape[1]} features")
        
        # 2. Data Preprocessing
        print("\n2. Data Preprocessing:")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"   Scaled features to mean=0, std=1")
        print(f"   Feature means: {X_scaled.mean(axis=0)[:3]}...")
        print(f"   Feature stds: {X_scaled.std(axis=0)[:3]}...")
        
        # 3. Data Splitting
        print("\n3. Data Splitting:")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"   Training set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # 4. Model Training
        print("\n4. Model Training:")
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        print(f"   Model trained successfully")
        print(f"   Model parameters: {model.coef_.shape}")
        
        # 5. Model Evaluation
        print("\n5. Model Evaluation:")
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"   Training accuracy: {train_score:.3f}")
        print(f"   Test accuracy: {test_score:.3f}")
        
        # 6. Model Interpretation
        print("\n6. Model Interpretation:")
        feature_importance = np.abs(model.coef_[0])
        print(f"   Feature importance: {feature_importance}")
        
        # Store workflow results
        self.data['workflow'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_scaled': X_scaled, 'scaler': scaler
        }
        self.models['workflow'] = model
        self.results['workflow'] = {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def demonstrate_learning_types_comparison(self):
        """Compare different types of learning on the same dataset."""
        
        print("\n=== Learning Types Comparison ===\n")
        
        # Generate data
        X, y = make_classification(
            n_samples=500, n_features=2, n_informative=2, 
            n_redundant=0, n_clusters_per_class=1, random_state=42
        )
        
        # Split for supervised learning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 1. Supervised Learning
        print("1. Supervised Learning (Classification):")
        clf = LogisticRegression(random_state=42)
        clf.fit(X_train, y_train)
        supervised_score = clf.score(X_test, y_test)
        print(f"   Accuracy: {supervised_score:.3f}")
        
        # 2. Unsupervised Learning (Clustering)
        print("\n2. Unsupervised Learning (Clustering):")
        kmeans = KMeans(n_clusters=2, random_state=42)
        y_cluster = kmeans.fit_predict(X)
        
        # Evaluate clustering quality
        silhouette_avg = silhouette_score(X, y_cluster)
        print(f"   Silhouette Score: {silhouette_avg:.3f}")
        
        # 3. Semi-supervised Learning (using some labels)
        print("\n3. Semi-supervised Learning:")
        # Use only 10% of labels for training
        n_labeled = int(0.1 * len(X_train))
        X_labeled = X_train[:n_labeled]
        y_labeled = y_train[:n_labeled]
        
        # Train on labeled data only
        clf_semi = LogisticRegression(random_state=42)
        clf_semi.fit(X_labeled, y_labeled)
        semi_score = clf_semi.score(X_test, y_test)
        print(f"   Accuracy (10% labels): {semi_score:.3f}")
        
        # Compare results
        print(f"\nComparison:")
        print(f"   Supervised (100% labels): {supervised_score:.3f}")
        print(f"   Semi-supervised (10% labels): {semi_score:.3f}")
        print(f"   Unsupervised (clustering quality): {silhouette_avg:.3f}")
        
        # Store comparison results
        self.results['comparison'] = {
            'supervised': supervised_score,
            'semi_supervised': semi_score,
            'unsupervised': silhouette_avg
        }
    
    def visualize_results(self):
        """Create visualizations of the ML results."""
        
        print("\n=== Creating Visualizations ===\n")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Classification Results
        if 'classification' in self.data:
            X_train = self.data['classification']['X_train']
            y_train = self.data['classification']['y_train']
            
            axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6)
            axes[0, 0].set_title(f'Classification Data\n(Accuracy: {self.results["classification_accuracy"]:.3f})')
            axes[0, 0].set_xlabel('Feature 1')
            axes[0, 0].set_ylabel('Feature 2')
        
        # 2. Regression Results
        if 'regression' in self.data:
            X_test = self.data['regression']['X_test']
            y_test = self.data['regression']['y_test']
            y_pred = self.models['regression'].predict(X_test)
            
            axes[0, 1].scatter(X_test, y_test, alpha=0.6, label='Actual')
            axes[0, 1].plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
            axes[0, 1].set_title(f'Regression Results\n(MSE: {self.results["regression_mse"]:.3f})')
            axes[0, 1].set_xlabel('Feature')
            axes[0, 1].set_ylabel('Target')
            axes[0, 1].legend()
        
        # 3. Clustering Results
        if 'clustering' in self.data:
            X = self.data['clustering']['X']
            y_pred = self.data['clustering']['y_pred']
            
            axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.6)
            axes[1, 0].set_title(f'Clustering Results\n(Silhouette: {self.results["silhouette_score"]:.3f})')
            axes[1, 0].set_xlabel('Feature 1')
            axes[1, 0].set_ylabel('Feature 2')
        
        # 4. Learning Types Comparison
        if 'comparison' in self.results:
            methods = ['Supervised', 'Semi-supervised', 'Unsupervised']
            scores = [
                self.results['comparison']['supervised'],
                self.results['comparison']['semi_supervised'],
                self.results['comparison']['unsupervised']
            ]
            
            bars = axes[1, 1].bar(methods, scores, color=['blue', 'green', 'orange'])
            axes[1, 1].set_title('Learning Types Comparison')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("Visualizations created successfully!")
    
    def demonstrate_traditional_vs_ml(self):
        """Demonstrate the difference between traditional programming and ML."""
        
        print("\n=== Traditional Programming vs Machine Learning ===\n")
        
        # Traditional Programming Example: Rule-based spam detection
        print("1. Traditional Programming (Rule-based):")
        print("-" * 50)
        
        def traditional_spam_detector(email_text):
            """Rule-based spam detection."""
            spam_keywords = ['free', 'money', 'winner', 'urgent', 'limited time']
            email_lower = email_text.lower()
            
            spam_score = 0
            for keyword in spam_keywords:
                if keyword in email_lower:
                    spam_score += 1
            
            return spam_score > 2  # If more than 2 keywords, classify as spam
        
        # Test traditional approach
        test_emails = [
            "Hello, how are you?",
            "FREE MONEY! You're a WINNER! Limited time offer!",
            "Meeting tomorrow at 3 PM",
            "URGENT: You've won $1000! Claim now!"
        ]
        
        print("Traditional Rule-based Results:")
        for email in test_emails:
            is_spam = traditional_spam_detector(email)
            print(f"   '{email[:30]}...' -> {'SPAM' if is_spam else 'NOT SPAM'}")
        
        # Machine Learning Example: Learning from data
        print("\n2. Machine Learning Approach:")
        print("-" * 50)
        
        # Simulate training data
        training_emails = [
            "Hello, how are you?",
            "Meeting tomorrow at 3 PM",
            "Project update for next week",
            "FREE MONEY! You're a WINNER!",
            "URGENT: Claim your prize now!",
            "Limited time offer!",
            "Team lunch on Friday",
            "CONGRATULATIONS! You've won!"
        ]
        
        training_labels = [0, 0, 0, 1, 1, 1, 0, 1]  # 0 = not spam, 1 = spam
        
        # Simple feature extraction
        def extract_features(email):
            """Extract simple features from email text."""
            email_lower = email.lower()
            features = [
                email_lower.count('free'),
                email_lower.count('money'),
                email_lower.count('winner'),
                email_lower.count('urgent'),
                email_lower.count('limited'),
                email_lower.count('congratulations'),
                len(email.split()),  # word count
                email_lower.count('!')  # exclamation marks
            ]
            return features
        
        # Extract features from training data
        X_train = np.array([extract_features(email) for email in training_emails])
        y_train = np.array(training_labels)
        
        # Train a simple ML model
        ml_model = LogisticRegression(random_state=42)
        ml_model.fit(X_train, y_train)
        
        # Test ML approach
        print("Machine Learning Results:")
        for email in test_emails:
            features = extract_features(email)
            prediction = ml_model.predict([features])[0]
            probability = ml_model.predict_proba([features])[0][1]  # spam probability
            print(f"   '{email[:30]}...' -> {'SPAM' if prediction else 'NOT SPAM'} (confidence: {probability:.3f})")
        
        print(f"\nKey Differences:")
        print(f"   Traditional: Fixed rules, no learning")
        print(f"   ML: Learns patterns from data, adapts to new examples")
        print(f"   Traditional: Manual rule creation")
        print(f"   ML: Automatic pattern discovery")


def run_demonstrations():
    """Run all demonstrations."""
    
    print("Machine Learning Basics Demonstrations")
    print("=" * 50)
    
    # Create ML basics instance
    ml_basics = MLBasics()
    
    # Run demonstrations
    ml_basics.demonstrate_supervised_learning()
    ml_basics.demonstrate_unsupervised_learning()
    ml_basics.demonstrate_ml_workflow()
    ml_basics.demonstrate_learning_types_comparison()
    ml_basics.demonstrate_traditional_vs_ml()
    
    # Create visualizations
    ml_basics.visualize_results()
    
    return ml_basics


def demonstrate_ml_concepts():
    """Demonstrate key ML concepts with simple examples."""
    
    print("\n" + "="*60)
    print("KEY MACHINE LEARNING CONCEPTS")
    print("="*60)
    
    # 1. Data Types in ML
    print("\n1. Data Types in Machine Learning:")
    print("-" * 40)
    
    # Numerical data
    numerical_data = np.array([1.5, 2.3, 3.1, 4.2, 5.0])
    print(f"   Numerical data: {numerical_data}")
    print(f"   Mean: {numerical_data.mean():.2f}")
    print(f"   Standard deviation: {numerical_data.std():.2f}")
    
    # Categorical data
    categorical_data = np.array(['red', 'blue', 'red', 'green', 'blue'])
    unique_values, counts = np.unique(categorical_data, return_counts=True)
    print(f"   Categorical data: {categorical_data}")
    print(f"   Unique values: {unique_values}")
    print(f"   Counts: {counts}")
    
    # 2. Feature Engineering
    print("\n2. Feature Engineering:")
    print("-" * 40)
    
    # Original features
    original_features = np.array([[1, 2], [3, 4], [5, 6]])
    print(f"   Original features:\n{original_features}")
    
    # Engineered features (polynomial)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False)
    engineered_features = poly.fit_transform(original_features)
    print(f"   Engineered features (polynomial):\n{engineered_features}")
    
    # 3. Model Complexity
    print("\n3. Model Complexity Trade-off:")
    print("-" * 40)
    
    # Generate data with noise
    X = np.linspace(0, 10, 100)
    y_true = 2 * X + 1
    y_noisy = y_true + np.random.normal(0, 1, 100)
    
    # Simple model (linear)
    from sklearn.linear_model import LinearRegression
    simple_model = LinearRegression()
    simple_model.fit(X.reshape(-1, 1), y_noisy)
    simple_score = simple_model.score(X.reshape(-1, 1), y_noisy)
    
    # Complex model (polynomial)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    complex_model = Pipeline([
        ('poly', PolynomialFeatures(degree=10)),
        ('linear', LinearRegression())
    ])
    complex_model.fit(X.reshape(-1, 1), y_noisy)
    complex_score = complex_model.score(X.reshape(-1, 1), y_noisy)
    
    print(f"   Simple model (linear) R²: {simple_score:.3f}")
    print(f"   Complex model (degree 10) R²: {complex_score:.3f}")
    print(f"   Note: Complex model may overfit to noise")


if __name__ == "__main__":
    # Run all demonstrations
    ml_basics = run_demonstrations()
    
    # Demonstrate key concepts
    demonstrate_ml_concepts()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Supervised learning uses labeled data for prediction")
    print("2. Unsupervised learning finds patterns without labels")
    print("3. ML workflow includes data preprocessing, training, and evaluation")
    print("4. Different learning types have different strengths")
    print("5. ML can learn patterns that are hard to encode as rules") 