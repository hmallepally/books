"""
Lab 6.1 — AI Alert Classifier
Book: The Ethical Hacker's Playbook, Chapter 6

Build a machine learning classifier that triages SOC alerts,
distinguishing true positives from false positives.

Usage:
    python alert_classifier.py
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────
# Stage 1: Generate Synthetic SOC Alert Data
# ──────────────────────────────────────────────────────────

def generate_alert_data(n_alerts=500):
    """
    Generate realistic synthetic SOC alert data.
    
    In a real SOC, alerts come from SIEM platforms (Splunk, Sentinel, 
    Chronicle) and EDR tools (CrowdStrike, SentinelOne). For this lab,
    we generate synthetic data that mimics real alert patterns.
    
    Learning: The ratio of false positives to true positives in real SOCs
    is typically 80-90% false positives. Our data reflects this imbalance,
    which is critical for building a realistic classifier.
    """
    random.seed(42)
    np.random.seed(42)
    
    alert_types = [
        "brute_force_login", "port_scan", "malware_download",
        "phishing_email", "data_exfiltration", "privilege_escalation",
        "lateral_movement", "dns_tunneling", "suspicious_powershell",
        "failed_mfa",
    ]
    
    source_ips = [f"10.0.{random.randint(1,254)}.{random.randint(1,254)}" 
                  for _ in range(50)]
    external_ips = [f"{random.randint(1,223)}.{random.randint(0,255)}"
                    f".{random.randint(0,255)}.{random.randint(1,254)}" 
                    for _ in range(30)]
    
    users = [f"user_{i:03d}" for i in range(100)]
    
    alerts = []
    base_time = datetime(2025, 3, 15, 8, 0, 0)
    
    for i in range(n_alerts):
        alert_type = random.choice(alert_types)
        hour = random.randint(0, 23)
        is_business_hours = 8 <= hour <= 18
        source_ip = random.choice(source_ips + external_ips)
        is_internal = source_ip.startswith("10.")
        user = random.choice(users)
        
        # Generate features that correlate with true/false positive
        repeat_count = random.randint(1, 50)
        
        # True positive signals (realistic patterns)
        is_true_positive = False
        
        if alert_type in ("malware_download", "data_exfiltration", 
                          "dns_tunneling"):
            # These alert types have higher true positive rates
            is_true_positive = random.random() < 0.6
        elif alert_type == "brute_force_login" and repeat_count > 20:
            # High-volume brute force is likely real
            is_true_positive = random.random() < 0.7
        elif alert_type == "suspicious_powershell" and not is_business_hours:
            # PowerShell at 3am is suspicious
            is_true_positive = random.random() < 0.65
        elif alert_type == "privilege_escalation" and not is_internal:
            # External privilege escalation attempts
            is_true_positive = random.random() < 0.5
        else:
            # Most other alerts are false positives (reflects reality)
            is_true_positive = random.random() < 0.15
        
        timestamp = base_time + timedelta(hours=i * 0.5)
        
        alerts.append({
            "alert_id": f"ALT-{i+1:05d}",
            "timestamp": timestamp.isoformat(),
            "alert_type": alert_type,
            "source_ip": source_ip,
            "is_internal_ip": is_internal,
            "user": user,
            "hour_of_day": hour,
            "is_business_hours": is_business_hours,
            "repeat_count": repeat_count,
            "severity_score": random.uniform(1, 10),
            "is_true_positive": is_true_positive,
        })
    
    return pd.DataFrame(alerts)


# ──────────────────────────────────────────────────────────
# Stage 2: Feature Engineering
# ──────────────────────────────────────────────────────────

def engineer_features(df):
    """
    Extract meaningful features from raw alert data.
    
    Feature engineering is where domain expertise meets machine learning.
    A SOC analyst knows that PowerShell at 3am is more suspicious than 
    PowerShell at 2pm. We encode that knowledge into features.
    
    Learning: The features you choose matter more than the algorithm.
    A simple Random Forest with great features beats a complex neural 
    network with bad features every time.
    """
    features = pd.DataFrame()
    
    # Encode alert type as numeric (one-hot would be better for production)
    alert_type_map = {t: i for i, t in enumerate(
        df["alert_type"].unique()
    )}
    features["alert_type_encoded"] = df["alert_type"].map(alert_type_map)
    
    # Time features
    features["hour_of_day"] = df["hour_of_day"]
    features["is_business_hours"] = df["is_business_hours"].astype(int)
    
    # Network features
    features["is_internal_ip"] = df["is_internal_ip"].astype(int)
    
    # Behavioral features
    features["repeat_count"] = df["repeat_count"]
    features["severity_score"] = df["severity_score"]
    
    # Derived features
    features["is_high_volume"] = (df["repeat_count"] > 15).astype(int)
    features["is_off_hours"] = (~df["is_business_hours"]).astype(int)
    features["is_high_severity"] = (df["severity_score"] > 7).astype(int)
    
    return features


# ──────────────────────────────────────────────────────────
# Stage 3: Train the Classifier
# ──────────────────────────────────────────────────────────

def train_classifier(features, labels):
    """
    Train a Random Forest classifier.
    
    Random Forest is an ensemble method that builds multiple decision trees
    and lets them vote. It's ideal for SOC alert classification because:
    - Handles imbalanced data well (with class_weight='balanced')
    - Provides feature importance (tells you WHY it classified)
    - Resistant to overfitting
    - No feature scaling required
    
    Learning: In production, you'd retrain this weekly as new ground truth
    labels come in from analyst investigations. The model improves as 
    your team validates its classifications.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, 
        stratify=labels
    )
    
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",  # Handle imbalanced data
        random_state=42,
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    
    print("\n📊 Model Performance:")
    print("=" * 50)
    print(classification_report(
        y_test, y_pred, 
        target_names=["False Positive", "True Positive"]
    ))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives  (correct FP):  {cm[0][0]}")
    print(f"  False Positives (missed FP):   {cm[0][1]}")
    print(f"  False Negatives (missed TP):   {cm[1][0]}")
    print(f"  True Positives  (correct TP):  {cm[1][1]}")
    
    f1 = f1_score(y_test, y_pred)
    print(f"\n  F1 Score: {f1:.3f}")
    
    # Feature importance
    importance = pd.Series(
        clf.feature_importances_, 
        index=features.columns
    ).sort_values(ascending=False)
    
    print("\n📈 Feature Importance:")
    for feat, imp in importance.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s} {imp:.3f} {bar}")
    
    return clf, X_test, y_test


# ──────────────────────────────────────────────────────────
# Stage 4: LLM Explanation Prompt
# ──────────────────────────────────────────────────────────

def generate_explanation_prompt(alert, prediction, probability):
    """
    Generate an LLM prompt to explain a classification.
    
    AI classification is powerful, but analysts need to understand WHY.
    LLMs can generate natural-language explanations that help analysts 
    trust and learn from the ML model.
    
    Learning: This is the "human-AI collaboration" pattern. The ML model
    makes the fast decision (milliseconds). The LLM explains it (seconds).
    The analyst validates it (minutes instead of hours).
    """
    prompt = f"""You are a SOC analyst AI assistant. Explain the following 
alert classification and recommend response actions.

ALERT DETAILS:
- ID: {alert.get('alert_id', 'N/A')}
- Type: {alert.get('alert_type', 'N/A')}
- Source IP: {alert.get('source_ip', 'N/A')} ({'internal' if alert.get('is_internal_ip') else 'external'})
- Time: {alert.get('timestamp', 'N/A')} ({'business hours' if alert.get('is_business_hours') else 'off-hours'})
- Repeat Count: {alert.get('repeat_count', 'N/A')}
- Severity Score: {alert.get('severity_score', 'N/A'):.1f}/10

ML CLASSIFICATION: {'TRUE POSITIVE (likely threat)' if prediction else 'FALSE POSITIVE (likely benign)'}
CONFIDENCE: {probability:.1%}

TASKS:
1. Explain in 2-3 sentences WHY this alert was classified this way
2. If TRUE POSITIVE: recommend specific containment + response actions
3. If FALSE POSITIVE: suggest a tuning rule to reduce similar noise
4. Rate urgency: Immediate / Within 1 hour / Within 4 hours / Can wait"""

    return prompt


# ──────────────────────────────────────────────────────────
# Main Lab
# ──────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🛡️  AI Alert Classifier — Lab 6.1")
    print("   The Ethical Hacker's Playbook")
    print("=" * 60)
    
    # Stage 1: Generate data
    print("\n[Stage 1] Generating synthetic SOC alert data...")
    df = generate_alert_data(n_alerts=500)
    tp_count = df["is_true_positive"].sum()
    fp_count = len(df) - tp_count
    print(f"  Generated {len(df)} alerts: {tp_count} true positives, "
          f"{fp_count} false positives ({fp_count/len(df)*100:.0f}% FP rate)")
    
    # Stage 2: Feature engineering
    print("\n[Stage 2] Engineering features...")
    features = engineer_features(df)
    labels = df["is_true_positive"].astype(int)
    print(f"  Extracted {features.shape[1]} features")
    
    # Stage 3: Train classifier
    print("\n[Stage 3] Training Random Forest classifier...")
    clf, X_test, y_test = train_classifier(features, labels)
    
    # Stage 4: Demonstrate AI explanation
    print("\n[Stage 4] Generating AI explanations for sample alerts...")
    print("=" * 50)
    
    # Pick 3 interesting alerts to explain
    sample_indices = random.sample(range(len(df)), 3)
    
    for idx in sample_indices:
        alert = df.iloc[idx].to_dict()
        alert_features = features.iloc[[idx]]
        prediction = clf.predict(alert_features)[0]
        probability = clf.predict_proba(alert_features)[0][prediction]
        
        label = "🔴 TRUE POSITIVE" if prediction else "🟢 FALSE POSITIVE"
        print(f"\n{'─' * 50}")
        print(f"Alert {alert['alert_id']}: {alert['alert_type']}")
        print(f"  Classification: {label} ({probability:.1%} confidence)")
        
        prompt = generate_explanation_prompt(alert, prediction, probability)
        print(f"\n📋 LLM Explanation Prompt:")
        print(prompt)
    
    print(f"\n{'=' * 60}")
    print("✅ Lab complete! Key takeaways:")
    print("  1. Even simple ML can achieve 85%+ accuracy on alert triage")
    print("  2. Feature engineering matters more than algorithm choice")
    print("  3. LLM explanations build analyst trust in AI decisions")
    print("  4. False positive reduction = analyst time saved")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
