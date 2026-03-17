# Lab 6: AI Alert Classifier

**Book Reference**: Chapter 6 — *The Blue Team: AI-Powered Defense*

---

## Purpose

Build a machine learning classifier that distinguishes true positive SOC alerts from false positives. Then use an LLM to explain classifications and recommend response actions.

## What You'll Learn

1. **Data Generation** — Create realistic synthetic SOC alert data
2. **Feature Engineering** — Extract meaningful features from security alerts
3. **ML Model Training** — Train a Random Forest classifier with scikit-learn
4. **Evaluation** — Measure precision, recall, and false positive rate
5. **AI Enhancement** — Use LLMs to explain WHY an alert is a true/false positive

## The Key Insight

> *"With a few hundred training examples, even a simple ML model can achieve 85%+ accuracy in distinguishing true from false positive alerts. Scale that to production quality and you've just saved your analysts 8,000 false positive investigations per day."*

## Architecture

```
  SOC Alert Feed
       │
       ▼
  ┌─────────────────┐
  │ Feature Extract  │  Source IP, destination, time, alert type,
  │                  │  user context, frequency
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Random Forest   │  Trained on labeled true/false positive data
  │  Classifier      │  → Probability score + classification
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  LLM Explainer  │  "Why is this alert a true positive?"
  │  (Prompt)       │  → Response action recommendations
  └─────────────────┘
```

## Requirements

```
pip install scikit-learn pandas numpy
```

## Usage

```bash
# Train the model and classify sample alerts
python alert_classifier.py

# Output: model performance metrics + classified alerts + LLM prompt
```

## Key Metrics to Watch

| Metric | What It Means | Target |
|--------|--------------|--------|
| **Precision** | When we say "true positive," how often are we right? | > 85% |
| **Recall** | Of all actual threats, how many did we catch? | > 90% |
| **False Positive Rate** | How many benign alerts did we escalate? | < 15% |
| **F1 Score** | Balance of precision and recall | > 85% |
