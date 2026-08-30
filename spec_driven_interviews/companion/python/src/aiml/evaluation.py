"""
Module: Machine Learning Evaluation Metrics & Confusion Matrix
Corresponds to Chapter 23.
"""
import math
from typing import List

class ConfusionMatrix:
    def __init__(self, actuals: List[int], predictions: List[int]):
        assert len(actuals) == len(predictions), "Actuals and predictions must have identical length."
        self.tp = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 1)
        self.fp = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 1)
        self.fn = sum(1 for a, p in zip(actuals, predictions) if a == 1 and p == 0)
        self.tn = sum(1 for a, p in zip(actuals, predictions) if a == 0 and p == 0)
        
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        
    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        
    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
    def f_beta_score(self, beta: float = 2.0) -> float:
        p, r = self.precision, self.recall
        beta_sq = beta ** 2
        return (1 + beta_sq) * (p * r) / ((beta_sq * p) + r) if ((beta_sq * p) + r) > 0 else 0.0


def calculate_ndcg_at_k(relevance_scores: List[float], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at rank K (NDCG@K).
    """
    k = min(k, len(relevance_scores))
    if k == 0:
        return 0.0
        
    dcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
    
    return dcg / idcg if idcg > 0 else 0.0
