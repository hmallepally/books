import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.patterns.windowing import length_of_longest_substring_k_distinct, subarray_sum_equals_k
from src.aiml.evaluation import ConfusionMatrix, calculate_ndcg_at_k

def test_longest_substring_k_distinct():
    assert length_of_longest_substring_k_distinct("eceba", 2) == 3 # "ece"
    assert length_of_longest_substring_k_distinct("aa", 1) == 2
    assert length_of_longest_substring_k_distinct("", 2) == 0

def test_subarray_sum_equals_k():
    assert subarray_sum_equals_k([1, 1, 1], 2) == 2
    assert subarray_sum_equals_k([1, -1, 0], 0) == 3 # [1, -1], [0], [1, -1, 0]

def test_confusion_matrix():
    actuals     = [1, 1, 0, 0, 1, 0, 1, 0]
    predictions = [1, 0, 0, 1, 1, 0, 1, 0]
    cm = ConfusionMatrix(actuals, predictions)
    assert cm.tp == 3
    assert cm.fn == 1
    assert cm.fp == 1
    assert cm.tn == 3
    assert pytest.approx(cm.precision, 0.01) == 0.75
    assert pytest.approx(cm.recall, 0.01) == 0.75
    assert pytest.approx(cm.f1_score, 0.01) == 0.75

def test_ndcg():
    scores = [3.0, 2.0, 3.0, 0.0, 1.0, 2.0]
    ndcg_3 = calculate_ndcg_at_k(scores, 3)
    assert 0.0 <= ndcg_3 <= 1.0
