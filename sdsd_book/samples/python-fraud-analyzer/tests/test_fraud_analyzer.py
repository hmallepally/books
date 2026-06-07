import pytest
from fraud_analyzer import FraudAnalyzer, ReviewStatus

def test_standard_approval():
    analyzer = FraudAnalyzer()
    result = analyzer.evaluate("TX-1", 500.0, 10)
    assert result.status == ReviewStatus.APPROVED

def test_high_risk_rejection():
    analyzer = FraudAnalyzer()
    result = analyzer.evaluate("TX-2", 500.0, 95)
    assert result.status == ReviewStatus.REJECTED

def test_invariant_high_value_hard_gate():
    analyzer = FraudAnalyzer()
    
    # Even if risk score is perfect (0), over 10k must go to manual review
    result1 = analyzer.evaluate("TX-3", 10000.01, 0)
    assert result1.status == ReviewStatus.MANUAL_REVIEW
    assert "exceeds" in result1.reason
    
    # Boundary test: exactly 10k is fine (not strictly greater)
    result2 = analyzer.evaluate("TX-4", 10000.00, 0)
    assert result2.status == ReviewStatus.APPROVED
