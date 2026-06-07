from enum import Enum
from dataclasses import dataclass

class ReviewStatus(Enum):
    APPROVED = "APPROVED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECTED = "REJECTED"

@dataclass
class TransactionEval:
    transaction_id: str
    status: ReviewStatus
    reason: str

class FraudAnalyzer:
    def evaluate(self, transaction_id: str, amount: float, risk_score: int) -> TransactionEval:
        # SDSD Invariant: High Value Hard Gate
        if amount > 10000.00:
            return TransactionEval(transaction_id, ReviewStatus.MANUAL_REVIEW, "Amount exceeds $10,000 threshold.")
            
        # Standard rules
        if risk_score > 80:
            return TransactionEval(transaction_id, ReviewStatus.REJECTED, "Risk score too high.")
            
        return TransactionEval(transaction_id, ReviewStatus.APPROVED, "Clear.")
