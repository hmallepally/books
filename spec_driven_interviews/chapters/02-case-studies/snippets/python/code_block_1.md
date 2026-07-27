```python
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from uuid import UUID

@dataclass(frozen=True)
class TransactionRecord:
    """
    Represents an immutable, validated financial transaction record in AuraPay.
    Enforces pre-conditions on initialization.
    """
    transaction_id: UUID
    source_account_id: UUID
    destination_account_id: UUID
    amount: Decimal
    currency: str
    timestamp: datetime

    def __post_init__(self):
        if not self.transaction_id or not self.source_account_id or not self.destination_account_id:
            raise ValueError("Account IDs and Transaction ID cannot be null")
        if not self.amount or not self.currency or not self.timestamp:
            raise ValueError("Amount, currency, and timestamp cannot be null")
        if self.source_account_id == self.destination_account_id:
            raise ValueError("Source and destination accounts must be distinct")
        if self.amount <= 0:
            raise ValueError("Transaction amount must be strictly positive")
        if not self.currency.strip():
            raise ValueError("Currency code cannot be empty")
```
